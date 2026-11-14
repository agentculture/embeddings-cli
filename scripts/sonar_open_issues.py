#!/usr/bin/env python3
"""Fail when SonarCloud holds open issues for this pull request or branch.

    python3 scripts/sonar_open_issues.py --pull-request 12
    python3 scripts/sonar_open_issues.py --branch main

The quality gate only looks at new code against thresholds, so a pull request can pass it
while carrying open issues. This check is stricter: any issue whose status is Open or
Confirmed fails the run. An issue marked Accepted (or False positive, or Fixed) does not:
accepting an issue on SonarCloud is the way to keep it on purpose.

It reads the project key from ``sonar-project.properties`` and asks SonarCloud's public
web API for the issues of one analysis scope: a pull request (the issues its analysis
found in the changed code) or a branch (every issue on it). CI runs it straight after
the scan step, which waits for the analysis to finish (``sonar.qualitygate.wait``).

The token, if ``SONAR_TOKEN`` is set, is sent only to SonarCloud as a bearer header and
is never printed. Standard library only.

Exit codes: 0 no open issue, 1 open issues (each one is listed), 2 the check could not
be made (no project key, the request failed, or the answer was not understood).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HOST = "https://sonarcloud.io"
OPEN_STATUSES = frozenset({"OPEN", "CONFIRMED"})
PAGE_SIZE = 500
#: SonarCloud serves at most 10,000 results per search; stop there and say so.
MAX_RESULTS = 10_000


class CheckError(Exception):
    """The check could not be made (distinct from "open issues were found")."""


def project_key(properties: Path) -> str:
    """``sonar.projectKey`` from a ``sonar-project.properties`` file."""
    try:
        lines = properties.read_text(encoding="utf-8").splitlines()
    except OSError as err:
        raise CheckError(f"cannot read {properties}: {err}") from err
    for line in lines:
        name, sep, value = line.partition("=")
        if sep and name.strip() == "sonar.projectKey" and value.strip():
            return value.strip()
    raise CheckError(f"{properties} has no sonar.projectKey")


def fetch_json(url: str, token: str | None) -> dict:
    """GET one page of the web API. Only ever called with a SonarCloud https URL."""
    if not url.startswith(HOST + "/"):
        raise CheckError("refusing to request an address outside SonarCloud")
    request = urllib.request.Request(url, headers={"Accept": "application/json"})
    if token:
        request.add_header("Authorization", f"Bearer {token}")
    try:
        with urllib.request.urlopen(request, timeout=60) as response:  # nosec B310
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as err:
        raise CheckError(f"SonarCloud answered HTTP {err.code} for the issue search") from err
    except (urllib.error.URLError, TimeoutError, ValueError) as err:
        raise CheckError(f"could not read the issue search from SonarCloud: {err}") from err


def is_open(issue: dict) -> bool:
    """Open or Confirmed. Accepted, False positive and Fixed are not open."""
    status = issue.get("issueStatus") or issue.get("status") or ""
    return str(status).upper() in OPEN_STATUSES


def open_issues(key: str, scope: dict[str, str], token: str | None, fetch=fetch_json) -> list[dict]:
    """Every open issue of the project in ``scope`` (``pullRequest`` or ``branch``)."""
    found: list[dict] = []
    page = 1
    while True:
        query = {"componentKeys": key, "resolved": "false", "ps": str(PAGE_SIZE), "p": str(page)}
        query.update(scope)
        data = fetch(f"{HOST}/api/issues/search?{urllib.parse.urlencode(query)}", token)
        if not isinstance(data, dict) or "errors" in data or "issues" not in data:
            detail = data.get("errors") if isinstance(data, dict) else data
            raise CheckError(f"SonarCloud did not return an issue list: {detail}")
        issues = data["issues"]
        found.extend(issue for issue in issues if is_open(issue))
        total = int(data.get("total", data.get("paging", {}).get("total", 0)))
        if not issues or page * PAGE_SIZE >= min(total, MAX_RESULTS):
            if total > MAX_RESULTS:
                raise CheckError(f"more than {MAX_RESULTS} issues; the list is incomplete")
            return found
        page += 1


def describe(issue: dict) -> str:
    component = str(issue.get("component", ""))
    path = component.split(":", 1)[1] if ":" in component else component
    line = issue.get("line")
    where = f"{path}:{line}" if line else path
    return (
        f"{where}  {issue.get('rule', '?')}  "
        f"{str(issue.get('severity', '')).lower()}  {issue.get('message', '')}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    scope = parser.add_mutually_exclusive_group(required=True)
    scope.add_argument("--pull-request", metavar="<number>", help="check this pull request")
    scope.add_argument("--branch", metavar="<name>", help="check this branch")
    parser.add_argument(
        "--properties",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "sonar-project.properties",
        help="the sonar-project.properties to read the project key from",
    )
    args = parser.parse_args(argv)
    where = {"pullRequest": args.pull_request} if args.pull_request else {"branch": args.branch}
    label = f"pull request {args.pull_request}" if args.pull_request else f"branch {args.branch}"
    try:
        key = project_key(args.properties)
        issues = open_issues(key, where, os.environ.get("SONAR_TOKEN") or None)
    except CheckError as err:
        print(f"sonar-open-issues: could not check {label}: {err}", file=sys.stderr)
        return 2
    if not issues:
        print(f"sonar-open-issues: no open issue on {label} of {key}")
        return 0
    print(f"sonar-open-issues: {len(issues)} open issue(s) on {label} of {key}:", file=sys.stderr)
    for issue in issues:
        print("  " + describe(issue), file=sys.stderr)
    print(
        "fix each one, or mark it Accepted on SonarCloud if it is kept on purpose",
        file=sys.stderr,
    )
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
