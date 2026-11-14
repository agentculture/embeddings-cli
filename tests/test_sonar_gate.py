"""The CI check that fails on open SonarCloud issues (scripts/sonar_open_issues.py).

No request leaves the machine: the fetch function is replaced by a fake.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "sonar_open_issues.py"

spec = importlib.util.spec_from_file_location("sonar_open_issues", SCRIPT)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def issue(status: str, n: int = 1, legacy: bool = False) -> dict:
    record = {
        "rule": "python:S9073",
        "severity": "MAJOR",
        "component": "agentculture_demo:tests/test_x.py",
        "line": n,
        "message": f"message {n}",
    }
    record["status" if legacy else "issueStatus"] = status
    return record


def fake(pages: list[dict]):
    calls: list[tuple[str, str | None]] = []

    def fetch(url: str, token: str | None) -> dict:
        calls.append((url, token))
        return pages[min(len(calls), len(pages)) - 1]

    return fetch, calls


def test_project_key_is_read_from_this_repos_properties() -> None:
    assert gate.project_key(REPO_ROOT / "sonar-project.properties") == "agentculture_embeddings-cli"


def test_project_key_missing_is_a_check_error(tmp_path: Path) -> None:
    (tmp_path / "p").write_text("sonar.organization=x\n", encoding="utf-8")
    with pytest.raises(gate.CheckError):
        gate.project_key(tmp_path / "p")
    with pytest.raises(gate.CheckError):
        gate.project_key(tmp_path / "absent")


@pytest.mark.parametrize(
    "status,is_open",
    [
        ("OPEN", True),
        ("CONFIRMED", True),
        ("ACCEPTED", False),
        ("FALSE_POSITIVE", False),
        ("FIXED", False),
    ],
)
def test_only_open_and_confirmed_count(status: str, is_open: bool) -> None:
    assert gate.is_open(issue(status)) is is_open
    assert gate.is_open(issue(status, legacy=True)) is is_open


def test_scope_and_project_are_sent_and_accepted_issues_are_ignored() -> None:
    fetch, calls = fake([{"total": 3, "issues": [issue("OPEN", 1), issue("ACCEPTED", 2)]}])
    found = gate.open_issues("agentculture_demo", {"pullRequest": "12"}, "secret", fetch)
    assert [i["line"] for i in found] == [1]
    url, token = calls[0]
    assert url.startswith("https://sonarcloud.io/api/issues/search?")
    assert "componentKeys=agentculture_demo" in url
    assert "pullRequest=12" in url
    assert token == "secret"
    assert "secret" not in url


def test_every_page_is_read() -> None:
    first = {
        "total": gate.PAGE_SIZE + 2,
        "issues": [issue("OPEN", n) for n in range(gate.PAGE_SIZE)],
    }
    second = {"total": gate.PAGE_SIZE + 2, "issues": [issue("OPEN", 900), issue("CONFIRMED", 901)]}
    fetch, calls = fake([first, second])
    found = gate.open_issues("k", {"branch": "main"}, None, fetch)
    assert len(found) == gate.PAGE_SIZE + 2
    assert len(calls) == 2
    assert "p=2" in calls[1][0]


def test_an_error_answer_is_a_check_error_not_a_pass() -> None:
    fetch, _ = fake([{"errors": [{"msg": "Component key not found"}]}])
    with pytest.raises(gate.CheckError):
        gate.open_issues("k", {"branch": "main"}, None, fetch)


def test_requests_outside_sonarcloud_are_refused() -> None:
    with pytest.raises(gate.CheckError):
        gate.fetch_json("http://127.0.0.1:9/api/issues/search", None)


def run(monkeypatch, capsys, pages, argv, token="tok-do-not-print"):
    fetch, _ = fake(pages)
    monkeypatch.setattr(gate, "fetch_json", fetch)
    if token:
        monkeypatch.setenv("SONAR_TOKEN", token)
    real = gate.open_issues
    monkeypatch.setattr(gate, "open_issues", lambda k, s, t: real(k, s, t, fetch))
    code = gate.main(argv)
    out = capsys.readouterr()
    assert token not in out.out + out.err
    return code, out


def test_exit_0_when_nothing_is_open(monkeypatch, capsys) -> None:
    code, out = run(monkeypatch, capsys, [{"total": 1, "issues": [issue("ACCEPTED")]}],
                    ["--pull-request", "7"])  # fmt: skip
    assert code == 0
    assert "no open issue on pull request 7" in out.out


def test_exit_1_lists_each_open_issue(monkeypatch, capsys) -> None:
    pages = [{"total": 2, "issues": [issue("OPEN", 11), issue("CONFIRMED", 22)]}]
    code, out = run(monkeypatch, capsys, pages, ["--branch", "main"])
    assert code == 1
    assert "2 open issue(s) on branch main" in out.err
    assert "tests/test_x.py:11" in out.err
    assert "tests/test_x.py:22" in out.err
    assert "Accepted" in out.err


def test_exit_2_when_the_check_cannot_be_made(monkeypatch, capsys) -> None:
    code, out = run(monkeypatch, capsys, [{"errors": [{"msg": "nope"}]}], ["--branch", "main"])
    assert code == 2
    assert "could not check branch main" in out.err


def test_the_workflow_runs_the_check_after_the_scan() -> None:
    text = (REPO_ROOT / ".github" / "workflows" / "tests.yml").read_text(encoding="utf-8")
    scan = text.index("SonarCloud Scan")
    check = text.index("scripts/sonar_open_issues.py")
    assert scan < check
    # Guarded like the scan itself: no token, no scan, no check.
    step = text[text.rindex("- name:", 0, check) : check]
    assert "env.SONAR_TOKEN != ''" in step
    assert "sonar.qualitygate.wait=true" in (REPO_ROOT / "sonar-project.properties").read_text(
        encoding="utf-8"
    )
