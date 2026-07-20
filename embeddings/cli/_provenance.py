"""``reproducing_command`` — render the invocation that reproduces a structural output.

Why this exists (bug F2)
-------------------------
``docs/design-notes/semantic-measurement.md`` states the whole premise of this
CLI: prose comes from the agent, numbers come from the tool, and *every claim
is traceable to a command anyone can re-run*. Every structural verb already
stamps its output with ``params`` and ``seed`` — enough for a reader to
reconstruct the invocation by hand — but nothing emitted the invocation
itself. This module closes that gap: one helper, called from each structural
verb's payload assembly, that renders the actual command line that produced
the output sitting in front of the reader.

It matters most for ``map --html``: that artifact is designed to travel alone
(a screenshot, a shared file), so the command that reproduces it has to live
*inside* the file, not only in whatever terminal scrollback made it.

Design choices
--------------
* **Starts with ``embeddings``, never a path or an interpreter.** CLAUDE.md is
  explicit: invocations read ``embeddings ...``, not the absolute path a
  console-script shim resolves to, and not ``python -m embeddings.cli ...``.
  This is enforced structurally, not by inspecting ``sys.argv[0]`` and hoping
  it looks right — the renderer never reads ``argv[0]`` at all; it always
  emits the literal :data:`PROG` and appends whatever arguments follow it.
* **Every token is shell-quoted** (:func:`shlex.quote`), so a corpus path or
  probe text containing spaces, quotes, or shell metacharacters round-trips
  through a copy-paste instead of silently splitting into the wrong argv.
* **Credential-shaped arguments are redacted defensively.** Nothing this CLI
  exposes today accepts a secret on the command line — the gateway token
  (``EMBEDDINGS_API_KEY`` / ``COLLEAGUE_API_KEY``) is environment-only — but a
  provenance string is exactly the kind of text that gets pasted into an
  issue, a chat, or a committed Atlas, so this function does not trust that
  invariant to hold forever. Any flag whose name contains ``key``, ``token``,
  ``secret``, ``password``, ``passwd`` or ``credential`` (case-insensitive,
  checked against the flag name only, not its value) has its *value* replaced
  with ``***`` — both the ``--flag value`` and ``--flag=value`` spellings. The
  flag name itself is never redacted, only what follows it.
* **Injectable, not just sys.argv-coupled.** :func:`reproducing_command`
  accepts an optional ``argv`` sequence so a caller (or a test) can render a
  specific invocation without monkeypatching ``sys.argv`` globally. Every
  structural verb calls it with no argument, which defaults to
  ``sys.argv[1:]`` — the arguments this process actually ran with.
"""

from __future__ import annotations

import shlex
import sys
from typing import Sequence

#: What every rendered command starts with. Never the absolute path to the
#: installed console-script shim, never ``python -m ...`` — see the module
#: docstring and CLAUDE.md's "Naming" section, which this mirrors exactly.
PROG = "embeddings"

#: Case-insensitive substrings that mark a flag as credential-shaped. Matched
#: against the flag name only (leading dashes and any ``=value`` suffix
#: stripped first), never against positional arguments or values.
_CREDENTIAL_MARKERS: tuple[str, ...] = (
    "key",
    "token",
    "secret",
    "password",
    "passwd",
    "credential",
)

#: What a redacted value is replaced with. Never empty (an empty string could
#: be mistaken for "this flag takes no value"), never a real-looking value.
REDACTED = "***"

#: The one JSON key every structural verb's ``--json`` payload carries the
#: rendered command under. Kept as a constant, imported by every verb module,
#: so the key name can never drift between them.
REPRODUCED_BY_KEY = "reproduced_by"

#: Prefix for the trailing text-mode line every structural verb appends.
#: Kept alongside :func:`reproduce_line` so the exact wording lives in one
#: place rather than being retyped, and possibly drifting, in seven files.
REPRODUCE_LINE_PREFIX = "reproduce:"


def _flag_name(token: str) -> str:
    """``--api-key=xyz`` / ``--api-key`` -> ``api-key``; a bare positional is unchanged."""
    return token.lstrip("-").split("=", 1)[0].lower()


def _is_credential_flag(token: str) -> bool:
    if not token.startswith("-"):
        return False
    name = _flag_name(token)
    return any(marker in name for marker in _CREDENTIAL_MARKERS)


def reproducing_command(argv: Sequence[str] | None = None) -> str:
    """Render ``argv`` (default: live ``sys.argv[1:]``) as a re-runnable ``embeddings ...`` line.

    Always starts with :data:`PROG`. Every token is quoted with
    :func:`shlex.quote`. A credential-shaped flag's value is replaced with
    :data:`REDACTED` in both the ``--flag value`` and ``--flag=value`` forms;
    the flag name itself is left untouched so the rendered command still
    documents *which* flag was passed, just not what it was passed.

    Pass ``argv`` explicitly to render a specific invocation (e.g. in a test,
    or when the caller already has the parsed argument list) without
    monkeypatching ``sys.argv``.
    """
    tokens = list(argv) if argv is not None else sys.argv[1:]
    rendered: list[str] = [PROG]
    redact_next_value = False

    for token in tokens:
        if redact_next_value:
            rendered.append(REDACTED)
            redact_next_value = False
            continue

        if _is_credential_flag(token):
            if "=" in token:
                flag, _, _value = token.partition("=")
                rendered.append(f"{shlex.quote(flag)}={REDACTED}")
            else:
                rendered.append(shlex.quote(token))
                redact_next_value = True
            continue

        rendered.append(shlex.quote(token))

    return " ".join(rendered)


def reproduce_line(command: str) -> str:
    """The trailing text-mode line every structural verb appends verbatim.

    A single shared function so ``"reproduce: "`` is spelled once, not
    independently in each of the seven verb modules that call it.
    """
    return f"{REPRODUCE_LINE_PREFIX} {command}"
