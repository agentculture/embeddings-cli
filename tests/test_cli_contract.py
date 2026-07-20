"""Data-driven guard for the stable output/error contract (t21).

``embeddings/cli/_errors.py``, ``_output.py``, and ``cli/__init__.py`` are
marked *stable-contract* in ``CLAUDE.md``: every failure raises
:class:`~embeddings.cli._errors.CliError`, results always go to stdout,
errors/diagnostics always go to stderr, text-mode errors render
``error: <msg>`` then ``hint: <remediation>``, and every command (including
every sub-verb of a noun group) supports ``--json``.

This module does **not** hardcode verb names. Four sibling verb tasks
(``regions``, ``neighbors``, ``directions``, ``compare``) are landing in
parallel and are not present in this worktree; a test that enumerated verbs
by hand would silently stop covering the CLI the moment those merge. Instead
this walks the *real* parser built by :func:`embeddings.cli._build_parser`
(the same parser ``main()`` builds internally) and asserts the contract
against every subcommand it finds, at every depth. New verbs, and new
sub-verbs of existing noun groups, are covered automatically -- no edit to
this file is needed when they land.

Nothing here touches ``embeddings/cli/__init__.py``, ``_errors.py``, or
``_output.py``; it only reads their behaviour through ``main()``.
"""

from __future__ import annotations

import argparse
import json
from typing import Iterator

import pytest

from embeddings.cli import _build_parser, main
from tests._fakes import blocked_sockets

# A flag no real verb will ever declare, so "unrecognized arguments" is the
# only thing that can happen -- used to force every subcommand down the
# parse-error path without knowing anything about its real flags.
_BOGUS_FLAG = "--definitely-not-a-real-flag-t21-guard"


# --------------------------------------------------------------------------
# The walker: introspect the real parser, don't enumerate verbs by hand.
# --------------------------------------------------------------------------


def _iter_subcommands(
    parser: argparse.ArgumentParser, prefix: tuple[str, ...] = ()
) -> Iterator[tuple[tuple[str, ...], argparse.ArgumentParser]]:
    """Yield ``(path, subparser)`` for every subcommand reachable from ``parser``.

    Recurses through nested ``_SubParsersAction`` groups (noun groups with
    their own sub-verbs, e.g. ``cli overview``), so a path can have any depth.
    """
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, subparser in action.choices.items():
                path = prefix + (name,)
                yield path, subparser
                yield from _iter_subcommands(subparser, path)


def _own_option_strings(parser: argparse.ArgumentParser) -> set[str]:
    """Option strings declared directly on ``parser`` (not inherited from a parent)."""
    strings: set[str] = set()
    for action in parser._actions:
        strings.update(action.option_strings)
    return strings


def _direct_subverb_names(parser: argparse.ArgumentParser) -> list[str] | None:
    """Names of ``parser``'s own immediate sub-verbs, or ``None`` if it has none."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return list(action.choices.keys())
    return None


def _requires_arguments(parser: argparse.ArgumentParser) -> bool:
    """True if ``parser`` cannot be invoked with zero extra tokens (besides ``--json``)."""
    for action in parser._actions:
        if isinstance(action, (argparse._SubParsersAction, argparse._HelpAction)):
            continue
        if action.option_strings:
            if action.required:
                return True
        else:
            # A bare positional is required unless its nargs allows "zero".
            if action.nargs not in ("?", "*", argparse.REMAINDER):
                return True
    return False


def _discover_subcommands() -> list[tuple[tuple[str, ...], argparse.ArgumentParser]]:
    return list(_iter_subcommands(_build_parser()))


_SUBCOMMANDS = _discover_subcommands()
_SUBCOMMAND_IDS = [" ".join(path) for path, _parser in _SUBCOMMANDS]

_GROUPS = [(path, p) for path, p in _SUBCOMMANDS if _direct_subverb_names(p)]
_GROUP_IDS = [" ".join(path) for path, _parser in _GROUPS]

_ZERO_ARG_LEAVES = [
    (path, p)
    for path, p in _SUBCOMMANDS
    if not _requires_arguments(p) and p.get_default("func") is not None
]
_ZERO_ARG_LEAF_IDS = [" ".join(path) for path, _parser in _ZERO_ARG_LEAVES]


def test_discovery_finds_the_verbs_known_to_exist_today() -> None:
    """Sanity check the walker itself -- not an enumeration the guard depends on.

    ``_SUBCOMMANDS`` (used by every parametrized test below) is built from
    live introspection; this only proves that introspection isn't silently
    returning nothing. When regions/neighbors/directions/compare merge, this
    set grows on its own -- nothing here needs editing.
    """
    discovered = {path for path, _parser in _SUBCOMMANDS}
    for expected in (
        ("whoami",),
        ("learn",),
        ("explain",),
        ("overview",),
        ("doctor",),
        ("cli",),
        ("cli", "overview"),
        ("survey",),
    ):
        assert expected in discovered, f"walker did not find {expected!r}"


# --------------------------------------------------------------------------
# Property 1: every subcommand declares its own --json.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path,parser", _SUBCOMMANDS, ids=_SUBCOMMAND_IDS)
def test_every_subcommand_declares_its_own_json_flag(
    path: tuple[str, ...], parser: argparse.ArgumentParser
) -> None:
    assert "--json" in _own_option_strings(parser), (
        f"'embeddings {' '.join(path)}' does not declare its own --json -- argparse does "
        "not inherit a parent's flag, so every sub-verb of a noun group needs one too"
    )


# --------------------------------------------------------------------------
# Property 2: --help always exits 0 with output on stdout only.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path,parser", _SUBCOMMANDS, ids=_SUBCOMMAND_IDS)
def test_every_subcommand_help_exits_0_to_stdout_only(
    path: tuple[str, ...],
    parser: argparse.ArgumentParser,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        main([*path, "--help"])
    assert exc.value.code == 0
    captured = capsys.readouterr()
    assert captured.out.strip(), f"'embeddings {' '.join(path)} --help' wrote nothing to stdout"
    assert captured.err == "", f"'embeddings {' '.join(path)} --help' leaked onto stderr"


# --------------------------------------------------------------------------
# Property 3: an invalid flag errors to stderr only, exit 1, hint: in text mode.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path,parser", _SUBCOMMANDS, ids=_SUBCOMMAND_IDS)
def test_every_subcommand_invalid_flag_is_a_structured_error_text_mode(
    path: tuple[str, ...],
    parser: argparse.ArgumentParser,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        main([*path, _BOGUS_FLAG])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert (
        captured.out == ""
    ), f"'embeddings {' '.join(path)} {_BOGUS_FLAG}' leaked a result onto stdout on error"
    assert captured.err.startswith("error:"), (
        f"'embeddings {' '.join(path)} {_BOGUS_FLAG}' did not render 'error: ' on stderr: "
        f"{captured.err!r}"
    )
    assert "hint:" in captured.err, (
        f"'embeddings {' '.join(path)} {_BOGUS_FLAG}' is missing the required 'hint:' line: "
        f"{captured.err!r}"
    )


# --------------------------------------------------------------------------
# Property 4: same invalid-flag case, but --json mode -- structured JSON on
# stderr, still nothing on stdout.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path,parser", _SUBCOMMANDS, ids=_SUBCOMMAND_IDS)
def test_every_subcommand_invalid_flag_is_a_structured_error_json_mode(
    path: tuple[str, ...],
    parser: argparse.ArgumentParser,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        main([*path, "--json", _BOGUS_FLAG])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert (
        captured.out == ""
    ), f"'embeddings {' '.join(path)} --json {_BOGUS_FLAG}' leaked a result onto stdout"
    payload = json.loads(captured.err)
    assert payload["code"] == 1
    assert isinstance(payload["message"], str) and payload["message"]
    assert "remediation" in payload


# --------------------------------------------------------------------------
# Property 5: any noun group (has its own sub-verbs) exposes 'overview'.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path,parser", _GROUPS, ids=_GROUP_IDS)
def test_every_noun_group_exposes_overview(
    path: tuple[str, ...], parser: argparse.ArgumentParser
) -> None:
    names = _direct_subverb_names(parser)
    assert names is not None
    assert "overview" in names, (
        f"'embeddings {' '.join(path)}' has sub-verbs {names} but no 'overview' -- the "
        "agent-first rubric requires any noun with action-verbs to also expose one"
    )


# --------------------------------------------------------------------------
# Property 6 (bonus): a real successful run keeps the split too, for every
# subcommand that can be invoked with zero extra arguments. Sandboxed with
# blocked_sockets() and a throwaway cwd so an unknown future verb that
# happens to match "no required arguments" cannot reach the network or write
# into this worktree even if its handler runs for real here.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path,parser", _ZERO_ARG_LEAVES, ids=_ZERO_ARG_LEAF_IDS)
def test_bare_success_keeps_result_on_stdout_and_off_stderr(
    path: tuple[str, ...],
    parser: argparse.ArgumentParser,
    capsys: pytest.CaptureFixture[str],
    tmp_path: object,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.chdir(tmp_path)  # type: ignore[arg-type]
    with blocked_sockets():
        rc = main([*path, "--json"])
    captured = capsys.readouterr()
    if rc != 0:
        # Some verbs legitimately signal a non-CliError failure this way (e.g.
        # 'doctor' returns 1 for "unhealthy" while still emitting a full
        # result) -- the error-shaped assertions above already cover the
        # CliError path generically, so there is nothing more to assert here.
        return
    assert captured.out.strip(), f"'embeddings {' '.join(path)} --json' wrote nothing to stdout"
    result = json.loads(captured.out)
    assert result is not None
    assert not captured.err.startswith("error:"), (
        f"'embeddings {' '.join(path)} --json' succeeded but stderr looks like an error: "
        f"{captured.err!r}"
    )
