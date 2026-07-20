"""Regression guard: learn / overview / explain stay in lockstep with the parser (t20).

Verified in the repo before this task landed: ``embeddings --help`` already
registered ``survey``, ``regions``, ``neighbors``, ``directions``, ``compare``,
``gaps``, and ``map`` as fully callable top-level verbs, while
``embeddings learn --json`` listed all seven under ``planned_commands`` and
``embeddings overview``'s ``_PLANNED_VERBS`` did the same -- telling an agent a
working, callable surface did not exist. CLAUDE.md calls the planned/callable
separation load-bearing in *both* directions: a roadmap verb must never be
mistaken for a callable one, and (the bug this fixes) a callable verb must
never be mistaken for a roadmap one.

This module never hardcodes the set of verbs. Every expectation is derived
from the real parser :func:`embeddings.cli._build_parser` builds (the same
parser ``main()`` uses), by walking its registered ``_SubParsersAction``
choices -- the identical technique ``tests/test_cli_contract.py`` uses. A verb
added or removed after this test lands changes what the walker finds and is
covered automatically; no edit to this file is needed, which is exactly the
property a hand-maintained list does not have.
"""

from __future__ import annotations

import argparse
import json
from typing import Iterator

import pytest

from embeddings.cli import _build_parser, main
from embeddings.cli._commands.overview import _PLANNED_VERBS
from embeddings.explain import known_paths

# --------------------------------------------------------------------------
# The walker: introspect the real parser, don't enumerate verbs by hand.
# Mirrors tests/test_cli_contract.py's _iter_subcommands.
# --------------------------------------------------------------------------


def _iter_subcommand_paths(
    parser: argparse.ArgumentParser, prefix: tuple[str, ...] = ()
) -> Iterator[tuple[str, ...]]:
    """Yield every subcommand path reachable from ``parser``, at every depth."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, subparser in action.choices.items():
                path = prefix + (name,)
                yield path
                yield from _iter_subcommand_paths(subparser, path)


def _top_level_verbs(parser: argparse.ArgumentParser) -> list[str]:
    """Names of the root parser's own immediate sub-verbs (depth 1)."""
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            return list(action.choices.keys())
    return []  # pragma: no cover - the root parser always registers subcommands


_PARSER = _build_parser()
_REGISTERED_PATHS = list(_iter_subcommand_paths(_PARSER))
_REGISTERED_PATH_SET = set(_REGISTERED_PATHS)
_REGISTERED_PATH_IDS = [" ".join(path) for path in _REGISTERED_PATHS]
_TOP_LEVEL_VERBS = _top_level_verbs(_PARSER)


def test_walker_finds_the_verbs_known_to_exist_today() -> None:
    """Sanity check on the walker itself -- not the source of truth below.

    Mirrors tests/test_cli_contract.py's identical sanity check. If this ever
    fails, the walker broke; the parametrized tests below already derive
    their expectations from ``_TOP_LEVEL_VERBS``/``_REGISTERED_PATHS`` and
    need no edits when a verb is added or removed.
    """
    discovered = set(_TOP_LEVEL_VERBS)
    for expected in (
        "whoami",
        "learn",
        "explain",
        "overview",
        "doctor",
        "cli",
        "survey",
        "regions",
        "neighbors",
        "directions",
        "compare",
        "gaps",
        "map",
    ):
        assert expected in discovered, f"walker did not find top-level verb {expected!r}"


def _learn_json(capsys: pytest.CaptureFixture[str]) -> dict:
    rc = main(["learn", "--json"])
    assert rc == 0
    return json.loads(capsys.readouterr().out)


# --------------------------------------------------------------------------
# `embeddings learn --json`: commands vs. planned_commands, parser-derived.
# --------------------------------------------------------------------------


def test_every_registered_top_level_verb_appears_in_learn_commands(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Every verb the parser actually registers must be advertised as callable.

    This is the exact regression the task fixes: `survey`, `regions`,
    `neighbors`, `directions`, `compare`, `gaps`, and `map` were all fully
    registered and callable while `learn --json` told an agent they were
    merely planned.
    """
    payload = _learn_json(capsys)
    advertised = {tuple(entry["path"])[0] for entry in payload["commands"]}
    missing = set(_TOP_LEVEL_VERBS) - advertised
    assert not missing, (
        f"registered top-level verb(s) {sorted(missing)} are callable via the real "
        "parser but do not appear anywhere in `embeddings learn --json`'s commands list"
    )


def test_learn_commands_names_no_verb_the_parser_does_not_register(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The inverse drift: a stale `commands` entry naming a verb that no longer exists."""
    payload = _learn_json(capsys)
    for entry in payload["commands"]:
        top = tuple(entry["path"])[0]
        assert top in _TOP_LEVEL_VERBS, (
            f"`embeddings learn --json` advertises {entry['path']!r} as a command, but "
            f"{top!r} is not a top-level verb the real parser registers"
        )


def test_learn_planned_commands_contains_nothing_the_parser_already_registers(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The exact bug this task fixes: a verb cannot be both callable and 'planned'.

    Checked against the exact registered *path*, not just the top-level verb
    name, so a legitimately new sub-verb of an existing noun group (e.g. a
    hypothetical future `directions cluster` while only `directions discover`
    exists today) would still be a valid thing to list as planned.
    """
    payload = _learn_json(capsys)
    for entry in payload["planned_commands"]:
        path = tuple(entry["path"])
        assert path not in _REGISTERED_PATH_SET, (
            f"`embeddings learn --json` lists {list(path)} as planned, but "
            f"`embeddings {' '.join(path)}` is already a registered, callable command"
        )


# --------------------------------------------------------------------------
# `embeddings explain`: every registered verb path must resolve.
# --------------------------------------------------------------------------


@pytest.mark.parametrize("path", _REGISTERED_PATHS, ids=_REGISTERED_PATH_IDS)
def test_explain_resolves_every_registered_verb_path(
    path: tuple[str, ...], capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(["explain", *path, "--json"])
    assert rc == 0, f"`embeddings explain {' '.join(path)}` did not exit 0"
    payload = json.loads(capsys.readouterr().out)
    markdown = payload["markdown"]
    assert markdown.strip(), f"`embeddings explain {' '.join(path)}` resolved but returned empty"


def test_known_paths_cover_every_registered_verb_path() -> None:
    """Sanity check on the catalog's coverage, independent of any one lookup."""
    catalog_paths = set(known_paths())
    missing = [path for path in _REGISTERED_PATHS if path not in catalog_paths]
    assert not missing, f"no explain catalog entry for: {missing}"


# --------------------------------------------------------------------------
# `embeddings overview`: the same drift, on its own hand-maintained list.
# --------------------------------------------------------------------------


def _leading_token(line: str) -> str:
    parts = line.split()
    return parts[0] if parts else ""


def test_overview_planned_verbs_names_nothing_the_parser_already_registers() -> None:
    """`overview`'s roadmap list must not name an already-callable verb either.

    `_PLANNED_VERBS` entries are written as `"<verb> ... - description"`; the
    leading token is the verb name being advertised as not-yet-implemented.
    """
    for line in _PLANNED_VERBS:
        verb = _leading_token(line)
        assert verb not in _TOP_LEVEL_VERBS, (
            f"embeddings overview lists {line!r} as a planned verb, but {verb!r} is "
            "already a registered, callable top-level command"
        )
