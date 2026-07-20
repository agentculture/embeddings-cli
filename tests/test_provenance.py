"""Tests for bug F2 — every structural output cites the command that reproduces it.

``docs/evidence/2026-07-20-live-end-to-end.md``'s "Bugs found" section (F2)
records that ``params`` and ``seed`` were stamped on every structural payload,
but nothing emitted the literal command line a reader could re-run. This file
proves the fix in two layers:

* **The helper itself** (:mod:`embeddings.cli._provenance`) — rendering,
  quoting, the ``embeddings``-prefix normalisation (never a path, never
  ``python -m ...``), and credential redaction.
* **The wiring** — each of the seven survey-surface verbs (``survey``,
  ``regions``, ``neighbors``, ``directions discover``, ``compare``, ``gaps``,
  ``map``) carries the rendered command under one JSON key
  (``reproduced_by``) and one trailing text-mode line (``reproduce: ...``),
  plus ``map --html``'s artifact carrying it in its provenance block.

Every wiring test below builds atlases directly on disk and runs entirely
under :func:`tests._fakes.blocked_sockets` — none of the seven verbs need a
live gateway for the invocations exercised here (``survey`` without
``--apply``, every reader verb, and ``directions discover``/``compare``
resolved from atlas item ids alone).
"""

from __future__ import annotations

import argparse
import html as html_mod
import json
import shlex
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pytest

from embeddings.atlas.io import write_atlas
from embeddings.atlas.schema import SCHEMA_VERSION, AtlasManifest, AtlasRecord
from embeddings.cli import main
from embeddings.cli._provenance import (
    PROG,
    REDACTED,
    REPRODUCE_LINE_PREFIX,
    REPRODUCED_BY_KEY,
    reproduce_line,
    reproducing_command,
)
from tests._fakes import blocked_sockets, deterministic_vector

DIM = 6

# --------------------------------------------------------------------------
# The helper: rendering, quoting, prefix normalisation
# --------------------------------------------------------------------------


def test_no_arguments_renders_just_the_prog() -> None:
    assert reproducing_command([]) == PROG


def test_renders_prog_then_every_argument_in_order() -> None:
    command = reproducing_command(["regions", "--atlas", "atlas", "--seed", "0"])
    assert command == "embeddings regions --atlas atlas --seed 0"


def test_arguments_containing_spaces_are_shell_quoted() -> None:
    command = reproducing_command(["survey", "my corpus.jsonl"])
    # shlex.split must recover the exact original tokens.
    assert shlex.split(command) == ["embeddings", "survey", "my corpus.jsonl"]
    assert "'my corpus.jsonl'" in command


def test_arguments_containing_shell_metacharacters_round_trip() -> None:
    hostile = "it's a $(rm -rf /) & <injected>"
    command = reproducing_command(["neighbors", hostile])
    assert shlex.split(command) == ["embeddings", "neighbors", hostile]


def test_every_token_is_individually_quoted_and_reparses_exactly() -> None:
    argv = ["directions", "discover", "--pairs", "a b/pairs.jsonl", "--top", "2"]
    command = reproducing_command(argv)
    assert shlex.split(command) == [PROG, *argv]


# --------------------------------------------------------------------------
# The `embeddings` normalisation — never a path, never `python -m ...`
# --------------------------------------------------------------------------


def test_default_argv_reads_sys_argv_and_still_starts_with_embeddings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["/home/user/.venv/bin/embeddings", "map", "--atlas", "atlas"])
    command = reproducing_command()
    assert command == "embeddings map --atlas atlas"
    assert "/home/user/.venv/bin" not in command


def test_sys_argv0_being_python_dash_m_form_never_leaks_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `python -m embeddings.cli ...` sets argv[0] to the module's __file__ path,
    # not to "embeddings" -- the renderer must not echo it either way, because
    # it never reads argv[0] at all.
    monkeypatch.setattr(
        sys,
        "argv",
        ["/usr/lib/python3.12/embeddings/cli/__init__.py", "gaps", "--json"],
    )
    command = reproducing_command()
    assert command.startswith(f"{PROG} ")
    assert "__init__.py" not in command
    assert "python" not in command.lower()


def test_explicit_argv_overrides_sys_argv_without_monkeypatching_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "argv", ["embeddings", "map", "--json"])
    # Passing argv explicitly must win over whatever sys.argv happens to hold --
    # this is the "injectable, no global monkeypatching required" property.
    command = reproducing_command(["regions", "--seed", "1"])
    assert command == "embeddings regions --seed 1"


# --------------------------------------------------------------------------
# Credential redaction
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flag",
    [
        "--api-key",
        "--API-KEY",
        "--embed-api-key",
        "--token",
        "--auth-token",
        "--secret",
        "--client-secret",
        "--password",
        "--passwd",
        "--credential",
    ],
)
def test_credential_shaped_flag_value_is_redacted_space_separated(flag: str) -> None:
    command = reproducing_command(["survey", "corpus.jsonl", flag, "sk-super-secret-value"])
    assert "sk-super-secret-value" not in command
    assert REDACTED in command
    # The flag name itself is never redacted -- only the value that follows it.
    assert flag.lower() in command.lower()


@pytest.mark.parametrize("flag", ["--api-key", "--token", "--secret"])
def test_credential_shaped_flag_value_is_redacted_inline_equals_form(flag: str) -> None:
    command = reproducing_command(["survey", "corpus.jsonl", f"{flag}=sk-super-secret-value"])
    assert "sk-super-secret-value" not in command
    assert REDACTED in command
    assert f"{flag}={REDACTED}" in command


def test_redaction_only_consumes_the_immediate_next_token() -> None:
    argv = ["survey", "corpus.jsonl", "--api-key", "sk-secret", "--batch-size", "32"]
    command = reproducing_command(argv)
    tokens = shlex.split(command)
    assert tokens[-2:] == ["--batch-size", "32"]
    assert "sk-secret" not in command


def test_a_value_merely_containing_the_word_key_is_not_redacted() -> None:
    # Only the *flag name* is inspected for credential markers, never the
    # value -- otherwise ordinary prose ("the key to happiness") would trigger
    # redaction on a completely unrelated flag.
    command = reproducing_command(["neighbors", "find the key to happiness", "-k", "5"])
    assert "find the key to happiness" in command
    assert REDACTED not in command


def test_a_positional_argument_containing_key_is_never_treated_as_a_flag() -> None:
    command = reproducing_command(["neighbors", "--id", "api-key-reference-doc"])
    assert "api-key-reference-doc" in command
    assert REDACTED not in command


def test_non_credential_flags_are_never_redacted() -> None:
    argv = ["regions", "--atlas", "atlas", "--clusters", "5", "--seed", "0"]
    assert reproducing_command(argv) == "embeddings " + " ".join(argv)


def test_multiple_credential_flags_are_each_independently_redacted() -> None:
    argv = [
        "survey",
        "corpus.jsonl",
        "--api-key",
        "first-secret",
        "--embed-url",
        "http://localhost:8001/v1",
        "--token",
        "second-secret",
    ]
    command = reproducing_command(argv)
    assert "first-secret" not in command
    assert "second-secret" not in command
    assert "http://localhost:8001/v1" in command
    assert command.count(REDACTED) == 2


def test_credential_flag_as_the_final_token_does_not_crash() -> None:
    # Defensive: no current flag is a bare credential switch with no value,
    # but the renderer must not raise if one were ever passed with nothing
    # after it.
    command = reproducing_command(["survey", "corpus.jsonl", "--api-key"])
    assert command == "embeddings survey corpus.jsonl --api-key"


def test_reproduce_line_uses_the_shared_prefix() -> None:
    line = reproduce_line("embeddings regions --atlas atlas --seed 0")
    assert line == f"{REPRODUCE_LINE_PREFIX} embeddings regions --atlas atlas --seed 0"
    assert line.startswith("reproduce:")


# --------------------------------------------------------------------------
# Wiring: each of the seven survey-surface verbs
# --------------------------------------------------------------------------


def _expected(argv: Sequence[str]) -> str:
    """Independently-computed expected rendering (mirrors the module under test)."""
    return " ".join([PROG, *(shlex.quote(token) for token in argv)])


def _write_atlas(target: Path, ids: list[str], texts: list[str], vectors: np.ndarray) -> Path:
    manifest = AtlasManifest(
        schema_version=SCHEMA_VERSION,
        model="fake-embedding-model",
        endpoint="http://localhost:8002/v1",
        params={"batch_size": 32},
        seed=0,
        limitations=[],
        uncertain=[],
    )
    records = [
        AtlasRecord(id=item_id, text=text, vector_index=index, metadata={})
        for index, (item_id, text) in enumerate(zip(ids, texts))
    ]
    return write_atlas(manifest, records, vectors.astype(np.float32), path=target, overwrite=True)


def _blob_atlas(target: Path, *, items: int = 6) -> Path:
    ids = [f"item-{i}" for i in range(items)]
    texts = [f"document number {i}" for i in range(items)]
    vectors = np.asarray([deterministic_vector(text, DIM) for text in texts], dtype=np.float64)
    return _write_atlas(target, ids, texts, vectors)


def _write_pairs(path: Path, pairs: list[tuple[str, str]]) -> Path:
    lines = [json.dumps({"a": a, "b": b}) for a, b in pairs]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _write_corpus(path: Path, texts: list[str]) -> Path:
    lines = [json.dumps({"id": f"doc-{i}", "text": text}) for i, text in enumerate(texts)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _run(monkeypatch: pytest.MonkeyPatch, argv: list[str]) -> int:
    """Invoke ``main(argv)`` with ``sys.argv`` set to match, under no sockets."""
    monkeypatch.setattr(sys, "argv", [PROG, *argv])
    with blocked_sockets():
        return main(argv)


def _assert_json_and_text_wiring(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    base_argv: list[str],
) -> None:
    """Shared assertion body for every verb: --json key, then the text-mode line."""
    json_argv = [*base_argv, "--json"]
    rc = _run(monkeypatch, json_argv)
    out = capsys.readouterr().out
    assert rc == 0, out
    payload = json.loads(out)
    assert REPRODUCED_BY_KEY in payload
    assert payload[REPRODUCED_BY_KEY] == _expected(json_argv)

    rc = _run(monkeypatch, base_argv)
    out = capsys.readouterr().out
    assert rc == 0, out
    lines = [line for line in out.splitlines() if line.strip()]
    assert lines[-1] == reproduce_line(_expected(base_argv))


@pytest.fixture
def atlas_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return _blob_atlas(tmp_path / "atlas")


def test_survey_dry_run_wiring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    corpus = _write_corpus(tmp_path / "corpus.jsonl", ["alpha document", "beta document"])
    _assert_json_and_text_wiring(monkeypatch, capsys, ["survey", str(corpus)])


def test_regions_wiring(
    atlas_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "regions",
        "--atlas",
        str(atlas_dir),
        "--clusters",
        "2",
        "--runs",
        "1",
        "--resamples",
        "0",
    ]
    _assert_json_and_text_wiring(monkeypatch, capsys, argv)


def test_neighbors_wiring(
    atlas_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["neighbors", "--id", "item-0", "--atlas", str(atlas_dir), "-k", "2"]
    _assert_json_and_text_wiring(monkeypatch, capsys, argv)


def test_directions_discover_wiring(
    atlas_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    pairs = _write_pairs(
        atlas_dir.parent / "pairs.jsonl",
        [("item-0", "item-1"), ("item-2", "item-3"), ("item-4", "item-5")],
    )
    argv = [
        "directions",
        "discover",
        "--atlas",
        str(atlas_dir),
        "--pairs",
        str(pairs),
        "--runs",
        "1",
        "--resamples",
        "0",
    ]
    _assert_json_and_text_wiring(monkeypatch, capsys, argv)


def test_compare_wiring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    atlas_a = _blob_atlas(tmp_path / "atlas-a")
    atlas_b = _blob_atlas(tmp_path / "atlas-b")
    argv = ["compare", "--atlas-a", str(atlas_a), "--atlas-b", str(atlas_b), "--top-k", "3"]
    _assert_json_and_text_wiring(monkeypatch, capsys, argv)


def test_gaps_wiring(
    atlas_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "gaps",
        "--atlas",
        str(atlas_dir),
        "--clusters",
        "2",
        "--runs",
        "1",
        "--resamples",
        "0",
    ]
    _assert_json_and_text_wiring(monkeypatch, capsys, argv)


def test_map_wiring(
    atlas_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "map",
        "--atlas",
        str(atlas_dir),
        "--clusters",
        "2",
        "--runs",
        "1",
        "--resamples",
        "0",
    ]
    _assert_json_and_text_wiring(monkeypatch, capsys, argv)


# --------------------------------------------------------------------------
# Introspection verbs are untouched
# --------------------------------------------------------------------------


@pytest.mark.parametrize("argv", [["whoami"], ["learn"], ["overview"], ["doctor"], ["cli"]])
def test_introspection_verbs_do_not_carry_a_reproduced_by_key(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    monkeypatch.setattr(sys, "argv", [PROG, *argv, "--json"])
    with blocked_sockets():
        rc = main([*argv, "--json"])
    out = capsys.readouterr().out
    assert rc == 0, out
    payload = json.loads(out)
    assert REPRODUCED_BY_KEY not in payload


# --------------------------------------------------------------------------
# map --html: the artifact's own provenance block carries the command
# --------------------------------------------------------------------------


def test_map_html_artifact_embeds_the_reproducing_command_escaped(
    atlas_dir: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    target = atlas_dir.parent / "map.html"
    argv = [
        "map",
        "--atlas",
        str(atlas_dir),
        "--html",
        str(target),
        "--apply",
        "--clusters",
        "2",
        "--runs",
        "1",
        "--resamples",
        "0",
    ]
    rc = _run(monkeypatch, argv)
    assert rc == 0
    artifact = target.read_text(encoding="utf-8")
    expected = _expected(argv)
    # HTML-escaped, matching every other value in the provenance block --
    # never interpolated raw. shlex-quoted argument text containing no
    # HTML-special characters here, so escape() is a no-op on this string;
    # the point is that it goes through the same _escape() call as every
    # neighbouring row.
    assert html_mod.escape(expected, quote=True) in artifact
    assert "Reproduce" in artifact


def test_map_html_artifact_reproducing_command_is_html_escaped_when_hostile(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A corpus path containing markup-shaped characters must not inject markup.

    ``reproducing_command`` shell-quotes its tokens but does not HTML-escape
    them -- that is ``map``'s job, via the same ``_escape`` every other
    provenance row goes through. This proves the artifact never interpolates
    the rendered command raw.
    """
    from embeddings.cli._commands import map as map_mod

    monkeypatch.chdir(tmp_path)
    atlas_dir = _blob_atlas(tmp_path / "atlas")
    hostile_argv = ["map", "--atlas", str(atlas_dir), "--seed", "<script>alert(1)</script>"]
    monkeypatch.setattr(sys, "argv", [PROG, *hostile_argv])

    args = argparse.Namespace(
        atlas=str(atlas_dir),
        json=False,
        html=None,
        apply=False,
        force=False,
        seed=0,
        clusters=0,
        runs=1,
        resamples=0,
        max_iterations=100,
        tolerance=1e-6,
    )
    payload = map_mod.build_payload(args)
    assert "<script>" in payload[REPRODUCED_BY_KEY]

    artifact = map_mod.render_html(payload)
    assert "<script>alert(1)</script>" not in artifact
    assert "&lt;script&gt;" in artifact
