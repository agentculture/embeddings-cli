"""Tests for ``embeddings survey`` — the one verb that writes an Atlas.

Two properties get disproportionate attention here, because they are the ones
whose failure destroys work rather than merely reporting it wrongly:

* **Dry run writes nothing.** ``test_dry_run_performs_zero_filesystem_writes``
  hashes the entire temp tree (corpus, cache root, and the working directory the
  default ``./atlas`` target would land in) before and after a preview run and
  asserts the snapshots are identical — not "the atlas directory is absent",
  which a stray cache write or a partially-built scratch directory would sail
  past.
* **``--force`` replaces whole, never merges.** The pre-existing atlas in
  ``test_force_replaces_the_atlas_whole`` carries a record and a stray file that
  the new survey does not produce; both must be gone afterwards.

No test here opens a socket to an embedding backend except
``test_unreachable_backend_exits_2``, which must — proving a real connection
failure surfaces as exit 2 without a traceback is the whole point of it, and it
targets ``tests._fakes.DEAD_PORT`` (reserved, nothing listening), never the real
gateway port. Everywhere else a :class:`FakeEmbeddingTransport` is wired into a
real :class:`EmbeddingClient` through the ``build_client`` seam.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import pytest

from embeddings.atlas import read_atlas
from embeddings.atlas.io import MANIFEST_FILENAME, RECORDS_FILENAME, VECTORS_FILENAME
from embeddings.backends import EmbeddingClient
from embeddings.cli import main
from embeddings.cli._commands import survey as survey_mod
from tests._fakes import (
    FakeEmbeddingTransport,
    UnreachableTransport,
    blocked_sockets,
    dead_endpoint,
    wire_transport,
)

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------


def _write_corpus(path: Path, texts: Sequence[str]) -> Path:
    lines = [json.dumps({"id": f"doc-{i}", "text": text}) for i, text in enumerate(texts)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _tree_snapshot(root: Path) -> dict[str, str]:
    """Content-addressed snapshot of every path under ``root``.

    Directories are recorded by name so a created-then-emptied directory still
    shows as a difference; files are recorded by sha256 of their bytes so a
    same-size in-place rewrite is caught too.
    """
    snapshot: dict[str, str] = {}
    for entry in sorted(root.rglob("*")):
        key = str(entry.relative_to(root))
        if entry.is_dir():
            snapshot[key + "/"] = "<dir>"
        else:
            snapshot[key] = hashlib.sha256(entry.read_bytes()).hexdigest()
    return snapshot


class FlakyTransport:
    """Fails ``failures`` times with a transport-level error, then serves normally."""

    def __init__(self, failures: int, dim: int = 8) -> None:
        self.remaining_failures = failures
        self.inner = FakeEmbeddingTransport(dim=dim)
        self.attempts = 0

    def __call__(self, texts: Sequence[str], model: str = "fake-embedding-model") -> dict:
        self.attempts += 1
        if self.remaining_failures > 0:
            self.remaining_failures -= 1
            raise ConnectionResetError("simulated: transient gateway reset")
        return self.inner(texts, model)


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An isolated working directory with the embed cache pinned inside it."""
    monkeypatch.setenv("EMBEDDINGS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("EMBEDDINGS_EMBED_URL", raising=False)
    monkeypatch.delenv("EMBEDDINGS_EMBED_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _wire_fake(
    monkeypatch: pytest.MonkeyPatch, fake: Any, *, model: str = "fake-embedding-model"
) -> None:
    """Replace survey's client factory with one backed by ``fake``. Opens no socket."""

    def _build(base_url: str | None, model_name: str | None) -> EmbeddingClient:
        return EmbeddingClient(
            base_url=base_url or "http://fake.invalid/v1",
            model=model_name or model,
            transport=wire_transport(fake),
        )

    monkeypatch.setattr(survey_mod, "build_client", _build)


def _run(argv: list[str]) -> int:
    return main(argv)


# --------------------------------------------------------------------------
# Registration + parser contract
# --------------------------------------------------------------------------


def test_survey_is_registered_on_the_root_parser(capsys: pytest.CaptureFixture[str]) -> None:
    rc = _run([])
    assert rc == 0
    assert "survey" in capsys.readouterr().out


def test_survey_declares_its_own_json_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        _run(["survey", "--help"])
    assert exc.value.code == 0
    assert "--json" in capsys.readouterr().out


def test_every_survey_flag_has_help_text(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        _run(["survey", "--help"])
    help_text = capsys.readouterr().out
    for flag in ("--apply", "--force", "--atlas", "--batch-size", "--seed", "--embed-url"):
        assert flag in help_text


def test_missing_corpus_is_a_structured_parse_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        _run(["survey"])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert "hint: " in captured.err


def test_parse_error_honours_json_mode(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        _run(["survey", "--json", "--nonsuch-flag"])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    payload = json.loads(captured.err)
    assert payload["code"] == 1
    assert payload["remediation"]


# --------------------------------------------------------------------------
# Dry run: the zero-write guarantee
# --------------------------------------------------------------------------


def test_dry_run_performs_zero_filesystem_writes(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _write_corpus(sandbox / "corpus.jsonl", ["alpha", "beta", "gamma"])
    (sandbox / "cache").mkdir()
    (sandbox / "cache" / "preexisting.txt").write_text("untouched", encoding="utf-8")

    fake = FakeEmbeddingTransport()
    _wire_fake(monkeypatch, fake)

    before = _tree_snapshot(sandbox)
    with blocked_sockets():
        rc = _run(["survey", str(corpus), "--json"])
    after = _tree_snapshot(sandbox)

    assert rc == 0
    assert after == before, "dry run must not create, modify, or delete anything"
    assert fake.call_count == 0, "dry run must not embed"
    assert not (sandbox / "atlas").exists()
    capsys.readouterr()


def test_dry_run_previews_corpus_backend_and_target(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _write_corpus(sandbox / "corpus.jsonl", ["alpha", "beta", "gamma"])
    _wire_fake(monkeypatch, FakeEmbeddingTransport(), model="pinned-model")

    rc = _run(["survey", str(corpus), "--json", "-o", "out/atlas"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["applied"] is False
    assert payload["items"] == 3
    assert payload["model"] == "pinned-model"
    assert payload["endpoint"] == "http://fake.invalid/v1"
    assert payload["atlas"].endswith("out/atlas")
    assert payload["atlas_exists"] is False
    assert payload["cache_hits"] == 0
    assert payload["cache_misses"] == 3


def test_dry_run_text_mode_names_the_target_and_says_it_wrote_nothing(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _write_corpus(sandbox / "corpus.jsonl", ["alpha"])
    _wire_fake(monkeypatch, FakeEmbeddingTransport())

    rc = _run(["survey", str(corpus)])
    assert rc == 0

    captured = capsys.readouterr()
    assert "atlas" in captured.out
    assert "--apply" in captured.out + captured.err


def test_dry_run_reports_an_existing_target(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _write_corpus(sandbox / "corpus.jsonl", ["alpha"])
    _wire_fake(monkeypatch, FakeEmbeddingTransport())
    (sandbox / "atlas").mkdir()

    before = _tree_snapshot(sandbox)
    rc = _run(["survey", str(corpus), "--json"])
    assert rc == 0
    assert _tree_snapshot(sandbox) == before

    payload = json.loads(capsys.readouterr().out)
    assert payload["atlas_exists"] is True
    assert payload["would_replace"] is False


def test_dry_run_cache_estimate_counts_hits(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _write_corpus(sandbox / "corpus.jsonl", ["alpha", "beta"])
    fake = FakeEmbeddingTransport()
    _wire_fake(monkeypatch, fake)

    assert _run(["survey", str(corpus), "--apply"]) == 0
    capsys.readouterr()

    rc = _run(["survey", str(corpus), "--json"])
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["cache_hits"] == 2
    assert payload["cache_misses"] == 0


# --------------------------------------------------------------------------
# Apply
# --------------------------------------------------------------------------


def test_apply_writes_a_readable_atlas(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _write_corpus(sandbox / "corpus.jsonl", ["alpha", "beta", "gamma"])
    fake = FakeEmbeddingTransport(dim=8)
    _wire_fake(monkeypatch, fake, model="pinned-model")

    with blocked_sockets():
        rc = _run(["survey", str(corpus), "--apply", "--json", "--seed", "7", "--batch-size", "2"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["applied"] is True
    assert payload["items"] == 3

    atlas_dir = sandbox / "atlas"
    for name in (MANIFEST_FILENAME, RECORDS_FILENAME, VECTORS_FILENAME):
        assert (atlas_dir / name).is_file()

    atlas = read_atlas(atlas_dir)
    assert atlas.manifest.model == "pinned-model"
    assert atlas.manifest.endpoint == "http://fake.invalid/v1"
    assert atlas.manifest.seed == 7
    assert atlas.manifest.params["batch_size"] == 2
    assert atlas.manifest.uncertain == []
    assert [record.id for record in atlas.records] == ["doc-0", "doc-1", "doc-2"]
    assert [record.text for record in atlas.records] == ["alpha", "beta", "gamma"]
    assert atlas.vectors.shape == (3, 8)


def test_apply_preserves_corpus_metadata_on_records(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = sandbox / "corpus.jsonl"
    corpus.write_text(
        json.dumps({"id": "a", "text": "alpha", "source": "wiki", "year": 2020}) + "\n",
        encoding="utf-8",
    )
    _wire_fake(monkeypatch, FakeEmbeddingTransport())

    assert _run(["survey", str(corpus), "--apply"]) == 0
    capsys.readouterr()

    atlas = read_atlas(sandbox / "atlas")
    assert atlas.records[0].metadata == {"source": "wiki", "year": 2020}


def test_duplicate_texts_share_one_vector_row(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _write_corpus(sandbox / "corpus.jsonl", ["alpha", "alpha", "beta"])
    _wire_fake(monkeypatch, FakeEmbeddingTransport())

    assert _run(["survey", str(corpus), "--apply"]) == 0
    capsys.readouterr()

    atlas = read_atlas(sandbox / "atlas")
    assert len(atlas.records) == 3
    assert atlas.vectors.shape[0] == 2
    assert atlas.records[0].vector_index == atlas.records[1].vector_index


def test_apply_onto_an_existing_atlas_exits_1_and_names_force(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _write_corpus(sandbox / "corpus.jsonl", ["alpha"])
    _wire_fake(monkeypatch, FakeEmbeddingTransport())

    assert _run(["survey", str(corpus), "--apply"]) == 0
    capsys.readouterr()

    before = _tree_snapshot(sandbox)
    rc = _run(["survey", str(corpus), "--apply"])
    assert rc == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert "--force" in captured.err
    assert "hint: " in captured.err
    assert _tree_snapshot(sandbox) == before, "a refused apply must leave the atlas untouched"


def test_force_replaces_the_atlas_whole(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    first = _write_corpus(sandbox / "first.jsonl", ["alpha", "stale-only-in-the-old-atlas"])
    _wire_fake(monkeypatch, FakeEmbeddingTransport())

    assert _run(["survey", str(first), "--apply"]) == 0
    capsys.readouterr()
    stray = sandbox / "atlas" / "stray-artifact.txt"
    stray.write_text("left over from the previous survey", encoding="utf-8")

    second = _write_corpus(sandbox / "second.jsonl", ["alpha"])
    rc = _run(["survey", str(second), "--apply", "--force"])
    assert rc == 0
    capsys.readouterr()

    atlas = read_atlas(sandbox / "atlas")
    assert [record.text for record in atlas.records] == ["alpha"]
    assert not stray.exists(), "--force must replace the atlas whole, never merge into it"


def test_force_without_apply_still_writes_nothing(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _write_corpus(sandbox / "corpus.jsonl", ["alpha"])
    _wire_fake(monkeypatch, FakeEmbeddingTransport())

    assert _run(["survey", str(corpus), "--apply"]) == 0
    capsys.readouterr()

    before = _tree_snapshot(sandbox)
    rc = _run(["survey", str(corpus), "--force", "--json"])
    assert rc == 0
    assert _tree_snapshot(sandbox) == before

    payload = json.loads(capsys.readouterr().out)
    assert payload["applied"] is False
    assert payload["would_replace"] is True


# --------------------------------------------------------------------------
# The live-embedding limitation (design note §4)
# --------------------------------------------------------------------------


def test_live_embedding_records_the_float_variance_limitation(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _write_corpus(sandbox / "corpus.jsonl", ["alpha", "beta"])
    _wire_fake(monkeypatch, FakeEmbeddingTransport())

    assert _run(["survey", str(corpus), "--apply"]) == 0
    capsys.readouterr()

    atlas = read_atlas(sandbox / "atlas")
    assert atlas.manifest.limitations, "a live-embedded atlas must carry a limitation"
    joined = " ".join(atlas.manifest.limitations)
    assert "float" in joined
    assert survey_mod.LIVE_EMBEDDING_LIMITATION in atlas.manifest.limitations


def test_fully_cache_served_atlas_carries_no_float_variance_limitation(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _write_corpus(sandbox / "corpus.jsonl", ["alpha", "beta"])
    fake = FakeEmbeddingTransport()
    _wire_fake(monkeypatch, fake)

    assert _run(["survey", str(corpus), "--apply"]) == 0
    calls_after_first = fake.call_count
    capsys.readouterr()

    assert _run(["survey", str(corpus), "--apply", "--force"]) == 0
    capsys.readouterr()

    assert fake.call_count == calls_after_first, "second run must be served entirely from cache"
    atlas = read_atlas(sandbox / "atlas")
    assert survey_mod.LIVE_EMBEDDING_LIMITATION not in atlas.manifest.limitations
    assert atlas.manifest.limitations == []


# --------------------------------------------------------------------------
# Batching, progress, retry
# --------------------------------------------------------------------------


def test_embedding_runs_in_configurable_batches(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _write_corpus(sandbox / "corpus.jsonl", [f"item-{i}" for i in range(5)])
    fake = FakeEmbeddingTransport()
    _wire_fake(monkeypatch, fake)

    assert _run(["survey", str(corpus), "--apply", "--batch-size", "2"]) == 0
    capsys.readouterr()

    assert fake.call_count == 3
    assert [len(call.texts) for call in fake.calls] == [2, 2, 1]


def test_progress_diagnostics_go_to_stderr_and_stdout_stays_pure_json(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _write_corpus(sandbox / "corpus.jsonl", [f"item-{i}" for i in range(4)])
    _wire_fake(monkeypatch, FakeEmbeddingTransport())

    assert _run(["survey", str(corpus), "--apply", "--batch-size", "2", "--json"]) == 0

    captured = capsys.readouterr()
    json.loads(captured.out)  # stdout must be exactly one JSON document
    assert "4" in captured.err, "progress must report items completed on stderr"
    assert captured.err.strip(), "a looping agent needs liveness output"


def test_zero_batch_size_is_a_user_error(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _write_corpus(sandbox / "corpus.jsonl", ["alpha"])
    _wire_fake(monkeypatch, FakeEmbeddingTransport())

    rc = _run(["survey", str(corpus), "--apply", "--batch-size", "0"])
    assert rc == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "hint: " in captured.err


def test_transient_failure_is_retried(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(survey_mod, "RETRY_BACKOFF_SECONDS", 0.0)
    corpus = _write_corpus(sandbox / "corpus.jsonl", ["alpha", "beta"])
    flaky = FlakyTransport(failures=1)
    _wire_fake(monkeypatch, flaky)

    rc = _run(["survey", str(corpus), "--apply", "--retries", "2"])
    assert rc == 0

    captured = capsys.readouterr()
    assert flaky.attempts == 2, "the failed batch must have been retried once"
    assert "retry" in captured.err.lower()
    assert read_atlas(sandbox / "atlas").vectors.shape[0] == 2


def test_retries_are_bounded_and_exhaustion_surfaces_the_backend_error(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(survey_mod, "RETRY_BACKOFF_SECONDS", 0.0)
    corpus = _write_corpus(sandbox / "corpus.jsonl", ["alpha"])
    dead = UnreachableTransport()
    _wire_fake(monkeypatch, dead)

    rc = _run(["survey", str(corpus), "--apply", "--retries", "2"])
    assert rc == 2
    assert len(dead.calls) == 3, "bounded attempts: the initial try plus --retries retries"

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert "hint: " in captured.err
    assert not (sandbox / "atlas").exists(), "a failed survey writes no atlas"


def test_unreachable_backend_exits_2(sandbox: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """The one sanctioned socket in this file — a real connect to a dead port."""
    corpus = _write_corpus(sandbox / "corpus.jsonl", ["alpha"])

    rc = _run(["survey", str(corpus), "--apply", "--retries", "0", "--embed-url", dead_endpoint()])
    assert rc == 2

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert str(dead_endpoint()) in captured.err
    assert "hint: " in captured.err


# --------------------------------------------------------------------------
# Corpus-level user errors
# --------------------------------------------------------------------------


def test_missing_corpus_file_exits_1(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _wire_fake(monkeypatch, FakeEmbeddingTransport())
    rc = _run(["survey", str(sandbox / "nope.jsonl")])
    assert rc == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "hint: " in captured.err


def test_empty_corpus_exits_1_without_writing(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = sandbox / "empty.jsonl"
    corpus.write_text("", encoding="utf-8")
    _wire_fake(monkeypatch, FakeEmbeddingTransport())

    before = _tree_snapshot(sandbox)
    rc = _run(["survey", str(corpus), "--apply"])
    assert rc == 1
    assert _tree_snapshot(sandbox) == before

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "hint: " in captured.err


# --------------------------------------------------------------------------
# Client construction, backoff, and environment-class write failures
# --------------------------------------------------------------------------


def test_build_client_prefers_explicit_arguments_over_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDINGS_EMBED_URL", "http://from-env:9/v1")
    monkeypatch.setenv("EMBEDDINGS_EMBED_MODEL", "model-from-env")

    explicit = survey_mod.build_client("http://explicit:1/v1/", "explicit-model")
    assert explicit.base_url == "http://explicit:1/v1"
    assert explicit.model == "explicit-model"

    inherited = survey_mod.build_client(None, None)
    assert inherited.base_url == "http://from-env:9/v1"
    assert inherited.model == "model-from-env"


def test_retry_backoff_grows_between_attempts(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    slept: list[float] = []
    monkeypatch.setattr(survey_mod.time, "sleep", slept.append)
    corpus = _write_corpus(sandbox / "corpus.jsonl", ["alpha"])
    _wire_fake(monkeypatch, FlakyTransport(failures=2))

    assert _run(["survey", str(corpus), "--apply", "--retries", "2"]) == 0
    capsys.readouterr()

    base = survey_mod.RETRY_BACKOFF_SECONDS
    assert slept == [base, base * 2], "backoff must be bounded and increase per attempt"


def test_negative_retries_is_a_user_error(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _write_corpus(sandbox / "corpus.jsonl", ["alpha"])
    _wire_fake(monkeypatch, FakeEmbeddingTransport())

    assert _run(["survey", str(corpus), "--apply", "--retries", "-1"]) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "hint: " in captured.err


def test_text_preview_of_an_existing_atlas_points_at_force(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _write_corpus(sandbox / "corpus.jsonl", ["alpha"])
    _wire_fake(monkeypatch, FakeEmbeddingTransport())
    (sandbox / "atlas").mkdir()

    assert _run(["survey", str(corpus)]) == 0
    captured = capsys.readouterr()
    assert "already exists" in captured.out
    assert "--force" in captured.out


def test_unwritable_cache_exits_2_without_a_traceback(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A regular file where the cache root should be: every cache read and write
    # under it fails with an OSError that is environment, not usage.
    (sandbox / "not-a-dir").write_text("", encoding="utf-8")
    monkeypatch.setenv("EMBEDDINGS_CACHE_DIR", str(sandbox / "not-a-dir" / "cache"))
    corpus = _write_corpus(sandbox / "corpus.jsonl", ["alpha"])
    _wire_fake(monkeypatch, FakeEmbeddingTransport())

    assert _run(["survey", str(corpus), "--apply"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert "hint: " in captured.err


def test_unwritable_atlas_target_exits_2_without_a_traceback(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    corpus = _write_corpus(sandbox / "corpus.jsonl", ["alpha"])
    _wire_fake(monkeypatch, FakeEmbeddingTransport())
    (sandbox / "atlas").write_text("a file where the atlas directory should be", encoding="utf-8")

    assert _run(["survey", str(corpus), "--apply", "--force"]) == 2
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Traceback" not in captured.err
    assert "hint: " in captured.err
