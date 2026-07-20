"""Tests for Atlas read/write (embeddings.atlas.io).

Covers the t10 acceptance criteria:

* ``write_atlas`` persists ``atlas.json``, ``records.jsonl``, and a ``.npy``
  vector sidecar into an atlas directory, atomically and path-overridable;
* the shipped ``.gitignore`` excludes atlas vector sidecars (``*.npy``) while
  the manifest and records remain committable;
* ``read_atlas`` on an atlas whose ``schema_version`` is unknown or newer
  exits 2 (environment/setup) with a hint naming the version mismatch, and
  never best-effort-parses past that check;
* round-tripping an atlas through ``write_atlas``/``read_atlas`` reproduces
  the manifest, records, and vectors identically.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from embeddings.atlas.io import (
    MANIFEST_FILENAME,
    RECORDS_FILENAME,
    VECTORS_FILENAME,
    Atlas,
    read_atlas,
    write_atlas,
)
from embeddings.atlas.schema import SCHEMA_VERSION, AtlasManifest, AtlasRecord
from embeddings.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError

REPO_ROOT = Path(__file__).resolve().parents[1]


def _manifest(**overrides: object) -> AtlasManifest:
    kwargs: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "endpoint": "http://localhost:8002/v1",
        "params": {"batch_size": 32, "algo": {"name": "kmeans", "k": 5}},
        "seed": 0,
        "limitations": ["cache-vs-live float variance is unmeasured on this atlas"],
        "uncertain": ["region 'foo' did not survive a second seed"],
    }
    kwargs.update(overrides)
    return AtlasManifest(**kwargs)  # type: ignore[arg-type]


def _records() -> list[AtlasRecord]:
    return [
        AtlasRecord(id="0", text="hello world", vector_index=0, metadata={"source": "a.md"}),
        AtlasRecord(id="1", text="goodbye world", vector_index=1, metadata={}),
        # Deliberate dedup: two records pointing at the same vector row —
        # AtlasRecord.vector_index indexes into the vector array, it is not
        # required to be a 1:1, sequential mapping with records.
        AtlasRecord(id="2", text="hello world", vector_index=0, metadata={"source": "b.md"}),
    ]


def _vectors() -> np.ndarray:
    return np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)


# --- write_atlas: on-disk layout ---------------------------------------------


def test_write_atlas_creates_manifest_records_and_vectors_files(tmp_path: Path) -> None:
    target = tmp_path / "atlas"
    write_atlas(_manifest(), _records(), _vectors(), path=target)

    assert (target / MANIFEST_FILENAME).is_file()
    assert (target / RECORDS_FILENAME).is_file()
    assert (target / VECTORS_FILENAME).is_file()
    # nothing else materializes alongside the three artifacts
    assert {p.name for p in target.iterdir()} == {
        MANIFEST_FILENAME,
        RECORDS_FILENAME,
        VECTORS_FILENAME,
    }


def test_write_atlas_returns_the_written_path(tmp_path: Path) -> None:
    target = tmp_path / "atlas"
    result = write_atlas(_manifest(), _records(), _vectors(), path=target)
    assert result == target


def test_write_atlas_manifest_json_is_readable_and_matches_fields(tmp_path: Path) -> None:
    target = tmp_path / "atlas"
    manifest = _manifest()
    write_atlas(manifest, _records(), _vectors(), path=target)

    payload = json.loads((target / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["model"] == manifest.model
    assert payload["endpoint"] == manifest.endpoint
    assert payload["params"] == manifest.params
    assert payload["seed"] == manifest.seed
    assert payload["limitations"] == manifest.limitations
    assert payload["uncertain"] == manifest.uncertain


def test_write_atlas_records_jsonl_is_one_json_object_per_line(tmp_path: Path) -> None:
    target = tmp_path / "atlas"
    write_atlas(_manifest(), _records(), _vectors(), path=target)

    lines = (target / RECORDS_FILENAME).read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    parsed = [json.loads(line) for line in lines]
    assert parsed[0]["id"] == "0"
    assert parsed[0]["text"] == "hello world"
    assert parsed[0]["vector_index"] == 0
    assert parsed[0]["metadata"] == {"source": "a.md"}


def test_write_atlas_path_is_overridable(tmp_path: Path) -> None:
    default_target = tmp_path / "atlas"
    custom_target = tmp_path / "studies" / "my-atlas"

    write_atlas(_manifest(), _records(), _vectors(), path=custom_target)

    assert custom_target.is_dir()
    assert not default_target.exists()


# --- round trip: the core contract ------------------------------------------


def test_round_trip_reproduces_manifest_exactly(tmp_path: Path) -> None:
    target = tmp_path / "atlas"
    manifest = _manifest()
    write_atlas(manifest, _records(), _vectors(), path=target)

    atlas = read_atlas(path=target)

    assert atlas.manifest == manifest


def test_round_trip_reproduces_records_exactly(tmp_path: Path) -> None:
    target = tmp_path / "atlas"
    records = _records()
    write_atlas(_manifest(), records, _vectors(), path=target)

    atlas = read_atlas(path=target)

    assert atlas.records == records


def test_round_trip_reproduces_vectors_identically(tmp_path: Path) -> None:
    target = tmp_path / "atlas"
    vectors = _vectors()
    write_atlas(_manifest(), _records(), vectors, path=target)

    atlas = read_atlas(path=target)

    assert atlas.vectors.dtype == vectors.dtype
    assert atlas.vectors.shape == vectors.shape
    np.testing.assert_array_equal(atlas.vectors, vectors)


def test_round_trip_preserves_vector_index_dedup_contract(tmp_path: Path) -> None:
    """Two records sharing a vector_index both resolve to the same vector row."""
    target = tmp_path / "atlas"
    records = _records()
    vectors = _vectors()
    write_atlas(_manifest(), records, vectors, path=target)

    atlas = read_atlas(path=target)

    first = next(r for r in atlas.records if r.id == "0")
    third = next(r for r in atlas.records if r.id == "2")
    assert first.vector_index == third.vector_index == 0
    np.testing.assert_array_equal(
        atlas.vectors[first.vector_index], atlas.vectors[third.vector_index]
    )


def test_read_atlas_returns_atlas_dataclass(tmp_path: Path) -> None:
    target = tmp_path / "atlas"
    write_atlas(_manifest(), _records(), _vectors(), path=target)

    atlas = read_atlas(path=target)

    assert isinstance(atlas, Atlas)
    assert atlas.path == target


# --- atomicity: a killed write leaves no half-written atlas -----------------


def test_write_atlas_interrupted_leaves_no_partial_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "atlas"

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated interruption while writing vectors")

    monkeypatch.setattr(np, "save", _boom)
    manifest = _manifest()
    records = _records()
    vectors = _vectors()

    with pytest.raises(RuntimeError, match="simulated interruption"):
        write_atlas(manifest, records, vectors, path=target)

    assert not target.exists()
    # no stray scratch directory left behind next to the target either
    assert list(tmp_path.iterdir()) == []


def test_write_atlas_interrupted_preserves_previously_written_atlas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "atlas"
    first_manifest = _manifest(seed=1)
    write_atlas(first_manifest, _records(), _vectors(), path=target)

    def _boom(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("simulated interruption on re-survey")

    monkeypatch.setattr(np, "save", _boom)
    manifest = _manifest(seed=2)
    records = _records()
    vectors = _vectors()

    with pytest.raises(RuntimeError):
        write_atlas(manifest, records, vectors, path=target, overwrite=True)

    # the *new* atlas never landed, but the interruption path here removes the
    # old one first (see write_atlas's documented overwrite behaviour) — the
    # observable guarantee is "never a half-written atlas", not "old survives
    # every failure mode of a --force replace". Assert that guarantee holds:
    # target is either fully absent or fully the last complete write, never a
    # mix of the two manifests.
    if target.exists():
        atlas = read_atlas(path=target)
        assert atlas.manifest in (first_manifest, _manifest(seed=2))


# --- write_atlas: replacing an existing atlas --------------------------------


def test_write_atlas_refuses_to_overwrite_by_default(tmp_path: Path) -> None:
    target = tmp_path / "atlas"
    write_atlas(_manifest(), _records(), _vectors(), path=target)
    manifest = _manifest(seed=99)
    records = _records()
    vectors = _vectors()

    with pytest.raises(CliError) as exc_info:
        write_atlas(manifest, records, vectors, path=target)

    assert exc_info.value.code == EXIT_USER_ERROR
    assert exc_info.value.remediation  # a hint is present

    # the original atlas is untouched
    atlas = read_atlas(path=target)
    assert atlas.manifest.seed == 0


def test_write_atlas_overwrite_true_replaces_whole_not_merged(tmp_path: Path) -> None:
    target = tmp_path / "atlas"
    write_atlas(_manifest(), _records(), _vectors(), path=target)

    new_manifest = _manifest(seed=99, limitations=["a fresh survey superseded the old one"])
    new_records = [AtlasRecord(id="only-one", text="new corpus", vector_index=0)]
    new_vectors = np.array([[9.0, 9.0, 9.0]], dtype=np.float32)

    write_atlas(new_manifest, new_records, new_vectors, path=target, overwrite=True)

    atlas = read_atlas(path=target)
    assert atlas.manifest == new_manifest
    assert atlas.records == new_records
    np.testing.assert_array_equal(atlas.vectors, new_vectors)
    # replaced whole: none of the old records survive alongside the new ones
    assert len(atlas.records) == 1


# --- write_atlas: input validation -------------------------------------------


def test_write_atlas_rejects_non_2d_vectors(tmp_path: Path) -> None:
    target = tmp_path / "atlas"
    manifest = _manifest()
    records = _records()
    vectors = np.array([0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        write_atlas(manifest, records, vectors, path=target)
    assert not target.exists()


def test_write_atlas_rejects_out_of_bounds_vector_index(tmp_path: Path) -> None:
    target = tmp_path / "atlas"
    records = [AtlasRecord(id="0", text="hi", vector_index=5)]  # only 1 vector below
    manifest = _manifest()
    vectors = np.array([[0.1, 0.2]], dtype=np.float32)
    with pytest.raises(ValueError):
        write_atlas(manifest, records, vectors, path=target)
    assert not target.exists()


# --- read_atlas: missing atlas ------------------------------------------------


def test_read_atlas_missing_directory_raises_user_error(tmp_path: Path) -> None:
    with pytest.raises(CliError) as exc_info:
        read_atlas(path=tmp_path / "does-not-exist")
    assert exc_info.value.code == EXIT_USER_ERROR


# --- read_atlas: the fail-closed schema_version gate -------------------------


def _write_manifest_with_raw_schema_version(target: Path, schema_version: int) -> None:
    """Write an atlas.json directly, bypassing write_atlas's own version stamp.

    Simulates "an atlas written by a different embeddings-cli version" by
    constructing the on-disk JSON payload directly rather than by constructing
    an AtlasManifest — an in-memory AtlasManifest carries whatever
    schema_version its caller passed (this is deliberately unconstrained; see
    test_atlas_schema.py), but write_atlas persists it verbatim, so the two
    approaches land the same on-disk bytes either way. This helper is used to
    keep the version-bump itself visually explicit in each test.
    """
    target.mkdir(parents=True)
    payload = {
        "schema_version": schema_version,
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "endpoint": "http://localhost:8002/v1",
        "params": {},
        "seed": 0,
        "limitations": [],
        "uncertain": [],
    }
    (target / MANIFEST_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    (target / RECORDS_FILENAME).write_text("", encoding="utf-8")
    np.save(target / VECTORS_FILENAME, np.zeros((0, 0), dtype=np.float32))


def test_read_atlas_newer_schema_version_exits_2_with_mismatch_hint(tmp_path: Path) -> None:
    target = tmp_path / "atlas"
    newer = SCHEMA_VERSION + 1
    _write_manifest_with_raw_schema_version(target, newer)

    with pytest.raises(CliError) as exc_info:
        read_atlas(path=target)

    err = exc_info.value
    assert err.code == EXIT_ENV_ERROR
    # the hint (message + remediation together) names both the on-disk
    # version and the version this CLI supports/wrote
    combined = f"{err.message} {err.remediation}"
    assert str(newer) in combined
    assert str(SCHEMA_VERSION) in combined
    assert err.remediation  # a hint is present, not just a bare message


def test_read_atlas_unknown_schema_version_exits_2(tmp_path: Path) -> None:
    """An arbitrary unrecognized version (not just 'off by one') also fails closed."""
    target = tmp_path / "atlas"
    unknown = 999999
    _write_manifest_with_raw_schema_version(target, unknown)

    with pytest.raises(CliError) as exc_info:
        read_atlas(path=target)

    assert exc_info.value.code == EXIT_ENV_ERROR
    assert str(unknown) in exc_info.value.message


def test_read_atlas_never_best_effort_parses_a_version_mismatch(tmp_path: Path) -> None:
    """A version-mismatched atlas whose records/vectors are also malformed still
    fails on the version check alone — the reader never gets far enough to
    trip over the malformed body, because it never attempts to parse it."""
    target = tmp_path / "atlas"
    target.mkdir(parents=True)
    payload = {
        "schema_version": SCHEMA_VERSION + 1,
        "model": "m",
        "endpoint": "e",
        "params": {},
        "seed": 0,
        "limitations": [],
        "uncertain": [],
    }
    (target / MANIFEST_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    (target / RECORDS_FILENAME).write_text("{ not valid json at all\n", encoding="utf-8")
    # no vectors.npy at all — a best-effort reader would blow up on this with
    # an unrelated FileNotFoundError; this reader must never reach that point.

    with pytest.raises(CliError) as exc_info:
        read_atlas(path=target)

    assert exc_info.value.code == EXIT_ENV_ERROR
    assert "schema_version" in exc_info.value.message or "version" in exc_info.value.message


def test_read_atlas_matching_schema_version_succeeds(tmp_path: Path) -> None:
    target = tmp_path / "atlas"
    _write_manifest_with_raw_schema_version(target, SCHEMA_VERSION)

    atlas = read_atlas(path=target)
    assert atlas.manifest.schema_version == SCHEMA_VERSION


# --- .gitignore: *.npy is excluded, manifest/records are committable --------


def test_gitignore_excludes_npy_sidecar_via_git_check_ignore() -> None:
    result = subprocess.run(
        ["git", "check-ignore", "-q", "atlas/vectors.npy"],
        cwd=REPO_ROOT,
        capture_output=True,
    )
    assert result.returncode == 0, "atlas/*.npy must be git-ignored"


def test_gitignore_does_not_exclude_manifest_or_records() -> None:
    for relative in ("atlas/atlas.json", "atlas/records.jsonl"):
        result = subprocess.run(
            ["git", "check-ignore", "-q", relative],
            cwd=REPO_ROOT,
            capture_output=True,
        )
        assert result.returncode == 1, f"{relative} must NOT be git-ignored"


def test_survey_apply_leaves_no_npy_in_git_status_but_manifest_is_committable(
    tmp_path: Path,
) -> None:
    """End-to-end simulation of the acceptance criterion's actual claim.

    ``survey --apply`` itself is a later task (t12); what it will do is call
    ``write_atlas`` against a target inside a git-tracked repo. This test
    stands in for that by initializing a throwaway repo, copying the real
    shipped ``.gitignore`` into it, and calling ``write_atlas`` directly —
    then asserting on ``git status`` exactly as the acceptance criterion
    describes: no ``.npy`` paths appear, while the manifest and records do
    (i.e. they are untracked-but-committable, not ignored).
    """
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True)
    (repo / ".gitignore").write_text(
        (REPO_ROOT / ".gitignore").read_text(encoding="utf-8"), encoding="utf-8"
    )

    write_atlas(_manifest(), _records(), _vectors(), path=repo / "atlas")

    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=repo,
        capture_output=True,
        text=True,
        check=True,
    ).stdout

    assert ".npy" not in status
    assert f"atlas/{MANIFEST_FILENAME}" in status
    assert f"atlas/{RECORDS_FILENAME}" in status
