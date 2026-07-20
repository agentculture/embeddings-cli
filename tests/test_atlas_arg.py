"""Tests for the shared reader-verb atlas entry path.

Covers the t13 acceptance criteria for
``embeddings.cli._commands._atlas_arg``:

* a reader verb with no atlas at the resolved path exits 1 with the exact
  hint ``"run embeddings survey <corpus> --apply first, or pass --atlas
  <path>"``;
* the same reader with the gateway down but an atlas present succeeds --
  reader verbs never require the gateway, proven with
  :func:`tests._fakes.blocked_sockets` around the load itself;
* a version-mismatched atlas exits 2 (environment/setup), distinguishable
  from the "nothing surveyed yet" exit 1 above;
* a path that exists but is not a valid atlas directory is still exit 1, but
  with a message distinguishable from "does not exist at all".
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from embeddings.atlas.io import MANIFEST_FILENAME, write_atlas
from embeddings.atlas.schema import SCHEMA_VERSION, AtlasManifest, AtlasRecord
from embeddings.cli._commands._atlas_arg import (
    DEFAULT_ATLAS_PATH,
    MISSING_ATLAS_HINT,
    add_atlas_argument,
    load_atlas,
    resolve_atlas_path,
)
from embeddings.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from tests._fakes import blocked_sockets

# --------------------------------------------------------------------------
# Fixtures / helpers
# --------------------------------------------------------------------------


def _manifest(**overrides: object) -> AtlasManifest:
    kwargs: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "model": "Qwen/Qwen3-Embedding-0.6B",
        "endpoint": "http://localhost:8002/v1",
        "params": {"batch_size": 32},
        "seed": 0,
        "limitations": [],
        "uncertain": [],
    }
    kwargs.update(overrides)
    return AtlasManifest(**kwargs)  # type: ignore[arg-type]


def _records() -> list[AtlasRecord]:
    return [AtlasRecord(id="0", text="hello world", vector_index=0, metadata={})]


def _vectors() -> np.ndarray:
    return np.array([[0.1, 0.2, 0.3]], dtype=np.float32)


def _write_real_atlas(target: Path, **manifest_overrides: object) -> None:
    write_atlas(_manifest(**manifest_overrides), _records(), _vectors(), path=target)


def _write_version_mismatched_atlas(target: Path, schema_version: int) -> None:
    """Write only ``atlas.json`` at a bad ``schema_version`` -- read_atlas must
    fail on the version check alone, before ever looking at records/vectors,
    so this deliberately leaves ``records.jsonl``/``vectors.npy`` absent."""
    target.mkdir(parents=True)
    payload = {
        "schema_version": schema_version,
        "model": "m",
        "endpoint": "e",
        "params": {},
        "seed": 0,
        "limitations": [],
        "uncertain": [],
    }
    (target / MANIFEST_FILENAME).write_text(json.dumps(payload), encoding="utf-8")


def _args(atlas: str | None = None) -> argparse.Namespace:
    return argparse.Namespace(atlas=atlas)


# --------------------------------------------------------------------------
# add_atlas_argument / resolve_atlas_path
# --------------------------------------------------------------------------


def test_add_atlas_argument_registers_atlas_flag_defaulting_to_none() -> None:
    parser = argparse.ArgumentParser()
    add_atlas_argument(parser)
    args = parser.parse_args([])
    assert args.atlas is None


def test_add_atlas_argument_accepts_an_explicit_path() -> None:
    parser = argparse.ArgumentParser()
    add_atlas_argument(parser)
    args = parser.parse_args(["--atlas", "/tmp/my-atlas"])
    assert args.atlas == "/tmp/my-atlas"


def test_add_atlas_argument_help_mentions_the_flag() -> None:
    parser = argparse.ArgumentParser()
    add_atlas_argument(parser)
    assert "--atlas" in parser.format_help()


def test_resolve_atlas_path_defaults_when_unset() -> None:
    assert resolve_atlas_path(_args()) == Path(DEFAULT_ATLAS_PATH)


def test_resolve_atlas_path_honours_an_explicit_atlas_flag(tmp_path: Path) -> None:
    custom = tmp_path / "somewhere-else"
    assert resolve_atlas_path(_args(str(custom))) == custom


# --------------------------------------------------------------------------
# Criterion 1: no atlas at the resolved path -> exit 1, exact hint
# --------------------------------------------------------------------------


def test_load_atlas_missing_directory_exits_1_with_exact_hint(tmp_path: Path) -> None:
    missing = tmp_path / "does-not-exist"
    args = _args(str(missing))

    with pytest.raises(CliError) as exc_info:
        load_atlas(args)

    err = exc_info.value
    assert err.code == EXIT_USER_ERROR
    assert err.remediation == "run embeddings survey <corpus> --apply first, or pass --atlas <path>"
    assert err.remediation == MISSING_ATLAS_HINT


def test_load_atlas_default_path_missing_exits_1_with_exact_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    args = _args()

    with pytest.raises(CliError) as exc_info:
        load_atlas(args)

    assert exc_info.value.code == EXIT_USER_ERROR
    assert exc_info.value.remediation == MISSING_ATLAS_HINT


def test_load_atlas_path_exists_but_is_not_an_atlas_directory_exits_1(tmp_path: Path) -> None:
    not_an_atlas = tmp_path / "not-an-atlas"
    not_an_atlas.mkdir()
    (not_an_atlas / "some-other-file.txt").write_text("nope", encoding="utf-8")
    args = _args(str(not_an_atlas))

    with pytest.raises(CliError) as exc_info:
        load_atlas(args)

    err = exc_info.value
    assert err.code == EXIT_USER_ERROR
    assert err.remediation == MISSING_ATLAS_HINT
    assert str(not_an_atlas) in err.message


def test_missing_directory_and_invalid_directory_get_distinct_messages(tmp_path: Path) -> None:
    """Both are exit 1, but a caller should be able to tell them apart from the message."""
    missing = tmp_path / "does-not-exist"
    invalid = tmp_path / "invalid"
    invalid.mkdir()
    missing_args = _args(str(missing))
    invalid_args = _args(str(invalid))

    with pytest.raises(CliError) as missing_exc:
        load_atlas(missing_args)
    with pytest.raises(CliError) as invalid_exc:
        load_atlas(invalid_args)

    assert missing_exc.value.code == EXIT_USER_ERROR
    assert invalid_exc.value.code == EXIT_USER_ERROR
    assert missing_exc.value.message != invalid_exc.value.message


# --------------------------------------------------------------------------
# Criterion 2: gateway down + atlas present -> succeeds (no gateway needed)
# --------------------------------------------------------------------------


def test_load_atlas_succeeds_with_no_gateway_reachable(tmp_path: Path) -> None:
    target = tmp_path / "atlas"
    _write_real_atlas(target)

    with blocked_sockets():
        atlas = load_atlas(_args(str(target)))

    assert atlas.manifest.model == "Qwen/Qwen3-Embedding-0.6B"
    assert len(atlas.records) == 1
    assert atlas.records[0].text == "hello world"


def test_load_atlas_default_path_succeeds_with_no_gateway_reachable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_real_atlas(tmp_path / DEFAULT_ATLAS_PATH)

    with blocked_sockets():
        atlas = load_atlas(_args())

    assert len(atlas.records) == 1


# --------------------------------------------------------------------------
# Criterion 3: version mismatch -> exit 2, distinguishable from exit 1
# --------------------------------------------------------------------------


def test_load_atlas_version_mismatch_exits_2(tmp_path: Path) -> None:
    target = tmp_path / "atlas"
    _write_version_mismatched_atlas(target, SCHEMA_VERSION + 1)
    args = _args(str(target))

    with pytest.raises(CliError) as exc_info:
        load_atlas(args)

    err = exc_info.value
    assert err.code == EXIT_ENV_ERROR
    assert str(SCHEMA_VERSION + 1) in err.message
    assert err.remediation


def test_load_atlas_unknown_version_exits_2(tmp_path: Path) -> None:
    target = tmp_path / "atlas"
    _write_version_mismatched_atlas(target, 999999)
    args = _args(str(target))

    with pytest.raises(CliError) as exc_info:
        load_atlas(args)

    assert exc_info.value.code == EXIT_ENV_ERROR


def test_missing_atlas_and_version_mismatch_are_distinguishable_exit_codes(
    tmp_path: Path,
) -> None:
    missing = tmp_path / "does-not-exist"
    mismatched = tmp_path / "mismatched"
    _write_version_mismatched_atlas(mismatched, SCHEMA_VERSION + 1)
    missing_args = _args(str(missing))
    mismatched_args = _args(str(mismatched))

    with pytest.raises(CliError) as missing_exc:
        load_atlas(missing_args)
    with pytest.raises(CliError) as mismatch_exc:
        load_atlas(mismatched_args)

    assert missing_exc.value.code == EXIT_USER_ERROR
    assert mismatch_exc.value.code == EXIT_ENV_ERROR
    assert missing_exc.value.code != mismatch_exc.value.code


def test_load_atlas_version_mismatch_never_reaches_the_missing_atlas_hint(tmp_path: Path) -> None:
    """A version-mismatched atlas is emphatically NOT "nothing surveyed yet" --
    its remediation must not be the exit-1 missing-atlas hint."""
    target = tmp_path / "atlas"
    _write_version_mismatched_atlas(target, SCHEMA_VERSION + 1)
    args = _args(str(target))

    with pytest.raises(CliError) as exc_info:
        load_atlas(args)

    assert exc_info.value.remediation != MISSING_ATLAS_HINT


# --------------------------------------------------------------------------
# load_atlas matches a direct read_atlas call once a valid atlas exists
# --------------------------------------------------------------------------


def test_load_atlas_returns_the_same_atlas_read_atlas_would(tmp_path: Path) -> None:
    from embeddings.atlas.io import read_atlas

    target = tmp_path / "atlas"
    _write_real_atlas(target)

    via_helper = load_atlas(_args(str(target)))
    via_direct = read_atlas(path=target)

    assert via_helper.manifest == via_direct.manifest
    assert via_helper.records == via_direct.records
