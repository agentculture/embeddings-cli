"""Tests for ``embeddings neighbors`` — nearest-Atlas-items lookup by text or item id.

Two properties get disproportionate attention here, because they are the
acceptance criteria that make this verb trustworthy rather than merely
convenient:

* **The item-id path opens no socket, ever.** ``test_id_path_never_touches_a_socket``
  wraps a real run in ``blocked_sockets()`` -- if this module ever reached past
  the Atlas to the embedding gateway while resolving ``--id``, that call would
  raise instead of succeeding. This is the whole point of accepting an id
  alternative to free text: reader verbs must work with no gateway.
* **Disambiguation is a flag, never a guess.** ``--id`` is required to read the
  positional as an Atlas item id; without it, the positional is *always*
  embedded as free text, even when it happens to spell an existing id.
  ``test_query_matching_an_existing_id_is_still_embedded_without_the_id_flag``
  proves the resolution is predictable rather than heuristic.

No test here opens a socket except the one explicitly proving disambiguation
correctness is unaffected by the embedding gateway being reachable at all
(the id-path test uses ``blocked_sockets()`` precisely because it must never
need one). Every free-text test wires a :class:`FakeEmbeddingTransport` into a
real :class:`EmbeddingClient` through the ``build_client`` seam, matching
``survey``'s pattern.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from embeddings.atlas.io import write_atlas
from embeddings.atlas.schema import SCHEMA_VERSION, AtlasManifest, AtlasRecord
from embeddings.backends import EmbeddingClient
from embeddings.cli import main
from embeddings.cli._commands import neighbors as neighbors_mod
from tests._fakes import FakeEmbeddingTransport, blocked_sockets, wire_transport

# --------------------------------------------------------------------------
# Fixture atlas: four hand-placed 8-dim vectors with exactly computable
# cosine distances, so ordering and the boundary-gap profile are asserted
# against numbers worked out by hand, not just "some ordering".
# --------------------------------------------------------------------------

_DIM = 8


def _vec(x: float, y: float) -> list[float]:
    return [x, y, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


# a=[1,0,...]; b close to a; c orthogonal to a; d opposite a.
_VEC_A = _vec(1.0, 0.0)
_VEC_B = _vec(0.9, 0.1)
_VEC_C = _vec(0.0, 1.0)
_VEC_D = _vec(-1.0, 0.0)


def _cosine_distance(u: list[float], v: list[float]) -> float:
    dot = sum(ux * vx for ux, vx in zip(u, v))
    norm_u = math.sqrt(sum(ux * ux for ux in u))
    norm_v = math.sqrt(sum(vx * vx for vx in v))
    return 1.0 - dot / (norm_u * norm_v)


def _manifest(**overrides: object) -> AtlasManifest:
    kwargs: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "model": "fake-embedding-model",
        "endpoint": "http://fake.invalid/v1",
        "params": {},
        "seed": 0,
        "limitations": [],
        "uncertain": [],
    }
    kwargs.update(overrides)
    return AtlasManifest(**kwargs)  # type: ignore[arg-type]


def _write_fixture_atlas(target: Path, **manifest_overrides: object) -> None:
    records = [
        AtlasRecord(id="a", text="a", vector_index=0, metadata={"tag": "near-origin"}),
        AtlasRecord(id="b", text="b", vector_index=1, metadata={}),
        AtlasRecord(id="c", text="c", vector_index=2, metadata={}),
        AtlasRecord(id="d", text="d", vector_index=3, metadata={}),
    ]
    vectors = np.array([_VEC_A, _VEC_B, _VEC_C, _VEC_D], dtype=np.float32)
    write_atlas(_manifest(**manifest_overrides), records, vectors, path=target)


def _wire_fake(monkeypatch: pytest.MonkeyPatch, fake: Any, *, model: str | None = None) -> None:
    """Replace neighbors' client factory with one backed by ``fake``. Opens no socket."""

    def _build(base_url: str | None, model_name: str | None) -> EmbeddingClient:
        return EmbeddingClient(
            base_url=base_url or "http://fake.invalid/v1",
            model=model_name or model or "fake-embedding-model",
            transport=wire_transport(fake),
        )

    monkeypatch.setattr(neighbors_mod, "build_client", _build)


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("EMBEDDINGS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("EMBEDDINGS_EMBED_URL", raising=False)
    monkeypatch.delenv("EMBEDDINGS_EMBED_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def _run(argv: list[str]) -> int:
    return main(argv)


# --------------------------------------------------------------------------
# Registration + parser contract
# --------------------------------------------------------------------------


def test_neighbors_is_registered_on_the_root_parser(capsys: pytest.CaptureFixture[str]) -> None:
    rc = _run([])
    assert rc == 0
    assert "neighbors" in capsys.readouterr().out


def test_neighbors_declares_its_own_json_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        _run(["neighbors", "--help"])
    assert exc.value.code == 0
    assert "--json" in capsys.readouterr().out


def test_every_neighbors_flag_has_help_text(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        _run(["neighbors", "--help"])
    help_text = capsys.readouterr().out
    for flag in ("--atlas", "--id", "--top-k", "-k", "--embed-url", "--embed-model", "--seed"):
        assert flag in help_text


def test_missing_query_is_a_structured_parse_error(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        _run(["neighbors"])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err.startswith("error: ")
    assert "hint: " in captured.err


# --------------------------------------------------------------------------
# Missing atlas -- reuses the shared _atlas_arg contract
# --------------------------------------------------------------------------


def test_missing_atlas_exits_1_with_the_shared_hint(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = _run(["neighbors", "some query", "--atlas", str(sandbox / "nope")])
    assert rc == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "hint: " in captured.err
    assert "survey" in captured.err


# --------------------------------------------------------------------------
# Free-text path
# --------------------------------------------------------------------------


def test_free_text_query_embeds_and_returns_ranked_neighbors(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    atlas_dir = sandbox / "atlas"
    _write_fixture_atlas(atlas_dir)
    fake = FakeEmbeddingTransport(dim=_DIM)
    fake.set_override("query-x", _VEC_A)
    _wire_fake(monkeypatch, fake)

    rc = _run(["neighbors", "query-x", "--atlas", str(atlas_dir), "--top-k", "4", "--json"])
    assert rc == 0
    assert fake.call_count == 1
    assert fake.last_call is not None
    assert fake.last_call.texts == ["query-x"]

    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "text"
    assert payload["query"] == "query-x"
    ids = [n["id"] for n in payload["neighbors"]]
    assert ids == ["a", "b", "c", "d"], "must be sorted nearest-first by cosine distance"
    assert payload["neighbors"][0]["distance"] == pytest.approx(0.0, abs=1e-6)
    assert payload["neighbors"][1]["distance"] == pytest.approx(
        _cosine_distance(_VEC_A, _VEC_B), abs=1e-6
    )
    assert payload["neighbors"][2]["distance"] == pytest.approx(1.0, abs=1e-6)
    assert payload["neighbors"][3]["distance"] == pytest.approx(2.0, abs=1e-6)


def test_query_matching_an_existing_id_is_still_embedded_without_the_id_flag(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """No guessing: a positional that happens to equal an Atlas item id is free
    text unless --id says otherwise -- the disambiguation is the flag, never a
    heuristic over the string's shape."""
    atlas_dir = sandbox / "atlas"
    _write_fixture_atlas(atlas_dir)
    fake = FakeEmbeddingTransport(dim=_DIM)
    _wire_fake(monkeypatch, fake)

    rc = _run(["neighbors", "a", "--atlas", str(atlas_dir), "--json"])
    assert rc == 0
    assert fake.call_count == 1, "the literal string 'a' must have been sent to the gateway"
    assert fake.last_call is not None
    assert fake.last_call.texts == ["a"]

    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "text"


def test_free_text_defaults_the_query_model_to_the_atlas_model(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    atlas_dir = sandbox / "atlas"
    _write_fixture_atlas(atlas_dir, model="atlas-native-model")

    captured_models: list[str | None] = []

    def _build(base_url: str | None, model_name: str | None) -> EmbeddingClient:
        captured_models.append(model_name)
        return EmbeddingClient(
            base_url=base_url or "http://fake.invalid/v1",
            model=model_name or "atlas-native-model",
            transport=wire_transport(FakeEmbeddingTransport(dim=_DIM)),
        )

    monkeypatch.setattr(neighbors_mod, "build_client", _build)

    rc = _run(["neighbors", "query-x", "--atlas", str(atlas_dir), "--json"])
    assert rc == 0
    capsys.readouterr()
    assert captured_models == ["atlas-native-model"]


def test_explicit_embed_model_flag_overrides_the_atlas_default(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    atlas_dir = sandbox / "atlas"
    _write_fixture_atlas(atlas_dir, model="atlas-native-model")

    captured_models: list[str | None] = []

    def _build(base_url: str | None, model_name: str | None) -> EmbeddingClient:
        captured_models.append(model_name)
        return EmbeddingClient(
            base_url=base_url or "http://fake.invalid/v1",
            model=model_name or "atlas-native-model",
            transport=wire_transport(FakeEmbeddingTransport(dim=_DIM)),
        )

    monkeypatch.setattr(neighbors_mod, "build_client", _build)

    rc = _run(
        [
            "neighbors",
            "query-x",
            "--atlas",
            str(atlas_dir),
            "--embed-model",
            "override-model",
            "--json",
        ]
    )
    assert rc == 0
    capsys.readouterr()
    assert captured_models == ["override-model"]


def test_dimension_mismatch_is_a_user_error_naming_embed_model(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    atlas_dir = sandbox / "atlas"
    _write_fixture_atlas(atlas_dir)  # atlas vectors are dim=8
    fake = FakeEmbeddingTransport(dim=4)  # query embeds at a different dimension
    _wire_fake(monkeypatch, fake)

    rc = _run(["neighbors", "query-x", "--atlas", str(atlas_dir)])
    assert rc == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "--embed-model" in captured.err
    assert "hint: " in captured.err


# --------------------------------------------------------------------------
# Atlas item-id path -- must never touch a socket
# --------------------------------------------------------------------------


def test_id_path_never_touches_a_socket(sandbox: Path, capsys: pytest.CaptureFixture[str]) -> None:
    atlas_dir = sandbox / "atlas"
    _write_fixture_atlas(atlas_dir)

    with blocked_sockets():
        rc = _run(["neighbors", "a", "--id", "--atlas", str(atlas_dir), "--top-k", "3", "--json"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["source"] == "atlas-item"
    assert payload["query"] == "a"

    ids = [n["id"] for n in payload["neighbors"]]
    assert "a" not in ids, "the queried item must not be its own neighbor"
    assert ids == ["b", "c", "d"], "must be sorted nearest-first by cosine distance"
    assert payload["neighbors"][0]["distance"] == pytest.approx(
        _cosine_distance(_VEC_A, _VEC_B), abs=1e-6
    )
    assert payload["neighbors"][1]["distance"] == pytest.approx(1.0, abs=1e-6)
    assert payload["neighbors"][2]["distance"] == pytest.approx(2.0, abs=1e-6)


def test_unknown_item_id_exits_1_with_a_helpful_hint(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    atlas_dir = sandbox / "atlas"
    _write_fixture_atlas(atlas_dir)

    with blocked_sockets():
        rc = _run(["neighbors", "no-such-id", "--id", "--atlas", str(atlas_dir)])
    assert rc == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "no-such-id" in captured.err
    assert "hint: " in captured.err


def test_id_path_top_k_larger_than_available_records_returns_what_exists(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    atlas_dir = sandbox / "atlas"
    _write_fixture_atlas(atlas_dir)

    with blocked_sockets():
        rc = _run(["neighbors", "a", "--id", "--atlas", str(atlas_dir), "--top-k", "50", "--json"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["k"] == 50
    assert payload["returned"] == 3
    assert len(payload["neighbors"]) == 3


# --------------------------------------------------------------------------
# --json shape, params, and seed
# --------------------------------------------------------------------------


def test_json_shape_includes_metric_params_and_seed(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    atlas_dir = sandbox / "atlas"
    _write_fixture_atlas(atlas_dir)

    with blocked_sockets():
        rc = _run(
            [
                "neighbors",
                "a",
                "--id",
                "--atlas",
                str(atlas_dir),
                "--top-k",
                "2",
                "--seed",
                "7",
                "--json",
            ]
        )
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    for key in (
        "query",
        "source",
        "atlas",
        "metric",
        "k",
        "returned",
        "neighbors",
        "boundary",
        "params",
        "seed",
    ):
        assert key in payload, f"missing key: {key}"

    assert payload["metric"] == "cosine"
    assert payload["seed"] == 7
    assert payload["k"] == 2
    assert payload["returned"] == 2

    params = payload["params"]
    assert params["metric"] == "cosine"
    assert params["k"] == 2
    assert params["source"] == "atlas-item"
    assert params["atlas"] == str(atlas_dir)

    for neighbor in payload["neighbors"]:
        for key in ("id", "text", "distance", "metadata"):
            assert key in neighbor


def test_neighbors_are_never_mixed_between_stdout_and_stderr(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    atlas_dir = sandbox / "atlas"
    _write_fixture_atlas(atlas_dir)

    with blocked_sockets():
        rc = _run(["neighbors", "a", "--id", "--atlas", str(atlas_dir), "--json"])
    assert rc == 0

    captured = capsys.readouterr()
    json.loads(captured.out)  # stdout is exactly one JSON document
    assert captured.err == ""


def test_text_mode_renders_query_metric_and_boundary(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    atlas_dir = sandbox / "atlas"
    _write_fixture_atlas(atlas_dir)

    with blocked_sockets():
        rc = _run(["neighbors", "a", "--id", "--atlas", str(atlas_dir)])
    assert rc == 0

    captured = capsys.readouterr()
    assert "cosine" in captured.out
    assert "b" in captured.out
    assert "boundary" in captured.out.lower()


# --------------------------------------------------------------------------
# Boundary description: the distance-gap profile
# --------------------------------------------------------------------------


def test_boundary_reports_the_largest_gap_between_b_c_and_c_d(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """b and c are ~1.0 apart in cosine distance, c and d ~1.0 apart too, but b
    sits far closer to the query than either -- so the largest single jump in
    the sorted-distance profile is measurable and known ahead of time."""
    atlas_dir = sandbox / "atlas"
    _write_fixture_atlas(atlas_dir)

    with blocked_sockets():
        rc = _run(["neighbors", "a", "--id", "--atlas", str(atlas_dir), "--top-k", "3", "--json"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    distances = [n["distance"] for n in payload["neighbors"]]
    boundary = payload["boundary"]

    expected_gaps = [distances[i + 1] - distances[i] for i in range(len(distances) - 1)]
    assert boundary["gaps"] == pytest.approx(expected_gaps, abs=1e-6)
    assert boundary["largest_gap_index"] == expected_gaps.index(max(expected_gaps))
    assert boundary["largest_gap"] == pytest.approx(max(expected_gaps), abs=1e-6)
    assert isinstance(boundary["note"], str) and boundary["note"]


def test_boundary_with_fewer_than_two_neighbors_reports_no_gap(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    atlas_dir = sandbox / "atlas"
    _write_fixture_atlas(atlas_dir)

    with blocked_sockets():
        rc = _run(["neighbors", "a", "--id", "--atlas", str(atlas_dir), "--top-k", "1", "--json"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    boundary = payload["boundary"]
    assert boundary["gaps"] == []
    assert boundary["largest_gap_index"] is None
    assert boundary["largest_gap"] is None


# --------------------------------------------------------------------------
# k validation
# --------------------------------------------------------------------------


def test_zero_top_k_is_a_user_error(sandbox: Path, capsys: pytest.CaptureFixture[str]) -> None:
    atlas_dir = sandbox / "atlas"
    _write_fixture_atlas(atlas_dir)

    with blocked_sockets():
        rc = _run(["neighbors", "a", "--id", "--atlas", str(atlas_dir), "--top-k", "0"])
    assert rc == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "hint: " in captured.err


def test_negative_top_k_is_a_user_error(sandbox: Path, capsys: pytest.CaptureFixture[str]) -> None:
    atlas_dir = sandbox / "atlas"
    _write_fixture_atlas(atlas_dir)

    with blocked_sockets():
        rc = _run(["neighbors", "a", "--id", "--atlas", str(atlas_dir), "--top-k", "-1"])
    assert rc == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "hint: " in captured.err


# --------------------------------------------------------------------------
# A lone atlas item queried by id: no neighbors, not an error
# --------------------------------------------------------------------------


def test_id_path_with_only_the_queried_item_returns_zero_neighbors(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    atlas_dir = sandbox / "atlas"
    write_atlas(
        _manifest(),
        [AtlasRecord(id="only", text="only", vector_index=0, metadata={})],
        np.array([_VEC_A], dtype=np.float32),
        path=atlas_dir,
    )

    with blocked_sockets():
        rc = _run(["neighbors", "only", "--id", "--atlas", str(atlas_dir), "--json"])
    assert rc == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["neighbors"] == []
    assert payload["returned"] == 0


def test_id_path_with_only_the_queried_item_text_mode_says_no_neighbors(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    atlas_dir = sandbox / "atlas"
    write_atlas(
        _manifest(),
        [AtlasRecord(id="only", text="only", vector_index=0, metadata={})],
        np.array([_VEC_A], dtype=np.float32),
        path=atlas_dir,
    )

    with blocked_sockets():
        rc = _run(["neighbors", "only", "--id", "--atlas", str(atlas_dir)])
    assert rc == 0
    assert "no neighbors found" in capsys.readouterr().out.lower()


def test_zero_magnitude_query_vector_is_a_user_error(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A stored all-zero vector cannot be compared by cosine distance at all --
    this is a degenerate Atlas item, reported as a usage error rather than a
    NaN silently poisoning the sort."""
    atlas_dir = sandbox / "atlas"
    zero_vector = [0.0] * _DIM
    write_atlas(
        _manifest(),
        [
            AtlasRecord(id="zero", text="zero", vector_index=0, metadata={}),
            AtlasRecord(id="a", text="a", vector_index=1, metadata={}),
        ],
        np.array([zero_vector, _VEC_A], dtype=np.float32),
        path=atlas_dir,
    )

    with blocked_sockets():
        rc = _run(["neighbors", "zero", "--id", "--atlas", str(atlas_dir)])
    assert rc == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert "hint: " in captured.err


# --------------------------------------------------------------------------
# build_client seam, exercised directly (mirrors survey.build_client's own test)
# --------------------------------------------------------------------------


def test_build_client_prefers_explicit_arguments_over_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("EMBEDDINGS_EMBED_URL", "http://from-env:9/v1")
    monkeypatch.setenv("EMBEDDINGS_EMBED_MODEL", "model-from-env")

    explicit = neighbors_mod.build_client("http://explicit:1/v1/", "explicit-model")
    assert explicit.base_url == "http://explicit:1/v1"
    assert explicit.model == "explicit-model"

    inherited = neighbors_mod.build_client(None, None)
    assert inherited.base_url == "http://from-env:9/v1"
    assert inherited.model == "model-from-env"
