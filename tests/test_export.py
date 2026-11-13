"""Tests for ``embeddings export`` — an Atlas with its vectors, as an embeddings-tumbler data file.

The output is checked against :mod:`tests._viewer_contract`, a checker written
in this repo from the viewer's data contract (schema version 1), not imported
from the viewer repo. Every Atlas here is synthetic: no gateway is contacted,
and :func:`tests._fakes.blocked_sockets` proves the verb never opens a socket.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from embeddings import __version__
from embeddings.atlas.io import write_atlas
from embeddings.atlas.schema import SCHEMA_VERSION, AtlasManifest, AtlasRecord
from embeddings.cli import main
from embeddings.cli._commands import export as export_mod
from tests._fakes import blocked_sockets
from tests._viewer_contract import check_pair

REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL = "fake-embedding-model"
# Deliberately distinctive host and port, so "appears nowhere" is a meaningful search.
ENDPOINT = "http://embed-gateway.example.internal:47913/v1"
ENDPOINT_HOST = "embed-gateway.example.internal"
ENDPOINT_PORT = "47913"
LIMITATIONS = ["vectors were embedded live, not served from cache"]
UNCERTAIN = ["region 3 did not survive a stability check"]
PARAMS = {"batch_size": 32, "corpus": "corpus.jsonl"}


def _texts(count: int) -> list[str]:
    return [f"quokka marmalade manifesto number {i} about zeppelins" for i in range(count)]


def _vectors(count: int, dim: int, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    data = rng.normal(size=(count, dim))
    # A few dominant directions, so the variance spectrum is far from flat.
    data[:, :3] *= np.array([9.0, 5.0, 3.0])
    data /= np.linalg.norm(data, axis=1, keepdims=True)
    return data.astype(np.float32)


def make_atlas(
    target: Path,
    *,
    count: int = 12,
    dim: int = 16,
    vectors: np.ndarray | None = None,
    texts: list[str] | None = None,
    ids: list[str] | None = None,
    vector_index: list[int] | None = None,
    limitations: list[str] | None = None,
    metadata: bool = True,
) -> Path:
    texts = texts if texts is not None else _texts(count)
    vectors = vectors if vectors is not None else _vectors(count, dim)
    ids = ids if ids is not None else [f"doc-{i}" for i in range(len(texts))]
    indexes = vector_index if vector_index is not None else list(range(len(texts)))
    manifest = AtlasManifest(
        schema_version=SCHEMA_VERSION,
        model=MODEL,
        endpoint=ENDPOINT,
        params=dict(PARAMS),
        seed=11,
        limitations=list(limitations if limitations is not None else LIMITATIONS),
        uncertain=list(UNCERTAIN),
    )
    records = [
        AtlasRecord(
            id=ids[i],
            text=text,
            vector_index=indexes[i],
            metadata={"topic": f"topic-{i % 3}", "note": f"aside {text}"} if metadata else {},
        )
        for i, text in enumerate(texts)
    ]
    return write_atlas(manifest, records, vectors, path=target)


def run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict]:
    with blocked_sockets():
        rc = main(["export", *argv, "--json"])
    captured = capsys.readouterr()
    stream = captured.out if rc == 0 else captured.err
    return rc, json.loads(stream)


def load_pair(json_path: Path) -> tuple[dict, np.ndarray]:
    data = json.loads(json_path.read_text(encoding="utf-8"))
    block = np.fromfile(json_path.parent / data["vectors"]["file"], dtype="<f4")
    return data, block.reshape(data["count"], data["dimension"])


def snapshot(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def pairwise(matrix: np.ndarray) -> np.ndarray:
    m = np.asarray(matrix, dtype=np.float64)
    sq = (m * m).sum(axis=1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None, :] - 2.0 * m @ m.T, 0.0))


@pytest.fixture
def atlas(tmp_path: Path) -> Path:
    return make_atlas(tmp_path / "atlas")


# --------------------------------------------------------------------------
# Criterion 1: write gating, atomicity, refusals
# --------------------------------------------------------------------------


def test_preview_by_default_writes_nothing(atlas: Path, tmp_path: Path, capsys) -> None:
    before = snapshot(tmp_path)
    out = tmp_path / "view.json"
    rc, payload = run(capsys, "--atlas", str(atlas), "--out", str(out))
    assert rc == 0
    assert snapshot(tmp_path) == before
    assert payload["applied"] is False
    assert payload["files"]["json"]["path"] == str(out)
    assert payload["files"]["block"]["path"] == str(tmp_path / "view.f32")
    assert payload["points"] == 12
    assert payload["dimension"] == 16
    assert payload["files"]["block"]["bytes"] == 12 * 16 * 4
    assert payload["files"]["json"]["bytes"] > 0
    assert payload["reduction"]["kind"] == "none"
    assert payload["text_included"] is True


def test_preview_text_mode_states_what_would_be_written(atlas: Path, tmp_path: Path, capsys):
    with blocked_sockets():
        rc = main(["export", "--atlas", str(atlas), "--out", str(tmp_path / "v.json")])
    out = capsys.readouterr().out
    assert rc == 0
    for expected in ("v.json", "v.f32", "points:", "12", "16", "reduction:", "none", "text"):
        assert expected in out
    assert "Nothing was written" in out
    assert not (tmp_path / "v.json").exists()


def test_apply_writes_a_pair_the_contract_accepts(atlas: Path, tmp_path: Path, capsys) -> None:
    out = tmp_path / "view.json"
    rc, payload = run(capsys, "--atlas", str(atlas), "--out", str(out), "--apply")
    assert rc == 0
    assert payload["applied"] is True
    assert check_pair(out) == []
    data, block = load_pair(out)
    assert data["vectors"]["file"] == "view.f32"
    assert (tmp_path / "view.f32").stat().st_size == 12 * 16 * 4
    # No reduction: the block is the Atlas's own float32 vectors, in record order.
    np.testing.assert_array_equal(block, np.load(atlas / "vectors.npy"))
    assert [p["id"] for p in data["points"]] == [f"doc-{i}" for i in range(12)]


def test_shared_vector_index_is_gathered_per_record(tmp_path: Path, capsys) -> None:
    vectors = _vectors(3, 8)
    atlas = make_atlas(
        tmp_path / "atlas", texts=_texts(4), vectors=vectors, vector_index=[0, 1, 1, 2]
    )
    out = tmp_path / "v.json"
    assert run(capsys, "--atlas", str(atlas), "--out", str(out), "--apply")[0] == 0
    assert check_pair(out) == []
    _, block = load_pair(out)
    np.testing.assert_array_equal(block, vectors[[0, 1, 1, 2]])


@pytest.mark.parametrize("existing", ["view.json", "view.f32"])
def test_existing_target_is_refused_without_force(
    atlas: Path, tmp_path: Path, capsys, existing: str
) -> None:
    (tmp_path / existing).write_bytes(b"keep me")
    before = snapshot(tmp_path)
    rc, err = run(capsys, "--atlas", str(atlas), "--out", str(tmp_path / "view.json"), "--apply")
    assert rc == 1
    assert "--force" in err["remediation"]
    assert snapshot(tmp_path) == before


def test_force_replaces_both_files(atlas: Path, tmp_path: Path, capsys) -> None:
    (tmp_path / "view.json").write_bytes(b"old")
    (tmp_path / "view.f32").write_bytes(b"old")
    out = tmp_path / "view.json"
    rc, payload = run(capsys, "--atlas", str(atlas), "--out", str(out), "--apply", "--force")
    assert rc == 0
    assert payload["would_replace"] is True
    assert check_pair(out) == []


def test_both_files_go_through_the_atomic_writer(atlas: Path, tmp_path: Path, monkeypatch):
    written: list[str] = []
    real = export_mod.write_atomic

    def spy(path, data):
        written.append(Path(path).name)
        real(path, data)

    monkeypatch.setattr(export_mod, "write_atomic", spy)
    assert (
        main(["export", "--atlas", str(atlas), "--out", str(tmp_path / "v.json"), "--apply"]) == 0
    )
    # The block first, then the JSON that names it.
    assert written == ["v.f32", "v.json"]


def test_a_failed_write_leaves_no_partial_or_temp_file(atlas: Path, tmp_path: Path, monkeypatch):
    def boom(path, data):
        raise OSError("disk full")

    monkeypatch.setattr(export_mod, "write_atomic", boom)
    out_dir = tmp_path / "out"
    out_dir.mkdir()
    rc = main(["export", "--atlas", str(atlas), "--out", str(out_dir / "v.json"), "--apply"])
    assert rc == 2
    assert list(out_dir.iterdir()) == []


def test_atlas_without_vectors_is_refused_with_a_survey_hint(tmp_path: Path, capsys) -> None:
    atlas = make_atlas(tmp_path / "atlas")
    (atlas / "vectors.npy").unlink()
    rc, err = run(capsys, "--atlas", str(atlas), "--out", str(tmp_path / "v.json"))
    assert rc == 1
    assert "embeddings survey" in err["remediation"]


def test_committed_study_without_vectors_is_refused_with_a_survey_hint(tmp_path, capsys):
    study = REPO_ROOT / "docs" / "studies" / "atlas-plain"
    assert (study / "atlas.json").is_file() and not (study / "vectors.npy").exists()
    rc, err = run(capsys, "--atlas", str(study), "--out", str(tmp_path / "v.json"), "--apply")
    assert rc == 1
    assert "embeddings survey" in err["remediation"]
    assert not (tmp_path / "v.json").exists()


def test_missing_atlas_is_refused_with_a_survey_hint(tmp_path: Path, capsys) -> None:
    rc, err = run(capsys, "--atlas", str(tmp_path / "nope"), "--out", str(tmp_path / "v.json"))
    assert rc == 1
    assert "embeddings survey" in err["remediation"]


@pytest.mark.parametrize("name", ["view.txt", ".hidden.json", "a b.json", "view"])
def test_out_must_be_a_json_name_with_a_plain_block_name(atlas, tmp_path, capsys, name) -> None:
    rc, err = run(capsys, "--atlas", str(atlas), "--out", str(tmp_path / name))
    assert rc == 1
    assert err["remediation"]


def test_empty_atlas_is_refused(tmp_path: Path, capsys) -> None:
    atlas = make_atlas(tmp_path / "atlas", texts=[], vectors=np.zeros((0, 4), np.float32))
    rc, err = run(capsys, "--atlas", str(atlas), "--out", str(tmp_path / "v.json"))
    assert rc == 1
    assert "no records" in err["message"]


def test_duplicate_record_ids_are_refused(tmp_path: Path, capsys) -> None:
    atlas = make_atlas(tmp_path / "atlas", count=3, ids=["a", "b", "a"])
    rc, err = run(capsys, "--atlas", str(atlas), "--out", str(tmp_path / "v.json"))
    assert rc == 1
    assert '"a"' in err["message"] or "'a'" in err["message"]


def test_non_finite_vectors_are_refused(tmp_path: Path, capsys) -> None:
    vectors = _vectors(4, 8)
    vectors[2, 3] = np.nan
    atlas = make_atlas(tmp_path / "atlas", texts=_texts(4), vectors=vectors)
    rc, err = run(capsys, "--atlas", str(atlas), "--out", str(tmp_path / "v.json"))
    assert rc == 2
    assert "doc-2" in err["message"]


# --------------------------------------------------------------------------
# Criterion 2 (with deviation d1): provenance, endpoint name, size
# --------------------------------------------------------------------------


def test_provenance_copies_model_seed_limitations_uncertain(atlas, tmp_path, capsys) -> None:
    out = tmp_path / "v.json"
    run(capsys, "--atlas", str(atlas), "--out", str(out), "--apply")
    data, _ = load_pair(out)
    prov = data["provenance"]
    assert data["model"] == MODEL
    assert prov["seed"] == 11
    assert prov["limitations"] == LIMITATIONS
    assert prov["uncertain"] == UNCERTAIN
    assert prov["params"] == PARAMS
    assert prov["producer"] == f"embeddings-cli {__version__}"
    assert prov["source"] == f"atlas: {atlas}"
    assert prov["synthetic"] is False
    assert data["schema_version"] == 1
    assert data["kind"] == "embeddings-tumbler/data"
    assert "endpoint" not in prov  # no --endpoint-name: omitted, never derived from the URL


def test_endpoint_name_option_is_written(atlas: Path, tmp_path: Path, capsys) -> None:
    out = tmp_path / "v.json"
    args = ["--atlas", str(atlas), "--out", str(out), "--apply"]
    rc, _ = run(capsys, *args, "--endpoint-name", "lobes gateway")
    assert rc == 0
    assert check_pair(out) == []
    assert load_pair(out)[0]["provenance"]["endpoint"] == "lobes gateway"


@pytest.mark.parametrize(
    "name",
    ["http://gw/v1", "LocalHost gateway", "10.0.0.7", "gateway:8001", "", "a://b"],
)
def test_endpoint_name_that_looks_like_an_address_is_refused(atlas, tmp_path, capsys, name):
    rc, err = run(
        capsys, "--atlas", str(atlas), "--out", str(tmp_path / "v.json"), "--endpoint-name", name
    )
    assert rc == 1
    assert "name" in err["remediation"]
    assert not (tmp_path / "v.json").exists()


@pytest.mark.parametrize("extra", [[], ["--labels-only"], ["--endpoint-name", "lobes gateway"]])
def test_endpoint_url_host_and_port_appear_nowhere_in_the_output(atlas, tmp_path, capsys, extra):
    out = tmp_path / "v.json"
    assert run(capsys, "--atlas", str(atlas), "--out", str(out), "--apply", *extra)[0] == 0
    for path in (out, tmp_path / "v.f32"):
        raw = path.read_bytes()
        for needle in (ENDPOINT, ENDPOINT_HOST, ENDPOINT_PORT, "localhost"):
            assert needle.encode() not in raw, f"{needle!r} leaked into {path.name}"


def test_endpoint_url_inside_copied_provenance_is_redacted(tmp_path: Path, capsys) -> None:
    atlas = make_atlas(
        tmp_path / "atlas",
        limitations=[f"embedded through {ENDPOINT} under load", "a second limitation"],
    )
    out = tmp_path / "v.json"
    assert run(capsys, "--atlas", str(atlas), "--out", str(out), "--apply")[0] == 0
    raw = out.read_text(encoding="utf-8")
    assert ENDPOINT_HOST not in raw and ENDPOINT_PORT not in raw
    limitations = json.loads(raw)["provenance"]["limitations"]
    assert len(limitations) == 2  # copied, never dropped
    assert limitations[0] == "embedded through <embedding endpoint> under load"


def test_size_is_within_five_percent_of_block_plus_records(tmp_path: Path, capsys) -> None:
    count, dim = 120, 1024
    atlas = make_atlas(tmp_path / "atlas", count=count, dim=dim)
    out = tmp_path / "v.json"
    assert run(capsys, "--atlas", str(atlas), "--out", str(out), "--apply")[0] == 0
    block_bytes = count * dim * 4
    assert (tmp_path / "v.f32").stat().st_size == block_bytes
    total = out.stat().st_size + (tmp_path / "v.f32").stat().st_size
    records = (atlas / "records.jsonl").stat().st_size
    assert total <= 1.05 * (block_bytes + records)


# --------------------------------------------------------------------------
# Criterion 3: labels-only
# --------------------------------------------------------------------------


def test_labels_only_contains_no_corpus_text(atlas: Path, tmp_path: Path, capsys) -> None:
    out = tmp_path / "v.json"
    rc, payload = run(capsys, "--atlas", str(atlas), "--out", str(out), "--apply", "--labels-only")
    assert rc == 0
    assert payload["text_included"] is False
    assert check_pair(out) == []
    data, _ = load_pair(out)
    assert data["text_included"] is False
    for point in data["points"]:
        assert point["label"] == point["id"]
        assert "metadata" not in point
        assert "group" not in point
    texts = _texts(12)
    for path in (out, tmp_path / "v.f32"):
        raw = path.read_bytes()
        for text in texts:
            assert text.encode() not in raw
        # Distinctive words of the corpus, and of the metadata, appear nowhere.
        for word in (b"quokka", b"marmalade", b"zeppelins", b"aside", b"topic-"):
            assert word not in raw


def test_text_mode_carries_text_and_metadata(atlas: Path, tmp_path: Path, capsys) -> None:
    out = tmp_path / "v.json"
    run(capsys, "--atlas", str(atlas), "--out", str(out), "--apply")
    data, _ = load_pair(out)
    assert data["text_included"] is True
    assert data["points"][4]["label"] == _texts(12)[4]
    assert data["points"][4]["metadata"]["topic"] == "topic-1"


# --------------------------------------------------------------------------
# Criterion 4: exact rotation and top-k
# --------------------------------------------------------------------------


@pytest.mark.parametrize("count,dim", [(40, 64), (30, 12), (5, 384)])
def test_rotation_preserves_every_pairwise_distance_on_disk(tmp_path, capsys, count, dim):
    atlas = make_atlas(tmp_path / "atlas", count=count, dim=dim)
    out = tmp_path / "v.json"
    rc, _ = run(capsys, "--atlas", str(atlas), "--out", str(out), "--apply", "--rotate")
    assert rc == 0
    assert check_pair(out) == []
    data, block = load_pair(out)
    assert data["dimension"] == dim
    assert data["reduction"] == {
        "kind": "rotation",
        "source_dimension": dim,
        "directions_kept": dim,
        "variance_dropped": 0,
    }
    original = np.load(atlas / "vectors.npy")
    assert np.max(np.abs(pairwise(block) - pairwise(original))) < 1e-5
    # Coordinate 0 carries the most variance, and so on down.
    variances = block.astype(np.float64).var(axis=0)
    assert np.all(np.diff(variances) <= 1e-6)


def test_rotation_is_deterministic(atlas: Path, tmp_path: Path, capsys) -> None:
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    run(capsys, "--atlas", str(atlas), "--out", str(a), "--apply", "--rotate")
    run(capsys, "--atlas", str(atlas), "--out", str(b), "--apply", "--rotate")
    assert (tmp_path / "a.f32").read_bytes() == (tmp_path / "b.f32").read_bytes()


def test_no_reduction_unless_requested(atlas: Path, tmp_path: Path, capsys) -> None:
    out = tmp_path / "v.json"
    run(capsys, "--atlas", str(atlas), "--out", str(out), "--apply")
    data, _ = load_pair(out)
    assert data["reduction"] == {
        "kind": "none",
        "source_dimension": 16,
        "directions_kept": 16,
        "variance_dropped": 0,
    }


def test_top_k_records_directions_kept_and_variance_dropped(tmp_path: Path, capsys) -> None:
    count, dim, k = 50, 32, 5
    atlas = make_atlas(tmp_path / "atlas", count=count, dim=dim)
    out = tmp_path / "v.json"
    rc, payload = run(
        capsys, "--atlas", str(atlas), "--out", str(out), "--apply", "--top-k", str(k)
    )
    assert rc == 0
    assert check_pair(out) == []
    data, block = load_pair(out)
    red = data["reduction"]
    assert data["dimension"] == k and block.shape == (count, k)
    assert red["kind"] == "top-k"
    assert red["source_dimension"] == dim
    assert red["directions_kept"] == k
    # Independent measurement: the share of total variance outside the top k eigenvalues.
    original = np.load(atlas / "vectors.npy").astype(np.float64)
    eig = np.sort(np.linalg.eigvalsh(np.cov(original, rowvar=False)))[::-1]
    expected = 1.0 - eig[:k].sum() / eig.sum()
    assert 0 < red["variance_dropped"] < 1
    assert red["variance_dropped"] == pytest.approx(expected, abs=1e-6)
    assert payload["reduction"] == red
    # A projection never lengthens a distance.
    assert np.all(pairwise(block) <= pairwise(original) + 1e-5)


@pytest.mark.parametrize("k", ["0", "-1", "16", "17"])
def test_top_k_must_be_below_the_source_dimension(atlas, tmp_path, capsys, k) -> None:
    rc, err = run(capsys, "--atlas", str(atlas), "--out", str(tmp_path / "v.json"), "--top-k", k)
    assert rc == 1
    assert err["remediation"]


def test_rotate_and_top_k_are_exclusive(atlas: Path, tmp_path: Path, capsys) -> None:
    with blocked_sockets(), pytest.raises(SystemExit) as exc:
        main(["export", "--atlas", str(atlas), "--rotate", "--top-k", "3", "--json"])
    assert exc.value.code == 1
    assert json.loads(capsys.readouterr().err)["remediation"]


# --------------------------------------------------------------------------
# Criterion 5: map unchanged, atlas untouched
# --------------------------------------------------------------------------


def _map_html(atlas: Path, out: Path) -> bytes:
    assert main(["map", "--atlas", str(atlas), "--html", str(out), "--apply", "--force"]) == 0
    return out.read_bytes()


def test_export_leaves_the_atlas_and_map_output_unchanged(tmp_path: Path, capsys) -> None:
    atlas = make_atlas(tmp_path / "atlas", count=20, dim=12)
    atlas_before = snapshot(atlas)
    map_before = _map_html(atlas, tmp_path / "map.html")
    for extra in ([], ["--rotate"], ["--top-k", "3"], ["--labels-only"]):
        out = tmp_path / "v.json"
        assert (
            main(["export", "--atlas", str(atlas), "--out", str(out), "--apply", "--force", *extra])
            == 0
        )
    assert snapshot(atlas) == atlas_before
    assert _map_html(atlas, tmp_path / "map.html") == map_before
