"""Tests for ``embeddings regions`` — clustering an Atlas into semantic regions.

Three acceptance criteria drive this file, and one hard-won caution shapes how
they are asserted.

The criteria (t14, design note ``docs/design-notes/semantic-measurement.md`` §3):

1. ``regions`` clusters the Atlas and emits per-region membership, size, and
   representative items;
2. the output carries ``params``, ``seed``, and a stability measure *per
   region*, in a ``--json`` shape asserted here;
3. a region that survives only one parameter setting is reported as a **finding
   about the parameters**, not as a fact about the model — and is still
   reported, never suppressed.

The caution comes from the stability task (t11): a stability assertion is only
as strong as the variability of what it measures. A clustering double that is a
pure function of its input passes a determinism test even with the seeding
deliberately broken. So the clustering under test here is the *real* one, and
``test_clustering_is_genuinely_seed_sensitive`` /
``test_clustering_is_genuinely_composition_sensitive`` are guard tests proving
it is not trivially constant — without them, every "stable" verdict in this
file would be unfalsifiable.

Nothing here opens a socket: ``regions`` is a reader verb and must work with no
gateway at all, which :func:`tests._fakes.blocked_sockets` proves directly.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from embeddings.atlas.io import write_atlas
from embeddings.atlas.schema import SCHEMA_VERSION, AtlasManifest, AtlasRecord
from embeddings.cli import main
from embeddings.cli._commands import regions as regions_mod
from embeddings.cli._commands._atlas_arg import MISSING_ATLAS_HINT
from embeddings.measure import (
    VERDICT_STABLE,
    VERDICT_UNVERIFIED,
    adjusted_rand_index,
)
from tests._fakes import blocked_sockets, deterministic_vector

# --------------------------------------------------------------------------
# Fixtures: two atlases with deliberately different amounts of real structure
# --------------------------------------------------------------------------

DIM = 8


def _write_atlas(target: Path, texts: list[str], vectors: np.ndarray) -> Path:
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
        AtlasRecord(id=text, text=text, vector_index=index, metadata={})
        for index, text in enumerate(texts)
    ]
    return write_atlas(manifest, records, vectors.astype(np.float32), path=target)


def blob_atlas(target: Path, *, blobs: int = 3, per_blob: int = 8, spread: float = 0.02) -> Path:
    """An atlas with ``blobs`` tight, well-separated groups — genuine structure.

    Each item sits on an axis-aligned centre plus a small jitter derived from
    :func:`tests._fakes.deterministic_vector`, so the whole fixture is
    reproducible from its text alone and no RNG state leaks between tests.
    """
    texts: list[str] = []
    rows: list[np.ndarray] = []
    for blob in range(blobs):
        centre = np.zeros(DIM, dtype=np.float64)
        centre[blob % DIM] = 1.0
        for item in range(per_blob):
            text = f"blob{blob}-item{item}"
            jitter = np.asarray(deterministic_vector(text, DIM), dtype=np.float64) * spread
            texts.append(text)
            rows.append(centre + jitter)
    return _write_atlas(target, texts, np.asarray(rows))


def ambiguous_atlas(target: Path, *, items: int = 24) -> Path:
    """An atlas with no cluster structure — hash-derived directions, nothing else.

    Clustering it still *produces* regions (k-means always does), which is
    exactly the case §3 is about: the regions are artifacts of ``k`` and the
    seed, and must be reported as such rather than asserted.
    """
    texts = [f"noise-{index}" for index in range(items)]
    rows = [np.asarray(deterministic_vector(text, DIM), dtype=np.float64) for text in texts]
    return _write_atlas(target, texts, np.asarray(rows))


def run_regions(*argv: str, capsys: pytest.CaptureFixture[str]) -> tuple[int, dict]:
    """Invoke ``embeddings regions ... --json`` and return ``(exit_code, payload)``."""
    with blocked_sockets():
        code = main(["regions", *argv, "--json"])
    out = capsys.readouterr().out
    return code, json.loads(out)


# --------------------------------------------------------------------------
# Criterion 1: membership, size, representative items
# --------------------------------------------------------------------------


def test_regions_recovers_the_blobs_with_membership_and_sizes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    atlas = blob_atlas(tmp_path / "atlas", blobs=3, per_blob=8)

    code, payload = run_regions("--atlas", str(atlas), "-k", "3", capsys=capsys)

    assert code == 0
    assert len(payload["regions"]) == 3
    assert sorted(region["size"] for region in payload["regions"]) == [8, 8, 8]

    # Every item lands in exactly one region, and membership recovers the blobs.
    members = [member for region in payload["regions"] for member in region["members"]]
    assert len(members) == 24
    assert len(set(members)) == 24
    for region in payload["regions"]:
        prefixes = {member.split("-")[0] for member in region["members"]}
        assert len(prefixes) == 1, "a region mixed two blobs"


def test_regions_emit_representative_items_nearest_the_centroid(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    atlas = blob_atlas(tmp_path / "atlas", blobs=3, per_blob=8)

    _, payload = run_regions(
        "--atlas", str(atlas), "-k", "3", "--representatives", "2", capsys=capsys
    )

    for region in payload["regions"]:
        representatives = region["representatives"]
        assert len(representatives) == 2
        assert [item["id"] for item in representatives] == sorted(
            (item["id"] for item in representatives),
            key=lambda ident: next(
                rep["distance"] for rep in representatives if rep["id"] == ident
            ),
        )
        for item in representatives:
            assert item["id"] in region["members"]
            assert item["text"]
            assert item["distance"] >= 0.0
        # Nearest-first ordering.
        distances = [item["distance"] for item in representatives]
        assert distances == sorted(distances)


def test_representatives_never_exceed_the_region_size(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    atlas = blob_atlas(tmp_path / "atlas", blobs=2, per_blob=3)

    _, payload = run_regions(
        "--atlas", str(atlas), "-k", "2", "--representatives", "50", capsys=capsys
    )

    for region in payload["regions"]:
        assert len(region["representatives"]) == region["size"]


# --------------------------------------------------------------------------
# Criterion 2: the --json shape — params, seed, per-region stability
# --------------------------------------------------------------------------


def test_json_shape_carries_algorithm_params_seed_and_stability(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    atlas = blob_atlas(tmp_path / "atlas", blobs=3, per_blob=8)

    _, payload = run_regions(
        "--atlas",
        str(atlas),
        "-k",
        "3",
        "--seed",
        "7",
        "--runs",
        "3",
        "--resamples",
        "4",
        capsys=capsys,
    )

    assert set(payload) >= {
        "atlas",
        "model",
        "items",
        "dimensions",
        "algorithm",
        "params",
        "seed",
        "verdict",
        "stability",
        "regions",
        "uncertain",
    }
    assert payload["algorithm"] == regions_mod.ALGORITHM
    assert payload["seed"] == 7
    assert payload["items"] == 24
    assert payload["dimensions"] == DIM

    params = payload["params"]
    assert params["algorithm"] == regions_mod.ALGORITHM
    assert params["clusters"] == 3
    assert params["runs"] == 3
    assert params["resamples"] == 4
    assert params["max_iterations"] == regions_mod.DEFAULT_MAX_ITERATIONS
    assert params["tolerance"] == regions_mod.DEFAULT_TOLERANCE
    assert params["representatives"] == regions_mod.DEFAULT_REPRESENTATIVES

    stability = payload["stability"]
    assert stability["params"] == params
    assert stability["seed"] == 7
    assert stability["verified"] is True
    assert stability["verdict"] == payload["verdict"]
    assert 0.0 <= stability["score"] <= 1.0


def test_every_region_carries_its_own_stability_with_params_and_seed(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    atlas = blob_atlas(tmp_path / "atlas", blobs=3, per_blob=8)

    _, payload = run_regions("--atlas", str(atlas), "-k", "3", "--seed", "5", capsys=capsys)

    for region in payload["regions"]:
        assert set(region) >= {
            "id",
            "size",
            "members",
            "representatives",
            "stability",
            "asserted",
            "finding",
        }
        stability = region["stability"]
        assert stability["measure"] == regions_mod.MEASURE_JACCARD
        assert stability["seed"] == 5
        assert stability["params"] == payload["params"]
        assert "verdict" in stability
        assert "score" in stability


def test_well_separated_blobs_are_stable_and_asserted(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    atlas = blob_atlas(tmp_path / "atlas", blobs=3, per_blob=8)

    _, payload = run_regions("--atlas", str(atlas), "-k", "3", capsys=capsys)

    assert payload["verdict"] == VERDICT_STABLE
    assert payload["uncertain"] == []
    for region in payload["regions"]:
        assert region["asserted"] is True
        assert region["finding"] is None
        assert region["stability"]["verdict"] == VERDICT_STABLE


def test_reader_verb_needs_no_gateway_and_never_mixes_streams(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    atlas = blob_atlas(tmp_path / "atlas", blobs=2, per_blob=5)

    with blocked_sockets():
        code = main(["regions", "--atlas", str(atlas), "-k", "2", "--json"])

    captured = capsys.readouterr()
    assert code == 0
    payload = json.loads(captured.out)  # stdout is pure JSON, nothing else
    assert payload["regions"]
    assert "{" not in captured.err


def test_text_mode_reports_regions_and_stability(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    atlas = blob_atlas(tmp_path / "atlas", blobs=2, per_blob=5)

    with blocked_sockets():
        code = main(["regions", "--atlas", str(atlas), "-k", "2"])

    out = capsys.readouterr().out
    assert code == 0
    assert "region 0" in out
    assert "seed:" in out
    assert "algorithm:" in out
    assert regions_mod.ALGORITHM in out


# --------------------------------------------------------------------------
# Criterion 3: one parameter setting -> a finding about the parameters
# --------------------------------------------------------------------------


def test_single_run_regions_are_unverified_findings_about_the_parameters(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    atlas = blob_atlas(tmp_path / "atlas", blobs=3, per_blob=8)

    _, payload = run_regions(
        "--atlas", str(atlas), "-k", "3", "--runs", "1", "--resamples", "0", capsys=capsys
    )

    assert payload["verdict"] == VERDICT_UNVERIFIED
    assert payload["stability"]["score"] is None
    assert payload["stability"]["verified"] is False

    # Still fully reported — suppression would hide real ambiguity (§3).
    assert len(payload["regions"]) == 3
    assert sum(region["size"] for region in payload["regions"]) == 24

    for region in payload["regions"]:
        assert region["asserted"] is False
        assert region["stability"]["verdict"] == VERDICT_UNVERIFIED
        assert region["stability"]["score"] is None
        finding = region["finding"]
        assert isinstance(finding, str) and finding
        assert "parameters" in finding
        assert "k=3" in finding
        assert "not" in finding

    assert len(payload["uncertain"]) >= len(payload["regions"])


def test_unstructured_space_still_reports_every_region_but_marks_it_uncertain(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    atlas = ambiguous_atlas(tmp_path / "atlas", items=24)

    _, payload = run_regions(
        "--atlas", str(atlas), "-k", "4", "--runs", "4", "--resamples", "8", capsys=capsys
    )

    assert len(payload["regions"]) == 4
    assert sum(region["size"] for region in payload["regions"]) == 24
    assert payload["verdict"] != VERDICT_STABLE

    uncertain_regions = [region for region in payload["regions"] if not region["asserted"]]
    assert uncertain_regions, "an unstructured space produced only asserted regions"
    for region in uncertain_regions:
        assert region["finding"]
        assert "parameters" in region["finding"]
    assert payload["uncertain"]


def test_text_mode_marks_an_unverified_region_rather_than_asserting_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    atlas = blob_atlas(tmp_path / "atlas", blobs=2, per_blob=5)

    with blocked_sockets():
        main(["regions", "--atlas", str(atlas), "-k", "2", "--runs", "1", "--resamples", "0"])

    out = capsys.readouterr().out
    assert VERDICT_UNVERIFIED in out
    assert "parameters" in out


# --------------------------------------------------------------------------
# Guard tests: the clustering is not trivially constant (the t11 caution)
# --------------------------------------------------------------------------


def test_clustering_is_genuinely_seed_sensitive() -> None:
    """Different seeds must produce different partitions on unstructured data.

    If they did not, every stability score in this file would be 1.0 by
    construction and the whole apparatus would be measuring nothing.
    """
    data = np.asarray(
        [deterministic_vector(f"noise-{index}", DIM) for index in range(24)], dtype=np.float64
    )
    partitions = [regions_mod.cluster_labels(data, seed, clusters=4) for seed in range(8)]
    agreements = [
        adjusted_rand_index(partitions[i], partitions[j])
        for i in range(len(partitions))
        for j in range(i + 1, len(partitions))
    ]
    assert min(agreements) < 1.0, "clustering ignored its seed"


def test_clustering_is_genuinely_composition_sensitive() -> None:
    """Resampling the item set must be able to change the partition.

    A clustering insensitive to composition would make bootstrap agreement
    vacuous — precisely the trap t11 fell into.
    """
    data = np.asarray(
        [deterministic_vector(f"noise-{index}", DIM) for index in range(24)], dtype=np.float64
    )
    reference = regions_mod.cluster_labels(data, 0, clusters=4)
    rng = np.random.default_rng(0)
    agreements = []
    for _ in range(8):
        indices = rng.integers(0, data.shape[0], size=data.shape[0])
        resampled = regions_mod.cluster_labels(data[indices], 0, clusters=4)
        agreements.append(adjusted_rand_index(reference[indices], resampled))
    assert min(agreements) < 1.0, "clustering ignored the composition of its input"


def test_structured_and_unstructured_spaces_do_not_score_alike(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The stability score must discriminate; a constant output would not."""
    structured = blob_atlas(tmp_path / "structured", blobs=3, per_blob=8)
    unstructured = ambiguous_atlas(tmp_path / "unstructured", items=24)

    _, structured_payload = run_regions("--atlas", str(structured), "-k", "3", capsys=capsys)
    _, unstructured_payload = run_regions("--atlas", str(unstructured), "-k", "3", capsys=capsys)

    assert structured_payload["stability"]["score"] > unstructured_payload["stability"]["score"]


def test_changing_k_changes_the_partition(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    atlas = blob_atlas(tmp_path / "atlas", blobs=4, per_blob=6)

    _, three = run_regions("--atlas", str(atlas), "-k", "3", capsys=capsys)
    _, five = run_regions("--atlas", str(atlas), "-k", "5", capsys=capsys)

    assert len(three["regions"]) == 3
    assert len(five["regions"]) == 5


# --------------------------------------------------------------------------
# Determinism over a fixed Atlas (design note §4, the guaranteed side)
# --------------------------------------------------------------------------


def test_same_atlas_same_params_same_seed_is_byte_identical(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    atlas = ambiguous_atlas(tmp_path / "atlas", items=24)
    argv = ["regions", "--atlas", str(atlas), "-k", "4", "--seed", "3", "--json"]

    with blocked_sockets():
        main(argv)
        first = capsys.readouterr().out
        main(argv)
        second = capsys.readouterr().out

    assert first == second


def test_a_different_seed_is_recorded_even_when_the_partition_matches(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    atlas = blob_atlas(tmp_path / "atlas", blobs=3, per_blob=8)

    _, first = run_regions("--atlas", str(atlas), "--seed", "1", "-k", "3", capsys=capsys)
    _, second = run_regions("--atlas", str(atlas), "--seed", "2", "-k", "3", capsys=capsys)

    assert first["seed"] == 1
    assert second["seed"] == 2
    assert first["stability"]["seed"] != second["stability"]["seed"]


# --------------------------------------------------------------------------
# Error paths — inherited from the shared reader entry point, plus validation
# --------------------------------------------------------------------------


def test_missing_atlas_exits_1_with_the_shared_hint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    code = main(["regions", "--atlas", str(tmp_path / "nope")])

    err = capsys.readouterr().err
    assert code == 1
    assert f"hint: {MISSING_ATLAS_HINT}" in err


def test_more_clusters_than_items_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    atlas = blob_atlas(tmp_path / "atlas", blobs=2, per_blob=2)

    code = main(["regions", "--atlas", str(atlas), "-k", "99"])

    err = capsys.readouterr().err
    assert code == 1
    assert "hint:" in err


@pytest.mark.parametrize(
    "flags",
    [
        ["-k", "1"],
        ["--runs", "0"],
        ["--resamples", "-1"],
        ["--representatives", "0"],
    ],
)
def test_invalid_parameters_exit_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], flags: list[str]
) -> None:
    atlas = blob_atlas(tmp_path / "atlas", blobs=2, per_blob=4)

    code = main(["regions", "--atlas", str(atlas), *flags])

    assert code == 1
    assert "hint:" in capsys.readouterr().err


def test_empty_atlas_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    target = tmp_path / "atlas"
    manifest = AtlasManifest(
        schema_version=SCHEMA_VERSION,
        model="fake-embedding-model",
        endpoint="http://localhost:8002/v1",
        params={},
        seed=0,
        limitations=[],
        uncertain=[],
    )
    write_atlas(manifest, [], np.zeros((0, DIM), dtype=np.float32), path=target)

    code = main(["regions", "--atlas", str(target)])

    assert code == 1
    assert "hint:" in capsys.readouterr().err


def test_regions_is_registered_on_the_top_level_parser(
    capsys: pytest.CaptureFixture[str],
) -> None:
    main([])
    assert "regions" in capsys.readouterr().out
