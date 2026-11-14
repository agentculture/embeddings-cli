"""Tests for :mod:`embeddings.measure.stability` — the stability policy of §3/§4.

These tests are the enforcement of a claim-discipline rule, not merely of a
function's arithmetic: a structural claim backed by one run must be *labeled*
``stability unverified`` and must be impossible to read as a measured score,
and a measurement over a fixed set of vectors must replay byte-identically.
"""

from __future__ import annotations

import json
import subprocess  # nosec B404 - fixed argv, no shell, used for a determinism check
import sys
from pathlib import Path

import numpy as np
import pytest

from embeddings.cli._errors import CliError
from embeddings.measure.stability import (
    MEASURE_ARI,
    VERDICT_MODERATE,
    VERDICT_STABLE,
    VERDICT_UNSTABLE,
    VERDICT_UNVERIFIED,
    AgreementDistribution,
    StabilityResult,
    adjusted_rand_index,
    bootstrap_agreement,
    measure_stability,
    seed_reruns,
)
from tests._fakes import deterministic_vector

# --------------------------------------------------------------------------
# Fixtures and clustering doubles
# --------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]


def separable_data(n_per_side: int = 12, dim: int = 4) -> np.ndarray:
    """Two well-separated blobs: rows 0..n-1 have x>0, rows n..2n-1 have x<0."""
    rows = []
    for index in range(n_per_side):
        vector = deterministic_vector(f"positive-{index}", dim)
        rows.append([abs(vector[0]) + 1.0, *vector[1:]])
    for index in range(n_per_side):
        vector = deterministic_vector(f"negative-{index}", dim)
        rows.append([-abs(vector[0]) - 1.0, *vector[1:]])
    return np.asarray(rows, dtype=np.float64)


def sign_clustering(data: np.ndarray, seed: int) -> np.ndarray:
    """A perfectly stable clustering: a pure function of the row, ignoring the seed."""
    return (np.asarray(data)[:, 0] >= 0.0).astype(np.int64)


def seed_driven_clustering(data: np.ndarray, seed: int) -> np.ndarray:
    """A maximally unstable clustering: labels depend only on the seed, not the data."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 3, size=len(data))


def sketch_kmeans(data: np.ndarray, seed: int, k: int = 3) -> np.ndarray:
    """A seed- *and* composition-sensitive clustering: random-init nearest-centroid.

    Deliberately not a degenerate double. ``sign_clustering`` is a pure function
    of each row, so its bootstrap agreement is 1.0 no matter which items are
    drawn — which would let a determinism test pass even if the resampling were
    unseeded. This one's output moves with both the derived seed (random initial
    centroids) and the drawn item multiset (a Lloyd update over it), so the
    byte-identity tests actually exercise the seed derivation *and* the
    resampling draw.
    """
    array = np.asarray(data, dtype=np.float64)
    rng = np.random.default_rng(seed)
    centroids = array[rng.choice(len(array), size=k, replace=False)]
    labels = np.argmin(((array[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2), axis=1)
    for cluster in range(k):
        members = array[labels == cluster]
        if len(members):
            centroids[cluster] = members.mean(axis=0)
    return np.argmin(((array[:, None, :] - centroids[None, :, :]) ** 2).sum(axis=2), axis=1)


class RecordingClustering:
    """Wraps a clustering callable and records every seed it was handed."""

    def __init__(self, inner=sign_clustering) -> None:
        self.inner = inner
        self.seeds: list[int] = []
        self.lengths: list[int] = []

    def __call__(self, data: np.ndarray, seed: int) -> np.ndarray:
        self.seeds.append(seed)
        self.lengths.append(len(data))
        return self.inner(data, seed)


# --------------------------------------------------------------------------
# 1. Adjusted Rand Index correctness
# --------------------------------------------------------------------------


def test_ari_identical_partitions_is_one():
    labels = [0, 0, 1, 1, 2, 2]
    assert adjusted_rand_index(labels, labels) == pytest.approx(1.0)


def test_ari_is_invariant_to_label_permutation():
    labels_a = [0, 0, 1, 1, 2, 2]
    permuted = [7, 7, "x", "x", 3, 3]
    # Same partition, totally different label names/types -> still 1.0.
    assert adjusted_rand_index(labels_a, permuted) == pytest.approx(1.0)

    other = [0, 1, 1, 1, 2, 2]
    baseline = adjusted_rand_index(labels_a, other)
    relabeled = ["b", "b", "c", "c", "a", "a"]
    assert adjusted_rand_index(relabeled, other) == pytest.approx(baseline)


def test_ari_known_value_matches_reference_implementation():
    # The reference value published for adjusted_rand_score([0,0,1,2],[0,0,1,1]).
    assert adjusted_rand_index([0, 0, 1, 2], [0, 0, 1, 1]) == pytest.approx(4.0 / 7.0)
    # A single cluster against all-singletons carries no information: exactly 0.
    assert adjusted_rand_index([0, 0, 0, 0], [0, 1, 2, 3]) == pytest.approx(0.0)


def test_ari_is_symmetric():
    a = [0, 0, 1, 1, 2, 3]
    b = [0, 1, 1, 1, 2, 2]
    assert adjusted_rand_index(a, b) == pytest.approx(adjusted_rand_index(b, a))


def test_ari_of_random_labelings_is_near_zero():
    rng = np.random.default_rng(20260720)
    scores = [
        adjusted_rand_index(rng.integers(0, 4, size=200), rng.integers(0, 4, size=200))
        for _ in range(200)
    ]
    assert abs(float(np.mean(scores))) < 0.02


def test_ari_can_be_negative_for_worse_than_chance_agreement():
    a = [0, 0, 1, 1]
    b = [0, 1, 0, 1]
    assert adjusted_rand_index(a, b) < 0.0


def test_ari_degenerate_partitions():
    assert adjusted_rand_index([0, 0, 0], [5, 5, 5]) == pytest.approx(1.0)
    assert adjusted_rand_index([0, 1, 2], [7, 8, 9]) == pytest.approx(1.0)
    assert adjusted_rand_index([0], [0]) == pytest.approx(1.0)


def _naive_ari(labels_a, labels_b) -> float:
    """An independent O(n^2) pair-counting ARI, sharing no code with the vectorized one.

    Cross-checking against this validates the contingency-table tabulation (the
    part that is easy to get subtly wrong) rather than restating the formula's
    result from the same machinery that produced it.
    """
    a = list(labels_a)
    b = list(labels_b)
    n = len(a)
    both = same_a = same_b = 0
    for i in range(n):
        for j in range(i + 1, n):
            in_a = a[i] == a[j]
            in_b = b[i] == b[j]
            same_a += in_a
            same_b += in_b
            both += in_a and in_b
    total = n * (n - 1) / 2.0
    expected = same_a * same_b / total
    maximum = (same_a + same_b) / 2.0
    if maximum == expected:
        return 1.0
    return (both - expected) / (maximum - expected)


def test_ari_matches_an_independent_pair_counting_implementation():
    rng = np.random.default_rng(7)
    for _ in range(50):
        n = int(rng.integers(4, 60))
        a = rng.integers(0, int(rng.integers(2, 8)), size=n)
        b = rng.integers(0, int(rng.integers(2, 8)), size=n)
        assert adjusted_rand_index(a, b) == pytest.approx(_naive_ari(a, b))


def test_ari_rejects_mismatched_lengths():
    with pytest.raises(CliError) as excinfo:
        adjusted_rand_index([0, 1, 2], [0, 1])
    assert excinfo.value.remediation


# --------------------------------------------------------------------------
# 2. The N-seed rerun harness
# --------------------------------------------------------------------------


def test_seed_reruns_derives_distinct_reproducible_seeds():
    data = separable_data()
    first = RecordingClustering()
    second = RecordingClustering()

    seed_reruns(data, first, runs=5, seed=1234)
    seed_reruns(data, second, runs=5, seed=1234)

    assert len(first.seeds) == 5
    assert len(set(first.seeds)) == 5, "each rerun must get its own derived seed"
    assert first.seeds == second.seeds, "the same base seed must replay the same run seeds"

    other = RecordingClustering()
    seed_reruns(data, other, runs=5, seed=1235)
    assert other.seeds != first.seeds, "a different base seed must produce different run seeds"


def test_seed_reruns_returns_one_labeling_per_run():
    data = separable_data()
    labelings = seed_reruns(data, sign_clustering, runs=3, seed=7)
    assert len(labelings) == 3
    assert all(len(labels) == len(data) for labels in labelings)


def test_seed_reruns_rejects_wrong_length_labels():
    data = separable_data()

    def truncating(chunk: np.ndarray, seed: int) -> np.ndarray:
        return sign_clustering(chunk, seed)[:-1]

    with pytest.raises(CliError):
        seed_reruns(data, truncating, runs=2, seed=1)


def test_stable_clustering_yields_perfect_seed_agreement():
    data = separable_data()
    distribution = measure_stability(
        data, sign_clustering, params={"k": 2}, seed=99, runs=4, resamples=0
    ).seed_agreement
    assert isinstance(distribution, AgreementDistribution)
    assert distribution.comparisons == 6  # C(4, 2)
    assert distribution.mean == pytest.approx(1.0)
    assert distribution.minimum == pytest.approx(1.0)
    assert distribution.spread == pytest.approx(0.0)


# --------------------------------------------------------------------------
# 3. Bootstrap resampling
# --------------------------------------------------------------------------


def test_bootstrap_agreement_resamples_items_reproducibly():
    data = separable_data()
    reference = sketch_kmeans(data, 0)

    first = RecordingClustering(sketch_kmeans)
    second = RecordingClustering(sketch_kmeans)
    a = bootstrap_agreement(data, first, reference_labels=reference, resamples=6, seed=42)
    b = bootstrap_agreement(data, second, reference_labels=reference, resamples=6, seed=42)

    assert a.comparisons == 6
    # Guards against a vacuous replay assertion: the values must actually vary
    # with the draw, so equality below is evidence the draw itself replayed.
    assert len(set(a.values)) > 1
    assert a.values == b.values
    assert first.seeds == second.seeds
    assert all(length == len(data) for length in first.lengths)

    different_seed = bootstrap_agreement(
        data, sketch_kmeans, reference_labels=reference, resamples=6, seed=43
    )
    assert different_seed.values != a.values


def test_bootstrap_agreement_is_perfect_for_a_row_pure_clustering():
    data = separable_data()
    reference = sign_clustering(data, 0)
    distribution = bootstrap_agreement(
        data, sign_clustering, reference_labels=reference, resamples=8, seed=3
    )
    assert distribution.mean == pytest.approx(1.0)


def test_bootstrap_agreement_is_low_for_a_seed_driven_clustering():
    data = separable_data()
    reference = seed_driven_clustering(data, 0)
    distribution = bootstrap_agreement(
        data, seed_driven_clustering, reference_labels=reference, resamples=12, seed=5
    )
    assert distribution.mean < 0.2


# --------------------------------------------------------------------------
# 4. The result type: params, seed, and the unverified state
# --------------------------------------------------------------------------


def test_result_always_carries_params_and_seed():
    data = separable_data()
    result = measure_stability(
        data, sign_clustering, params={"algorithm": "sign", "k": 2}, seed=11, runs=3, resamples=4
    )
    payload = result.to_dict()
    assert payload["params"] == {"algorithm": "sign", "k": 2}
    assert payload["seed"] == 11
    assert payload["measure"] == MEASURE_ARI


def test_single_run_is_labeled_stability_unverified():
    data = separable_data()
    result = measure_stability(data, sign_clustering, params={}, seed=1, runs=1, resamples=0)

    assert result.verified is False
    assert result.verdict == VERDICT_UNVERIFIED == "stability unverified"
    assert result.seed_agreement is None
    assert result.bootstrap_agreement is None
    assert result.is_uncertain is True


def test_unverified_score_is_none_not_a_number():
    data = separable_data()
    result = measure_stability(data, sign_clustering, params={}, seed=1, runs=1, resamples=0)

    assert result.score is None
    assert not isinstance(result.score, float)
    payload = result.to_dict()
    assert payload["score"] is None
    assert payload["verified"] is False
    assert payload["verdict"] == "stability unverified"
    # The serialized form must not admit a numeric reading of the unverified state.
    assert json.dumps(payload["score"]) == "null"


def test_unverified_score_cannot_be_consumed_as_a_measurement():
    data = separable_data()
    result = measure_stability(data, sign_clustering, params={}, seed=1, runs=1, resamples=0)
    with pytest.raises(CliError) as excinfo:
        result.score_or_raise()
    assert "unverified" in str(excinfo.value).lower()
    assert excinfo.value.remediation


def test_bootstrap_alone_verifies_a_single_run():
    data = separable_data()
    result = measure_stability(data, sign_clustering, params={}, seed=1, runs=1, resamples=6)
    assert result.verified is True
    assert result.verdict != VERDICT_UNVERIFIED
    assert isinstance(result.score, float)


def test_verified_stable_result_scores_and_is_not_uncertain():
    data = separable_data()
    result = measure_stability(data, sign_clustering, params={"k": 2}, seed=2, runs=4, resamples=6)
    assert result.verified is True
    assert result.verdict == VERDICT_STABLE
    assert result.score == pytest.approx(1.0)
    assert result.is_uncertain is False
    assert result.score_or_raise() == pytest.approx(1.0)


def test_unstable_result_is_reported_not_suppressed():
    data = separable_data()
    result = measure_stability(
        data, seed_driven_clustering, params={"k": 3}, seed=8, runs=5, resamples=6
    )
    # Reported, with its instability attached -- §3 forbids suppressing it.
    assert isinstance(result, StabilityResult)
    assert result.verified is True
    assert result.verdict == VERDICT_UNSTABLE
    assert result.score is not None
    assert result.score < 0.5
    assert result.is_uncertain is True

    note = result.uncertain_note("region 3")
    assert "region 3" in note
    assert VERDICT_UNSTABLE in note
    # Must be usable verbatim as an AtlasManifest.uncertain[] entry.
    assert isinstance(note, str)


def test_verdict_bands_are_ordered():
    assert VERDICT_STABLE != VERDICT_MODERATE != VERDICT_UNSTABLE
    for verdict, score in ((VERDICT_STABLE, 0.9), (VERDICT_MODERATE, 0.6), (VERDICT_UNSTABLE, 0.1)):
        distribution = AgreementDistribution.from_values(MEASURE_ARI, [score, score])
        result = StabilityResult(
            measure=MEASURE_ARI,
            params={},
            seed=0,
            runs=2,
            resamples=0,
            seed_agreement=distribution,
            bootstrap_agreement=None,
        )
        assert result.verdict == verdict


def test_score_is_the_conservative_minimum_of_the_evidence():
    seed_distribution = AgreementDistribution.from_values(MEASURE_ARI, [0.9, 0.9])
    bootstrap_distribution = AgreementDistribution.from_values(MEASURE_ARI, [0.4, 0.4])
    result = StabilityResult(
        measure=MEASURE_ARI,
        params={},
        seed=0,
        runs=2,
        resamples=2,
        seed_agreement=seed_distribution,
        bootstrap_agreement=bootstrap_distribution,
    )
    assert result.score == pytest.approx(0.4)
    assert result.verdict == VERDICT_UNSTABLE


def test_describe_states_the_unverified_label_in_text_mode():
    data = separable_data()
    unverified = measure_stability(data, sign_clustering, params={}, seed=1, runs=1, resamples=0)
    assert "stability unverified" in unverified.describe()

    verified = measure_stability(data, sign_clustering, params={}, seed=1, runs=3, resamples=3)
    assert "stability unverified" not in verified.describe()
    assert MEASURE_ARI in verified.describe() or "ARI" in verified.describe()


def test_invalid_arguments_are_rejected():
    data = separable_data()
    with pytest.raises(CliError):
        measure_stability(data, sign_clustering, params={}, seed=1, runs=0)
    with pytest.raises(CliError):
        measure_stability(data, sign_clustering, params={}, seed=1, runs=2, resamples=-1)
    empty_data = np.empty((0, 3))
    with pytest.raises(CliError):
        measure_stability(empty_data, sign_clustering, params={}, seed=1, runs=2)


# --------------------------------------------------------------------------
# 5. Determinism -- §4's guaranteed side of the line (fixed vectors on disk,
#    no live re-embedding)
# --------------------------------------------------------------------------

# The subprocess imports `reader_output` from this very module rather than
# re-declaring it, so the two processes provably run identical code.
READER_SCRIPT = """
import sys

import numpy as np

from tests.test_stability import reader_output

sys.stdout.write(reader_output(np.load(sys.argv[1])))
"""


def reader_output(vectors: np.ndarray) -> str:
    """Stand-in for a reader verb over a FIXED Atlas: vectors in, JSON bytes out.

    Uses :func:`sketch_kmeans`, whose answer genuinely moves with the derived
    seeds and the bootstrap draw, so byte-identity here is a claim about the
    seed derivation replaying -- not an artifact of a clustering that would
    return the same thing under any seed.
    """
    result = measure_stability(
        vectors, sketch_kmeans, params={"k": 3}, seed=4242, runs=5, resamples=10
    )
    return json.dumps(result.to_dict(), sort_keys=True)


def test_reader_output_is_not_trivially_constant():
    """Guard: the byte-identity tests below are vacuous if the payload never varies."""
    data = separable_data()
    result = measure_stability(
        data, sketch_kmeans, params={"k": 3}, seed=4242, runs=5, resamples=10
    )
    assert result.verified is True
    assert len(set(result.bootstrap_agreement.values)) > 1
    assert reader_output(data) != reader_output(separable_data(n_per_side=13))


def test_reader_over_a_fixed_atlas_replays_byte_identically(tmp_path):
    path = tmp_path / "vectors.npy"
    np.save(path, separable_data())

    first = reader_output(np.load(path))
    second = reader_output(np.load(path))
    assert first.encode("utf-8") == second.encode("utf-8")


def test_reader_over_a_fixed_atlas_is_byte_identical_across_processes(tmp_path):
    path = tmp_path / "vectors.npy"
    np.save(path, separable_data())

    expected = reader_output(np.load(path))
    completed = subprocess.run(  # nosec B603 - fixed argv, no shell, no user input
        [sys.executable, "-c", READER_SCRIPT, str(path)],
        cwd=str(REPO_ROOT),
        capture_output=True,
        check=True,
    )
    assert completed.stdout == expected.encode("utf-8")


def test_a_different_seed_changes_the_measurement_inputs():
    data = separable_data()
    recorder_a = RecordingClustering(seed_driven_clustering)
    recorder_b = RecordingClustering(seed_driven_clustering)
    measure_stability(data, recorder_a, params={}, seed=1, runs=4, resamples=4)
    measure_stability(data, recorder_b, params={}, seed=2, runs=4, resamples=4)
    assert recorder_a.seeds != recorder_b.seeds


def test_resample_count_does_not_perturb_the_seed_rerun_stream():
    data = separable_data()
    few = RecordingClustering()
    many = RecordingClustering()
    measure_stability(data, few, params={}, seed=77, runs=4, resamples=2)
    measure_stability(data, many, params={}, seed=77, runs=4, resamples=9)
    assert few.seeds[:4] == many.seeds[:4]
