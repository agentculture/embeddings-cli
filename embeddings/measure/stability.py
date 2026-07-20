"""Seed-resampling stability measurement — the apparatus every structural claim passes through.

The policy this module enforces (design note ``docs/design-notes/semantic-measurement.md``
§3): *a region that only exists at one parameter setting is a finding about the
parameters, not about the model.* UMAP and t-SNE layouts, and k-means and
HDBSCAN partitions, reliably produce structures that look meaningful — clean
blobs, apparent corridors, crisp boundaries — from noise as readily as from
signal, and the specific structure shifts substantially with the seed alone,
parameters held fixed. Reporting a one-run artifact as a measured fact would
collapse the split between the Surveyor agent's hypotheses and the CLI's
measurements, which is the only reason to have a deterministic tool at all.

So this module makes three things mechanical:

1. **Every result carries ``params`` and ``seed``.** :class:`StabilityResult`
   has no constructor path that omits them, and :meth:`StabilityResult.to_dict`
   always emits both keys.
2. **Stability is agreement across N seed reruns plus bootstrap resampling of
   the items**, measured with the Adjusted Rand Index for clusterings
   (:func:`adjusted_rand_index`, implemented here on numpy alone).
3. **A single run is labeled ``stability unverified``, never scored.**
   :attr:`StabilityResult.score` is ``None`` — not ``0.0``, not any other
   sentinel a caller could arithmetic its way past — and
   :meth:`StabilityResult.score_or_raise` raises rather than inventing one.

Two things this module deliberately does *not* do. It does not choose a
clustering algorithm: callers hand in a ``cluster_fn(data, seed) -> labels``
callable, so the measurement apparatus is independent of what is being
measured. And it does not suppress low-stability structures — §3 is explicit
that suppression would hide real ambiguity in the space, which is itself a
finding. Unstable results are returned, scored, and marked
:attr:`~StabilityResult.is_uncertain` so the caller can route them into the
Atlas manifest's ``uncertain[]`` (see :meth:`StabilityResult.uncertain_note`).

Determinism is scoped to §4's guaranteed side of the line: given the same input
array and the same base ``seed``, the whole N-run measurement replays exactly,
because every per-run seed and every bootstrap resample is derived from the
base seed through :class:`numpy.random.SeedSequence` spawning rather than from
process-global RNG state. That guarantee covers reader-side work over a fixed
Atlas and cache-served vectors; it does not and cannot extend across live
re-embedding, which varies at float precision.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from embeddings.cli._errors import EXIT_USER_ERROR, CliError

# The agreement measure implemented here. Named in output so a reader never has
# to guess which statistic a number is; other claim kinds (neighborhood or rank
# agreement for directions/neighbors) will add their own constants.
MEASURE_ARI = "adjusted_rand_index"

# Verdict vocabulary. VERDICT_UNVERIFIED is a *label*, not a band: it says no
# stability evidence exists, which is categorically different from evidence of
# low stability. Keeping it in the same field as the bands is safe only because
# it is not a number and cannot be compared against a threshold.
VERDICT_UNVERIFIED = "stability unverified"
VERDICT_STABLE = "stable"
VERDICT_MODERATE = "moderate"
VERDICT_UNSTABLE = "unstable"

# Band edges on the agreement score. ARI is chance-corrected: ~0 is what random
# labelings score, so 0.5 is already weak agreement rather than a middling one.
STABLE_THRESHOLD = 0.75
MODERATE_THRESHOLD = 0.5

# Defaults chosen to be meaningful rather than token: reruns cost N clusterings
# but zero re-embedding (vectors come from the Atlas or the cache), per §3.
DEFAULT_RUNS = 5
DEFAULT_RESAMPLES = 20

# Digits kept when serializing. Enough to distinguish any agreement value a
# caller would act on, few enough that the JSON is stable and readable.
_ROUND_DIGITS = 6


class ClusterFn(Protocol):
    """``(data, seed) -> labels``: the clustering contract this module measures.

    ``data`` is an ``(n_items, n_features)`` array; the return value is any
    per-item label sequence of length ``n_items``. Labels need not be integers
    or contiguous — only the *partition* they induce is compared.
    """

    def __call__(self, data: np.ndarray, seed: int) -> Sequence[object]:  # pragma: no cover
        ...


def _user_error(message: str, remediation: str) -> CliError:
    return CliError(code=EXIT_USER_ERROR, message=message, remediation=remediation)


def _round(value: float) -> float:
    return float(round(float(value), _ROUND_DIGITS))


# --------------------------------------------------------------------------
# Adjusted Rand Index
# --------------------------------------------------------------------------


def _comb2(counts: np.ndarray) -> float:
    """Sum of ``C(n, 2)`` over ``counts`` — the pair count of a set of block sizes."""
    counts = counts.astype(np.float64, copy=False)
    return float(np.sum(counts * (counts - 1.0) / 2.0))


def _contingency(labels_a: np.ndarray, labels_b: np.ndarray) -> np.ndarray:
    """Cross-tabulation of two labelings, as a dense ``(n_a, n_b)`` count matrix."""
    _, index_a = np.unique(labels_a, return_inverse=True)
    _, index_b = np.unique(labels_b, return_inverse=True)
    n_a = int(index_a.max()) + 1
    n_b = int(index_b.max()) + 1
    flat = np.asarray(index_a, dtype=np.int64) * n_b + np.asarray(index_b, dtype=np.int64)
    return np.bincount(flat, minlength=n_a * n_b).reshape(n_a, n_b)


def _as_label_array(name: str, labels: Sequence[object]) -> np.ndarray:
    array = np.asarray(labels)
    if array.ndim != 1:
        raise _user_error(
            f"{name} must be a 1-D sequence of per-item labels, got shape {array.shape}",
            "pass one label per item, e.g. the array a clustering returned",
        )
    return array


def adjusted_rand_index(labels_a: Sequence[object], labels_b: Sequence[object]) -> float:
    """Adjusted Rand Index between two labelings of the same items.

    The Rand Index counts item *pairs* the two labelings agree about (together
    in both, or apart in both); the adjustment subtracts the agreement expected
    by chance given the two labelings' cluster-size profiles. That correction is
    what makes it usable here: an uncorrected index drifts toward 1.0 as cluster
    count grows, so it would reward exactly the over-partitioning that produces
    spurious "regions".

    Properties this relies on, all asserted in ``tests/test_stability.py``:

    * ``1.0`` for identical partitions;
    * **invariant to label permutation** and to label type — only the induced
      partition matters, so ``[0, 0, 1]``, ``[7, 7, 3]`` and ``["a", "a", "b"]``
      are the same labeling. This is the property that makes cross-seed
      comparison meaningful at all, since a clustering's label *names* are
      arbitrary per run;
    * ``~0.0`` for independent random labelings, and negative for
      worse-than-chance agreement.

    Degenerate cases (both labelings a single cluster, both all-singletons, or
    fewer than two items) leave the adjustment undefined — every labeling of
    such a set is the same partition. They return ``1.0``, matching the
    conventional reference behavior.

    Raises:
        CliError: if the two labelings differ in length, or are not 1-D.
    """
    array_a = _as_label_array("labels_a", labels_a)
    array_b = _as_label_array("labels_b", labels_b)
    if array_a.shape[0] != array_b.shape[0]:
        raise _user_error(
            "cannot compare labelings of different lengths "
            f"({array_a.shape[0]} vs {array_b.shape[0]})",
            "both labelings must cover the same item set, in the same order",
        )

    n_items = array_a.shape[0]
    if n_items < 2:
        return 1.0

    table = _contingency(array_a, array_b)
    index = _comb2(table.ravel())
    sum_a = _comb2(table.sum(axis=1))
    sum_b = _comb2(table.sum(axis=0))
    total_pairs = n_items * (n_items - 1) / 2.0
    expected = sum_a * sum_b / total_pairs
    maximum = (sum_a + sum_b) / 2.0

    if math.isclose(maximum, expected):
        # No pair-level information to adjust: both labelings are the trivial
        # partition (one cluster, or all singletons). They agree exactly.
        return 1.0
    return float((index - expected) / (maximum - expected))


# --------------------------------------------------------------------------
# Agreement distribution
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class AgreementDistribution:
    """The spread of agreement values behind one stability claim.

    A mean alone hides the case this module exists to catch: four reruns that
    agree perfectly plus one that disagrees completely averages to something
    reassuring. :attr:`minimum` and :attr:`spread` are therefore first-class,
    and :attr:`values` keeps the raw comparisons so a reader can audit them.
    """

    measure: str
    values: tuple[float, ...]
    mean: float
    minimum: float
    maximum: float
    spread: float
    stdev: float

    @property
    def comparisons(self) -> int:
        """How many pairwise agreement values this distribution summarizes."""
        return len(self.values)

    @classmethod
    def from_values(cls, measure: str, values: Sequence[float]) -> AgreementDistribution:
        """Summarize raw agreement values. Raises :class:`CliError` if there are none."""
        materialized = [float(value) for value in values]
        if not materialized:
            raise _user_error(
                "cannot summarize an empty set of agreement values",
                "run at least two seed reruns (runs >= 2) or one bootstrap resample",
            )
        array = np.asarray(materialized, dtype=np.float64)
        return cls(
            measure=measure,
            values=tuple(materialized),
            mean=float(array.mean()),
            minimum=float(array.min()),
            maximum=float(array.max()),
            spread=float(array.max() - array.min()),
            stdev=float(array.std()),
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "measure": self.measure,
            "comparisons": self.comparisons,
            "mean": _round(self.mean),
            "min": _round(self.minimum),
            "max": _round(self.maximum),
            "spread": _round(self.spread),
            "stdev": _round(self.stdev),
            "values": [_round(value) for value in self.values],
        }


# --------------------------------------------------------------------------
# The result type
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class StabilityResult:
    """A stability measurement, or the explicit absence of one.

    ``params`` and ``seed`` are required positional-or-keyword fields with no
    defaults, so no result can exist without them (§3: they are required keys on
    every structural verb's ``--json`` payload).

    The unverified state is represented by *both* agreement distributions being
    ``None``, which propagates to :attr:`score` being ``None`` and
    :attr:`verdict` being the literal :data:`VERDICT_UNVERIFIED`. It is never a
    numeric sentinel: ``0.0`` would be indistinguishable from the real,
    meaningful score of a clustering that is reproducible-at-chance-level, and a
    caller comparing ``score < 0.5`` would silently treat "we did not look" as
    "we looked and it was bad".
    """

    measure: str
    params: Mapping[str, object]
    seed: int
    runs: int
    resamples: int
    seed_agreement: AgreementDistribution | None = None
    bootstrap_agreement: AgreementDistribution | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def verified(self) -> bool:
        """True iff at least one form of rerun evidence exists."""
        return self.seed_agreement is not None or self.bootstrap_agreement is not None

    @property
    def score(self) -> float | None:
        """The stability score, or ``None`` when unverified — never a numeric sentinel.

        When both kinds of evidence exist the score is the **minimum** of their
        means, not a blend. A structure that survives seed reruns but dissolves
        under item resampling is unstable, and averaging would let the stronger
        piece of evidence conceal the weaker one.
        """
        means = [
            distribution.mean
            for distribution in (self.seed_agreement, self.bootstrap_agreement)
            if distribution is not None
        ]
        if not means:
            return None
        return min(means)

    @property
    def verdict(self) -> str:
        """:data:`VERDICT_UNVERIFIED`, or one of the stable/moderate/unstable bands."""
        score = self.score
        if score is None:
            return VERDICT_UNVERIFIED
        if score >= STABLE_THRESHOLD:
            return VERDICT_STABLE
        if score >= MODERATE_THRESHOLD:
            return VERDICT_MODERATE
        return VERDICT_UNSTABLE

    @property
    def is_uncertain(self) -> bool:
        """Whether the claim belongs in the Atlas manifest's ``uncertain[]``.

        True for unverified results as well as unstable ones — an unmeasured
        claim is not a confident one.
        """
        return self.verdict in (VERDICT_UNVERIFIED, VERDICT_UNSTABLE)

    def score_or_raise(self) -> float:
        """The score, or :class:`CliError` if there is none. The safe accessor.

        Use this wherever a numeric stability value is genuinely required, so
        that an unverified result fails loudly instead of being read as a low
        score by a caller that forgot to check :attr:`verified`.
        """
        score = self.score
        if score is None:
            raise _user_error(
                "stability unverified: this measurement has no rerun evidence, so it has "
                "no stability score",
                f"re-run with runs >= 2 (currently {self.runs}) or resamples >= 1 "
                f"(currently {self.resamples})",
            )
        return score

    def uncertain_note(self, subject: str) -> str:
        """One line about ``subject`` suitable verbatim for ``AtlasManifest.uncertain[]``."""
        if not self.verified:
            return (
                f"{subject}: {VERDICT_UNVERIFIED} — produced by a single run "
                f"(seed {self.seed}) with no seed reruns or bootstrap resampling"
            )
        score = self.score_or_raise()
        return (
            f"{subject}: {self.verdict} — {self.measure} {score:.3f} across {self.runs} seed "
            f"rerun(s) and {self.resamples} bootstrap resample(s) at seed {self.seed}"
        )

    def describe(self) -> str:
        """A one-line text-mode rendering; states the unverified label when it applies."""
        if not self.verified:
            return (
                f"{VERDICT_UNVERIFIED} (runs={self.runs}, resamples={self.resamples}) — "
                "no stability evidence; not reportable as a measured finding"
            )
        parts = []
        if self.seed_agreement is not None:
            parts.append(
                f"seed reruns {self.seed_agreement.mean:.3f} mean / "
                f"{self.seed_agreement.minimum:.3f} min over "
                f"{self.seed_agreement.comparisons} pair(s)"
            )
        if self.bootstrap_agreement is not None:
            parts.append(
                f"bootstrap {self.bootstrap_agreement.mean:.3f} mean over "
                f"{self.bootstrap_agreement.comparisons} resample(s)"
            )
        return f"{self.verdict} ({self.measure} {self.score_or_raise():.3f}): " + "; ".join(parts)

    def to_dict(self) -> dict[str, object]:
        """JSON-ready payload. ``params``, ``seed``, ``verdict`` and ``score`` always present."""
        score = self.score
        return {
            "measure": self.measure,
            "params": dict(self.params),
            "seed": self.seed,
            "runs": self.runs,
            "resamples": self.resamples,
            "verified": self.verified,
            "verdict": self.verdict,
            "score": None if score is None else _round(score),
            "seed_agreement": (
                None if self.seed_agreement is None else self.seed_agreement.to_dict()
            ),
            "bootstrap_agreement": (
                None if self.bootstrap_agreement is None else self.bootstrap_agreement.to_dict()
            ),
            "notes": list(self.notes),
        }


# --------------------------------------------------------------------------
# Seed derivation and the rerun harness
# --------------------------------------------------------------------------


def _spawn_seeds(seed: int, count: int) -> list[np.random.SeedSequence]:
    return list(np.random.SeedSequence(seed).spawn(count))


def _seed_int(sequence: np.random.SeedSequence) -> int:
    """A reproducible 32-bit integer seed to hand a clustering callable."""
    return int(sequence.generate_state(1, dtype=np.uint32)[0])


def _as_data(data: object) -> np.ndarray:
    array = np.asarray(data)
    if array.ndim != 2:
        raise _user_error(
            f"data must be a 2-D (n_items, n_features) array, got shape {array.shape}",
            "pass the Atlas vector matrix, one row per surveyed item",
        )
    if array.shape[0] == 0:
        raise _user_error(
            "cannot measure stability over an empty item set",
            "survey at least one item before measuring stability",
        )
    return array


def _labels_from(cluster_fn: ClusterFn, data: np.ndarray, seed: int) -> np.ndarray:
    labels = _as_label_array("cluster_fn result", cluster_fn(data, seed))
    if labels.shape[0] != data.shape[0]:
        raise _user_error(
            f"cluster_fn returned {labels.shape[0]} labels for {data.shape[0]} items",
            "cluster_fn(data, seed) must return exactly one label per row of data",
        )
    return labels


def seed_reruns(
    data: object,
    cluster_fn: ClusterFn,
    *,
    runs: int,
    seed: int,
) -> list[np.ndarray]:
    """Run ``cluster_fn`` ``runs`` times on the same data under derived seeds.

    Per-run seeds come from ``SeedSequence(seed).spawn(...)``, so they are
    distinct from each other, reproducible from the base ``seed`` alone, and
    independent of any process-global RNG state — the whole N-run measurement
    replays exactly.

    Raises:
        CliError: on ``runs < 1``, malformed data, or a wrong-length labeling.
    """
    if runs < 1:
        raise _user_error(
            f"runs must be at least 1, got {runs}",
            "use runs >= 2 to obtain seed-rerun agreement; runs=1 is stability unverified",
        )
    array = _as_data(data)
    return [_labels_from(cluster_fn, array, _seed_int(child)) for child in _spawn_seeds(seed, runs)]


def pairwise_agreement(
    labelings: Sequence[Sequence[object]],
    *,
    measure: str = MEASURE_ARI,
) -> AgreementDistribution:
    """Agreement over every unordered pair of ``labelings`` (``C(n, 2)`` comparisons).

    Raises:
        CliError: if fewer than two labelings are supplied — there is nothing to compare.
    """
    if len(labelings) < 2:
        raise _user_error(
            f"need at least 2 labelings to measure agreement, got {len(labelings)}",
            "increase runs to 2 or more",
        )
    values = [
        adjusted_rand_index(labelings[i], labelings[j])
        for i in range(len(labelings))
        for j in range(i + 1, len(labelings))
    ]
    return AgreementDistribution.from_values(measure, values)


def bootstrap_agreement(
    data: object,
    cluster_fn: ClusterFn,
    *,
    reference_labels: Sequence[object],
    resamples: int,
    seed: int,
    measure: str = MEASURE_ARI,
) -> AgreementDistribution:
    """Agreement between the reference labeling and clusterings of resampled item sets.

    Each resample draws ``n`` item indices **with replacement**, re-clusters that
    resampled matrix, and compares the result against ``reference_labels``
    restricted to the same drawn indices — so both labelings describe exactly the
    same multiset of items and the ARI is well defined. This is the check that
    catches a structure held up by a handful of individual items: a boundary that
    dissolves when the item set is perturbed was a fact about those items.

    Raises:
        CliError: on ``resamples < 1``, malformed data, or a reference labeling
            whose length does not match ``data``.
    """
    if resamples < 1:
        raise _user_error(
            f"resamples must be at least 1, got {resamples}",
            "pass resamples >= 1, or call measure_stability with resamples=0 to skip bootstrap",
        )
    array = _as_data(data)
    reference = _as_label_array("reference_labels", reference_labels)
    if reference.shape[0] != array.shape[0]:
        raise _user_error(
            f"reference_labels has {reference.shape[0]} labels for {array.shape[0]} items",
            "pass the labeling produced for this exact item set",
        )

    n_items = array.shape[0]
    values = []
    for child in _spawn_seeds(seed, resamples):
        index_sequence, cluster_sequence = child.spawn(2)
        indices = np.random.default_rng(index_sequence).integers(0, n_items, size=n_items)
        resampled = array[indices]
        labels = _labels_from(cluster_fn, resampled, _seed_int(cluster_sequence))
        values.append(adjusted_rand_index(reference[indices], labels))
    return AgreementDistribution.from_values(measure, values)


# --------------------------------------------------------------------------
# The entry point
# --------------------------------------------------------------------------


def measure_stability(
    data: object,
    cluster_fn: ClusterFn,
    *,
    params: Mapping[str, object],
    seed: int,
    runs: int = DEFAULT_RUNS,
    resamples: int = DEFAULT_RESAMPLES,
    measure: str = MEASURE_ARI,
) -> StabilityResult:
    """Measure how stable ``cluster_fn``'s structure is on ``data``, under ``seed``.

    Runs ``cluster_fn`` ``runs`` times under seeds derived from ``seed``, scores
    every pair of resulting labelings, then bootstrap-resamples the items
    ``resamples`` times and scores each against the first run's labeling. The
    two evidence streams are spawned from independent branches of the base seed,
    so changing ``resamples`` does not perturb the seed-rerun stream.

    With ``runs=1`` and ``resamples=0`` there is no rerun evidence at all; the
    result is returned anyway, labeled :data:`VERDICT_UNVERIFIED` with
    ``score=None``. That is the point — an unmeasured claim is reported as
    unmeasured, not quietly promoted to a finding and not silently dropped.

    Args:
        data: ``(n_items, n_features)`` vector matrix, from a fixed Atlas or the
            embed cache.
        cluster_fn: ``(data, seed) -> labels``. This module never chooses a
            clustering algorithm; the caller supplies one.
        params: the full parameter set in force, recorded verbatim on the result.
        seed: the base seed every derived seed and resample comes from.
        runs: number of seed reruns (>= 1).
        resamples: number of bootstrap resamples (>= 0; 0 skips the bootstrap).
        measure: the agreement measure's name, stamped on the output.

    Raises:
        CliError: on invalid arguments, empty/malformed data, or a ``cluster_fn``
            that returns the wrong number of labels.
    """
    if resamples < 0:
        raise _user_error(
            f"resamples must be zero or more, got {resamples}",
            "pass resamples=0 to skip bootstrap resampling",
        )
    array = _as_data(data)
    # Two independent branches: the run stream and the bootstrap stream. Spawning
    # them separately is what keeps run seeds unchanged when resamples changes.
    run_branch, bootstrap_branch = np.random.SeedSequence(seed).spawn(2)

    labelings = seed_reruns(array, cluster_fn, runs=runs, seed=_seed_int(run_branch))
    seed_distribution = pairwise_agreement(labelings, measure=measure) if runs >= 2 else None

    bootstrap_distribution = None
    if resamples > 0:
        bootstrap_distribution = bootstrap_agreement(
            array,
            cluster_fn,
            reference_labels=labelings[0],
            resamples=resamples,
            seed=_seed_int(bootstrap_branch),
            measure=measure,
        )

    notes: list[str] = []
    if seed_distribution is None and bootstrap_distribution is None:
        notes.append(
            "single run with no bootstrap resampling: reported as "
            f"'{VERDICT_UNVERIFIED}', not as a measured finding"
        )
    elif seed_distribution is None:
        notes.append("no seed-rerun evidence: stability rests on bootstrap resampling alone")
    elif bootstrap_distribution is None:
        notes.append("no bootstrap evidence: stability rests on seed reruns alone")

    return StabilityResult(
        measure=measure,
        params=dict(params),
        seed=seed,
        runs=runs,
        resamples=resamples,
        seed_agreement=seed_distribution,
        bootstrap_agreement=bootstrap_distribution,
        notes=tuple(notes),
    )


__all__ = [
    "DEFAULT_RESAMPLES",
    "DEFAULT_RUNS",
    "MEASURE_ARI",
    "MODERATE_THRESHOLD",
    "STABLE_THRESHOLD",
    "VERDICT_MODERATE",
    "VERDICT_STABLE",
    "VERDICT_UNSTABLE",
    "VERDICT_UNVERIFIED",
    "AgreementDistribution",
    "ClusterFn",
    "StabilityResult",
    "adjusted_rand_index",
    "bootstrap_agreement",
    "measure_stability",
    "pairwise_agreement",
    "seed_reruns",
]
