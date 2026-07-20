"""``embeddings gaps`` — under-covered areas, labeled by what the evidence can actually support.

The one distinction this verb exists to hold
--------------------------------------------
A sparse area of a surveyed corpus is a gap in **the corpus**. It is *not*, on
its own, a gap in what the model can represent. The model may represent that
area perfectly well; the survey simply never sent it anything from there. So
this verb emits two kinds of finding and never merges them:

* **corpus coverage** (:data:`EVIDENCE_CORPUS`) — derived from sparsity in the
  surveyed atlas. Says only that the corpus has few or no items in some part of
  the space. Says nothing about model capability.
* **model representation** (:data:`EVIDENCE_MODEL`) — derived **only** from
  actively probing a sparse area with inputs embedded at probe time, and
  measuring where they land relative to that area and to each other.

They are **separate top-level JSON keys** (``corpus_coverage_findings`` and
``model_representation_findings``), not one list with a ``kind`` field. That is
deliberate and structural: a shared list invites an aggregating agent to
concatenate, count, or rank the two together, and the moment it does the
distinction is gone with no error raised anywhere. A key an aggregator has to
name explicitly cannot be flattened by accident. For the same reason ``counts``
reports the two separately and publishes no combined total.

Why absence cannot be inferred from silence
--------------------------------------------
An empty region has at least four sufficient explanations — corpus sampling and
recency; ingestion failures (filtering, truncation, parse errors); the
measurement parameters that called the area sparse; and actual model behaviour
— and sparsity alone cannot discriminate among them. Only the last is a model
finding, and only a probe isolates it: embed inputs that *should* land in the
area and observe what the model does with them. That is why probing is the sole
route to a :data:`EVIDENCE_MODEL` finding, and why a run without a probe step
returns an empty ``model_representation_findings`` list rather than a hedged
one. See ``docs/design-notes/semantic-measurement.md`` §1, which this module
implements.

How sparse areas are located
-----------------------------
Two complementary corpus-coverage measurements, both over the k-means partition
this CLI already uses for ``regions``:

1. **Sparse regions.** Every active item's distance to its nearest other active
   item is computed (a k-NN density estimate at k=1); a region's *isolation* is
   the mean of its members' nearest-neighbour distances. Regions at or above the
   ``1 - --sparsity-quantile`` quantile of that distribution are reported as
   sparse. Isolation is used rather than "size over radius" because it degrades
   correctly at the extremes: a one-item region reads as maximally isolated,
   whereas any size/radius ratio divides by a radius of zero and reports the
   most isolated thing in the space as the densest.
2. **Inter-region voids.** The midpoint between each pair of region centroids is
   a point the corpus has no item at *by construction of the pair*; what matters
   is how far the nearest real item is from it. Midpoints whose nearest-item
   distance exceeds ``--void-factor`` times the corpus's own median
   nearest-neighbour distance are reported as voids. The reference is the
   corpus's own scale rather than an absolute cosine number, because "far" is
   only meaningful relative to how tightly this particular model packs this
   particular corpus.

Both thresholds are parameters, stamped in ``params`` and repeated inside each
finding's ``measurement``. Neither is a discovered truth.

The distinguishing measurement behind a model finding
------------------------------------------------------
"We probed" is not a measurement. Each probe is assigned to its nearest region
centroid, and a sparse region's model finding rests on three numbers a reader
can check:

* **where the probes landed** — how many of them were nearest *this* region;
* **probe spread** — mean pairwise cosine distance among the probes that landed
  here, compared against the **corpus-wide baseline spread** (the mean pairwise
  distance over the whole active corpus, i.e. how far apart this model puts
  arbitrary text). A probe spread at or above ``--coherence-factor`` times that
  baseline is indistinguishable from unrelated text;
* **distance to the nearest real item**, which says whether the probes reached
  the sparse area or merely pointed at it.

Three outcomes follow, and only two of them are findings:

* :data:`PROBE_VERDICT_COHERENT` — probes landed here and cohere. This is
  evidence the model *does* represent the area, so the sparsity was about the
  corpus. It is a model-representation finding pointing away from a model gap.
* :data:`PROBE_VERDICT_SCATTERED` — probes landed here and scattered. Evidence
  about the representation — evidence, never proof, since it stays conditioned
  on the probes being fair instances of the concept.
* :data:`PROBE_VERDICT_ELSEWHERE` / :data:`PROBE_VERDICT_TOO_FEW` — the probes
  missed the region, or too few landed to have a spread at all. **No model
  finding is emitted**; the attempt is recorded in ``inconclusive_probes`` so a
  reader can see the probe ran and told us nothing, which is different from the
  probe not having run.

Gateway behaviour
------------------
``gaps`` is the one reader verb that may touch the backend, because probing
requires embedding new inputs at probe time. Egress is never implicit: probing
is **off by default** and ``--probes``/``--holdout`` without ``--probe`` is a
user error rather than a silent upgrade to a network call. When probing *is*
requested and the gateway is unreachable, the verb degrades — corpus-coverage
findings are still emitted, ``probe_step`` is marked unavailable with the reason,
zero model findings are produced, and the exit code stays 0. The corpus half is
still valid; failing the whole verb would throw it away, and quietly dropping
the probe step would erase the very distinction this module exists to keep.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from embeddings._numeric import safe_norms
from embeddings.atlas.io import Atlas
from embeddings.backends import (
    ENDPOINT_FLAG,
    ENDPOINT_HELP,
    EmbeddingCache,
    EmbeddingClient,
    get_or_embed,
)
from embeddings.backends.client import MODEL_FLAG, MODEL_HELP
from embeddings.cli._commands._atlas_arg import add_atlas_argument, load_atlas
from embeddings.cli._commands.regions import (
    DEFAULT_CLUSTERS,
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_TOLERANCE,
    kmeans,
)
from embeddings.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from embeddings.cli._output import emit_diagnostic, emit_result
from embeddings.cli._provenance import (
    REPRODUCED_BY_KEY,
    reproduce_line,
    reproducing_command,
)
from embeddings.measure import StabilityResult, measure_stability
from embeddings.measure.stability import DEFAULT_RESAMPLES, DEFAULT_RUNS

#: The two evidence classes, spelled exactly as the design note names them.
#: Every finding carries one verbatim, so a reader who has flattened the two
#: lists (against this module's whole design) can still tell them apart.
EVIDENCE_CORPUS = "corpus coverage"
EVIDENCE_MODEL = "model representation"

#: Corpus-coverage finding types.
FINDING_SPARSE_REGION = "sparse_region"
FINDING_VOID = "inter_region_void"
#: The single model-representation finding type.
FINDING_PROBE_RESULT = "probe_result"

#: Probe outcomes. Only the first two ever become a model-representation
#: finding; the rest are recorded as inconclusive.
PROBE_VERDICT_COHERENT = "probes landed coherently"
PROBE_VERDICT_SCATTERED = "probes scattered"
PROBE_VERDICT_ELSEWHERE = "probes landed elsewhere"
PROBE_VERDICT_TOO_FEW = "too few probes landed to measure spread"

#: Probe provenance sources. Recorded per probe so a reader can audit exactly
#: what was sent to the model behind any model-representation claim.
PROVENANCE_FILE = "probes-file"
PROVENANCE_HOLDOUT = "holdout"

#: The one distance metric this verb computes or reports, matching ``neighbors``.
METRIC = "cosine"

#: Fraction of regions (by isolation, most isolated first) reported as sparse.
DEFAULT_SPARSITY_QUANTILE = 0.25
#: Multiple of the corpus's median nearest-neighbour distance above which an
#: inter-centroid midpoint counts as a void.
DEFAULT_VOID_FACTOR = 2.0
#: Fraction of the corpus-wide baseline spread below which probes count as
#: cohering rather than scattering.
DEFAULT_COHERENCE_FACTOR = 0.75
#: Cap on reported voids; region pairs grow quadratically in ``k``.
DEFAULT_MAX_VOIDS = 5
#: Base seed. Every derived seed in the run comes from this one value.
DEFAULT_SEED = 0

#: The prose every reader gets, in both output modes, whether or not they parse
#: a single key name. This is the sentence the verb exists to make unavoidable.
DISCLAIMER = (
    "Corpus sparsity alone says nothing about model capability. A sparse or empty area of "
    "this atlas means only that the surveyed corpus has few or no items there. That has at "
    "least four sufficient explanations — corpus sampling and recency, ingestion failures "
    "such as filtering or truncation, the measurement parameters that called the area "
    "sparse, and actual model behaviour — and sparsity cannot discriminate among them. Only "
    "an active probe, embedding new inputs at probe time and measuring where they land, can "
    "isolate the last one. The two finding lists are never merged, summed, or ranked "
    "together."
)


# --------------------------------------------------------------------------
# Egress seam
# --------------------------------------------------------------------------


def build_client(base_url: str | None, model: str | None) -> EmbeddingClient:
    """Construct the embedding client for the probe step — this module's only egress point.

    Reached only when ``--probe`` is passed; the corpus-coverage half never
    calls it. Kept as a single seam, matching ``survey.build_client`` and
    ``neighbors.build_client``, so tests substitute a fake-transport-backed
    client without a test-only flag leaking into the CLI surface.
    """
    kwargs: dict[str, str] = {}
    if base_url:
        kwargs["base_url"] = base_url
    if model:
        kwargs["model"] = model
    return EmbeddingClient(**kwargs)


# --------------------------------------------------------------------------
# Geometry
# --------------------------------------------------------------------------


def _cosine_distance_matrix(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """``(len(left), len(right))`` cosine distances.

    The matrix form of ``neighbors._cosine_distances``, which is one-to-many and
    raises on a zero-magnitude query. Sparsity detection needs all-pairs
    distances and must not raise on a degenerate row, so a zero-magnitude vector
    is treated as maximally distant (``2.0``) on both sides — the same
    convention ``neighbors`` applies to zero-magnitude atlas rows.
    """
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    # Guard by magnitude before dividing, not by filtering non-finite results
    # afterwards: a norm of 1e-300 divides to a perfectly finite number that
    # `isfinite` happily keeps, even though its direction is representation
    # noise. Degenerate rows/columns are forced to maximal distance instead.
    left_norms, left_degenerate = safe_norms(left)
    right_norms, right_degenerate = safe_norms(right)
    similarity = (left @ right.T) / np.outer(left_norms, right_norms)
    degenerate = left_degenerate[:, None] | right_degenerate[None, :]
    similarity = np.where(degenerate, -1.0, similarity)
    return 1.0 - np.clip(similarity, -1.0, 1.0)


def _nearest_neighbor_distances(data: np.ndarray) -> np.ndarray:
    """Each row's cosine distance to its nearest *other* row: a k=1 density estimate.

    A single-row matrix has no neighbour to measure against; it reports ``2.0``
    (maximally isolated), which is the honest reading of "nothing else is near
    it" rather than a zero that would read as maximally dense.
    """
    if data.shape[0] < 2:
        return np.full(data.shape[0], 2.0, dtype=np.float64)
    distances = _cosine_distance_matrix(data, data)
    np.fill_diagonal(distances, np.inf)
    return distances.min(axis=1)


def _mean_pairwise_distance(data: np.ndarray) -> float | None:
    """Mean pairwise cosine distance, or ``None`` when fewer than two rows exist.

    ``None`` rather than ``0.0``: a spread that could not be computed and a
    spread that was computed and came out tight must not be the same value, for
    the same reason :attr:`StabilityResult.score` refuses a numeric sentinel.
    """
    if data.shape[0] < 2:
        return None
    distances = _cosine_distance_matrix(data, data)
    upper = distances[np.triu_indices(data.shape[0], k=1)]
    return float(upper.mean())


def _round(value: float | None) -> float | None:
    return None if value is None else round(float(value), 6)


# --------------------------------------------------------------------------
# The partition and its sparse set
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Partition:
    """One k-means partition plus everything sparsity detection reads off it."""

    labels: np.ndarray
    centroids: np.ndarray
    nearest_neighbor: np.ndarray
    isolation: dict[int, float]
    threshold: float
    sparse_clusters: tuple[int, ...]

    def sparse_mask(self) -> np.ndarray:
        return np.isin(self.labels, np.asarray(self.sparse_clusters, dtype=np.int64))


def _partition(
    data: np.ndarray,
    seed: int,
    *,
    clusters: int,
    sparsity_quantile: float,
    max_iterations: int,
    tolerance: float,
) -> _Partition:
    """Cluster ``data`` and identify which clusters are sparse. The single source of truth.

    Both the reported partition and every stability rerun go through this
    function, so "which regions are sparse" is computed exactly one way. A
    rerun that used a different rule would be measuring the rule, not the space.
    """
    labels, centroids = kmeans(
        data,
        seed,
        clusters=min(clusters, data.shape[0]),
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    nearest_neighbor = _nearest_neighbor_distances(data)
    isolation = {
        int(cluster): float(nearest_neighbor[labels == cluster].mean())
        for cluster in np.unique(labels)
    }
    # The most *isolated* regions are the sparse ones, so the cut is at the
    # upper tail: --sparsity-quantile 0.25 means "the top quarter by isolation".
    threshold = float(np.quantile(list(isolation.values()), 1.0 - sparsity_quantile))
    sparse = tuple(cluster for cluster, value in sorted(isolation.items()) if value >= threshold)
    return _Partition(
        labels=labels,
        centroids=centroids,
        nearest_neighbor=nearest_neighbor,
        isolation=isolation,
        threshold=threshold,
        sparse_clusters=sparse,
    )


# --------------------------------------------------------------------------
# Probes
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class _Probe:
    """One probe input and where it came from.

    ``provenance`` travels with the probe all the way into the finding, because
    a model-representation claim is only as good as the probes behind it and a
    reader must be able to judge them.
    """

    text: str
    provenance: dict[str, object] = field(default_factory=dict)


def _read_probes_file(raw_path: str) -> list[_Probe]:
    """One probe per line; blank lines and ``#`` comments skipped, line numbers kept."""
    path = Path(raw_path)
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as err:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"cannot read probes file {path}: {err}",
            remediation=(
                "pass --probes <path> pointing at a readable UTF-8 text file with one probe "
                "text per line (blank lines and # comments are ignored)"
            ),
        ) from err

    probes = [
        _Probe(
            text=stripped,
            provenance={"source": PROVENANCE_FILE, "path": str(path), "line": line_number},
        )
        for line_number, line in enumerate(content.splitlines(), start=1)
        if (stripped := line.strip()) and not stripped.startswith("#")
    ]
    if not probes:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"no probe texts in {path} — every line is blank or a # comment",
            remediation="put one probe text per line in the file, or use --holdout <n> instead",
        )
    return probes


def _holdout_positions(atlas: Atlas, holdout: int, seed: int) -> np.ndarray:
    """Choose ``holdout`` record positions to hold out, reproducibly from ``seed`` alone."""
    total = len(atlas.records)
    if holdout >= total:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"--holdout {holdout} leaves nothing to analyse in a {total}-item atlas",
            remediation=f"pass --holdout {max(total - 2, 1)} or fewer, or survey a larger corpus",
        )
    chosen = np.random.default_rng(seed).choice(total, size=holdout, replace=False)
    return np.sort(chosen)


# --------------------------------------------------------------------------
# Findings
# --------------------------------------------------------------------------


def _sparse_region_finding(
    *,
    region_id: int,
    members: list[str],
    isolation: float,
    partition: _Partition,
    region_spread: float | None,
    baseline_spread: float | None,
    asserted: bool,
    stability_verdict: str,
) -> dict[str, object]:
    return {
        "evidence_class": EVIDENCE_CORPUS,
        "type": FINDING_SPARSE_REGION,
        "region_id": region_id,
        "size": len(members),
        "members": members,
        "measurement": {
            "metric": METRIC,
            "isolation": _round(isolation),
            "sparsity_threshold": _round(partition.threshold),
            "region_spread": _round(region_spread),
            "corpus_baseline_spread": _round(baseline_spread),
        },
        "shows": (
            f"Region {region_id} holds {len(members)} surveyed item(s) whose mean distance to "
            f"their nearest neighbour is {isolation:.6f}, at or above the "
            f"{partition.threshold:.6f} cut for this run. The corpus covers this part of the "
            "space thinly."
        ),
        "does_not_show": (
            "This does not show that the model cannot represent this area. Thin corpus "
            "coverage is equally consistent with corpus sampling, an ingestion failure, or "
            "the parameters used here. Run with --probe and probe inputs that belong in this "
            "region to test the model itself."
        ),
        "asserted": asserted,
        "stability_verdict": stability_verdict,
    }


def _void_finding(
    *,
    pair: tuple[int, int],
    nearest_id: str,
    nearest_distance: float,
    reference_distance: float,
    reference_basis: str,
    void_threshold: float,
    asserted: bool,
    stability_verdict: str,
) -> dict[str, object]:
    return {
        "evidence_class": EVIDENCE_CORPUS,
        "type": FINDING_VOID,
        "between_regions": list(pair),
        "measurement": {
            "metric": METRIC,
            "nearest_item_id": nearest_id,
            "nearest_item_distance": _round(nearest_distance),
            "reference_distance": _round(reference_distance),
            "reference_basis": reference_basis,
            "void_threshold": _round(void_threshold),
        },
        "shows": (
            f"The midpoint between regions {pair[0]} and {pair[1]} has no surveyed item within "
            f"{nearest_distance:.6f} (nearest: {nearest_id!r}), against a {reference_basis} of "
            f"{reference_distance:.6f} and a threshold of {void_threshold:.6f}. The corpus "
            "covers the space between these two regions thinly or not at all."
        ),
        "does_not_show": (
            "This does not show the model has nothing to say about the area between these "
            "regions — nothing was ever sent to it from there. It also does not show the "
            "midpoint corresponds to any expressible concept; it is an arithmetic point "
            "between two centroids, not a decoded meaning."
        ),
        "asserted": asserted,
        "stability_verdict": stability_verdict,
    }


def _probe_payload(probe: _Probe, detail: dict[str, object]) -> dict[str, object]:
    return {"text": probe.text, "provenance": dict(probe.provenance), **detail}


def _model_finding(
    *,
    region_id: int,
    members: list[str],
    verdict: str,
    measurement: dict[str, object],
    probes: list[dict[str, object]],
    coherence_factor: float,
    clusters: int,
    model: str | None,
    endpoint: str | None,
) -> dict[str, object]:
    landed = measurement["probes_landed"]
    spread = measurement["probe_spread"]
    threshold = measurement["coherence_threshold"]
    baseline = measurement["corpus_baseline_spread"]

    if verdict == PROBE_VERDICT_COHERENT:
        shows = (
            f"The {landed} probe(s) that landed nearest region {region_id} cohere: their mean "
            f"pairwise distance is {spread}, below the {threshold} coherence threshold "
            f"({coherence_factor} x the corpus-wide baseline spread of {baseline}). The model "
            "places these inputs together in this area, which is evidence that the sparsity "
            "reported here is a fact about the corpus rather than about the model."
        )
    else:
        shows = (
            f"The {landed} probe(s) that landed nearest region {region_id} scatter: their mean "
            f"pairwise distance is {spread}, at or above the {threshold} coherence threshold "
            f"({coherence_factor} x the corpus-wide baseline spread of {baseline}) — that is, "
            "no tighter than this model places arbitrary text. The model does not give these "
            "inputs a shared neighbourhood here. That is evidence about the representation in "
            "this area."
        )

    return {
        "evidence_class": EVIDENCE_MODEL,
        "type": FINDING_PROBE_RESULT,
        "region_id": region_id,
        "region_members": members,
        "verdict": verdict,
        "measurement": measurement,
        "probes": probes,
        "shows": shows,
        "does_not_show": (
            f"This is evidence, not proof. It is conditioned on these {landed} probe(s) being "
            f"fair instances of what belongs in region {region_id}, on the k={clusters} "
            "partition that defined the region, and on the stated coherence threshold. It does "
            "not show the model is incapable of representing this area, it does not generalise "
            "to any other region, and it is not a measurement of the corpus."
        ),
        "model": model,
        "endpoint": endpoint,
    }


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


def _validate_flags(args: argparse.Namespace) -> None:
    """Probing is opt-in, and a probe source without it is an error rather than an upgrade."""
    has_source = bool(args.probes) or int(args.holdout) > 0
    if has_source and not args.probe:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=(
                "--probes/--holdout was given but probing is disabled; this verb never reaches "
                "the embedding backend unless asked to"
            ),
            remediation=(
                "add --probe to enable the probe step (it embeds new inputs at probe time), or "
                "drop --probes/--holdout to run the corpus-coverage half alone"
            ),
        )
    if args.probe and not has_source:
        raise CliError(
            code=EXIT_USER_ERROR,
            message="--probe was given with no probe inputs to send",
            remediation=(
                "pass --probes <file> with one probe text per line, or --holdout <n> to hold "
                "out n atlas items and re-embed their text at probe time"
            ),
        )


def _validate_numbers(
    *,
    clusters: int,
    runs: int,
    resamples: int,
    sparsity_quantile: float,
    void_factor: float,
    coherence_factor: float,
) -> None:
    if clusters < 2:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"--clusters must be at least 2, got {clusters}",
            remediation=(
                "pass --clusters 2 or higher; a single cluster has no sparse and dense halves "
                f"to tell apart (default: {DEFAULT_CLUSTERS})"
            ),
        )
    if runs < 1:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"--runs must be at least 1, got {runs}",
            remediation=(
                "pass --runs 2 or higher for seed-rerun evidence; --runs 1 with --resamples 0 "
                "is reported as 'stability unverified'"
            ),
        )
    if resamples < 0:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"--resamples must be zero or more, got {resamples}",
            remediation="pass --resamples 0 to skip bootstrap resampling",
        )
    if not 0.0 < sparsity_quantile < 1.0:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"--sparsity-quantile must be strictly between 0 and 1, got "
            f"{sparsity_quantile}",
            remediation=(
                f"pass a fraction such as {DEFAULT_SPARSITY_QUANTILE} — the share of regions, "
                "most isolated first, reported as sparse"
            ),
        )
    if void_factor <= 0.0:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"--void-factor must be positive, got {void_factor}",
            remediation=f"pass a positive multiple (default: {DEFAULT_VOID_FACTOR})",
        )
    if coherence_factor <= 0.0:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"--coherence-factor must be positive, got {coherence_factor}",
            remediation=f"pass a positive fraction (default: {DEFAULT_COHERENCE_FACTOR})",
        )


def _active_matrix(atlas: Atlas, excluded: set[int], clusters: int) -> tuple[np.ndarray, list[int]]:
    """The vector matrix and record positions this run analyses, minus any held-out items."""
    if not atlas.records:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"the atlas at {atlas.path} contains no records — there is nothing to analyse",
            remediation="re-run `embeddings survey <corpus> --apply` with a non-empty corpus",
        )
    if atlas.vectors.ndim != 2 or atlas.vectors.shape[1] == 0:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=f"the atlas at {atlas.path} has no vectors to analyse",
            remediation=(
                "vectors.npy is regenerable bulk and is git-ignored; re-run the survey that "
                "produced this atlas to rebuild it"
            ),
        )

    positions = [index for index in range(len(atlas.records)) if index not in excluded]
    if clusters > len(positions):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=(
                f"--clusters {clusters} exceeds the {len(positions)} item(s) left to analyse "
                f"({len(excluded)} held out)"
            ),
            remediation=(
                f"pass --clusters {len(positions)} or fewer, lower --holdout, or survey a "
                "larger corpus"
            ),
        )
    rows = [atlas.records[position].vector_index for position in positions]
    if min(rows) < 0 or max(rows) >= atlas.vectors.shape[0]:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=(
                f"the atlas at {atlas.path} has a record pointing outside its vector array "
                f"(indices {min(rows)}..{max(rows)}, {atlas.vectors.shape[0]} vectors)"
            ),
            remediation="re-run the survey that produced this atlas to rebuild it",
        )
    return np.asarray(atlas.vectors[rows], dtype=np.float64), positions


# --------------------------------------------------------------------------
# The corpus-coverage half
# --------------------------------------------------------------------------


def _void_findings(
    *,
    partition: _Partition,
    region_ids: dict[int, int],
    data: np.ndarray,
    ids: list[str],
    void_factor: float,
    baseline_spread: float | None,
    max_voids: int,
    asserted: bool,
    stability_verdict: str,
) -> list[dict[str, object]]:
    """Midpoints between region centroids that no surveyed item is near."""
    reference = float(np.median(partition.nearest_neighbor))
    basis = "median nearest-neighbour distance"
    if reference <= 0.0:
        # Duplicate texts make the median zero; fall back to a scale that still
        # says something about how this model spaces this corpus, and name the
        # substitution in the finding rather than silently rescaling.
        reference = float(partition.nearest_neighbor.mean())
        basis = "mean nearest-neighbour distance (median was zero: duplicate items)"
    if reference <= 0.0 and baseline_spread:
        reference = baseline_spread
        basis = "corpus-wide baseline spread (all nearest-neighbour distances were zero)"
    threshold = void_factor * reference

    cluster_ids = sorted(partition.isolation)
    candidates: list[tuple[float, tuple[int, int], str]] = []
    for left_position, left in enumerate(cluster_ids):
        for right in cluster_ids[left_position + 1 :]:
            midpoint = (partition.centroids[left] + partition.centroids[right]) / 2.0
            distances = _cosine_distance_matrix(data, midpoint[None, :])[:, 0]
            nearest = int(np.argmin(distances))
            # Reported region ids, never k-means' internal labels: a void that
            # named clusters while the sparse findings named reported regions
            # would put two incompatible numbering schemes in one payload, and
            # "region 2" would mean different things three lines apart.
            left_id, right_id = sorted((region_ids[left], region_ids[right]))
            candidates.append((float(distances[nearest]), (left_id, right_id), ids[nearest]))

    candidates.sort(key=lambda item: (-item[0], item[1]))
    return [
        _void_finding(
            pair=pair,
            nearest_id=nearest_id,
            nearest_distance=distance,
            reference_distance=reference,
            reference_basis=basis,
            void_threshold=threshold,
            asserted=asserted,
            stability_verdict=stability_verdict,
        )
        for distance, pair, nearest_id in candidates[:max_voids]
        if distance > threshold
    ]


# --------------------------------------------------------------------------
# The probe half
# --------------------------------------------------------------------------


@dataclass
class _ProbeOutcome:
    """What the probe step produced — findings, inconclusive attempts, and its own status."""

    step: dict[str, object]
    findings: list[dict[str, object]] = field(default_factory=list)
    inconclusive: list[dict[str, object]] = field(default_factory=list)
    diagnostic: str | None = None


def _embed_probes(
    probes: list[_Probe], args: argparse.Namespace, atlas: Atlas
) -> tuple[np.ndarray, EmbeddingClient]:
    """Embed probe texts at probe time. The only place in this module that leaves the machine."""
    client = build_client(
        getattr(args, "embed_url", None), getattr(args, "embed_model", None) or atlas.manifest.model
    )
    vectors = get_or_embed(EmbeddingCache(), client, [probe.text for probe in probes])
    return np.asarray(vectors, dtype=np.float64), client


def _measure_probes(
    *,
    probes: list[_Probe],
    probe_vectors: np.ndarray,
    partition: _Partition,
    data: np.ndarray,
    ids: list[str],
    sparse_regions: dict[int, tuple[int, list[str]]],
    baseline_spread: float | None,
    coherence_factor: float,
    clusters: int,
    client: EmbeddingClient,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    """Assign probes to regions and turn each sparse region's probes into a verdict.

    Returns ``(model_findings, inconclusive)``. A sparse region only yields a
    model finding when at least two probes landed nearest it — with fewer there
    is no spread to compare against anything, and a verdict from one point would
    be a verdict from nothing.
    """
    cluster_ids = sorted(partition.isolation)
    centroids = np.asarray([partition.centroids[cluster] for cluster in cluster_ids])
    to_centroid = _cosine_distance_matrix(probe_vectors, centroids)
    assigned = [cluster_ids[int(index)] for index in np.argmin(to_centroid, axis=1)]
    to_corpus = _cosine_distance_matrix(probe_vectors, data)

    findings: list[dict[str, object]] = []
    inconclusive: list[dict[str, object]] = []

    for cluster, (region_id, members) in sorted(sparse_regions.items(), key=lambda kv: kv[1][0]):
        column = cluster_ids.index(cluster)
        landed = [index for index, target in enumerate(assigned) if target == cluster]
        probe_details = [
            _probe_payload(
                probes[index],
                {
                    "landed_in_region": True,
                    "distance_to_region_centroid": _round(float(to_centroid[index, column])),
                    "nearest_corpus_item": {
                        "id": ids[int(np.argmin(to_corpus[index]))],
                        "distance": _round(float(to_corpus[index].min())),
                    },
                },
            )
            for index in landed
        ]
        nearest_mean = (
            float(np.mean([to_corpus[index].min() for index in landed])) if landed else None
        )
        region_positions = np.flatnonzero(partition.labels == cluster)
        measurement: dict[str, object] = {
            "metric": METRIC,
            "probes_sent": len(probes),
            "probes_landed": len(landed),
            "probe_spread": _round(_mean_pairwise_distance(probe_vectors[landed])),
            "region_spread": _round(_mean_pairwise_distance(data[region_positions])),
            "corpus_baseline_spread": _round(baseline_spread),
            "coherence_threshold": _round(
                None if baseline_spread is None else coherence_factor * baseline_spread
            ),
            "mean_distance_to_nearest_corpus_item": _round(nearest_mean),
        }

        if not landed:
            inconclusive.append(
                _inconclusive(
                    region_id,
                    PROBE_VERDICT_ELSEWHERE,
                    measurement,
                    f"None of the {len(probes)} probe(s) landed nearest region {region_id}; "
                    "they were nearer other regions. That says the probes were not instances "
                    "of what this region holds, or that the region is unreachable by them — "
                    "it says nothing about how the model represents the region.",
                )
            )
            continue

        spread = measurement["probe_spread"]
        threshold = measurement["coherence_threshold"]
        if spread is None or threshold is None:
            inconclusive.append(
                _inconclusive(
                    region_id,
                    PROBE_VERDICT_TOO_FEW,
                    measurement,
                    f"Only {len(landed)} probe(s) landed nearest region {region_id}, so there "
                    "is no pairwise spread to compare against the corpus baseline. A verdict "
                    "from a single point would be a verdict from nothing.",
                )
            )
            continue

        verdict = (
            PROBE_VERDICT_COHERENT if float(spread) < float(threshold) else PROBE_VERDICT_SCATTERED
        )
        findings.append(
            _model_finding(
                region_id=region_id,
                members=members,
                verdict=verdict,
                measurement=measurement,
                probes=probe_details,
                coherence_factor=coherence_factor,
                clusters=clusters,
                model=client.model,
                endpoint=client.base_url,
            )
        )

    return findings, inconclusive


def _inconclusive(
    region_id: int, verdict: str, measurement: dict[str, object], note: str
) -> dict[str, object]:
    """A probe attempt that produced no finding — recorded, never silently dropped.

    "The probe ran and told us nothing" and "no probe ran" are different states,
    and a reader who cannot tell them apart will read the first as the second.
    """
    return {
        "region_id": region_id,
        "verdict": verdict,
        "measurement": measurement,
        "shows": note,
        "does_not_show": (
            "No model-representation finding follows from this. It is recorded so the attempt "
            "is visible, not as evidence in either direction."
        ),
    }


def _run_probe_step(
    *,
    args: argparse.Namespace,
    atlas: Atlas,
    probes: list[_Probe],
    sources: list[dict[str, object]],
    partition: _Partition,
    data: np.ndarray,
    ids: list[str],
    sparse_regions: dict[int, tuple[int, list[str]]],
    baseline_spread: float | None,
    coherence_factor: float,
    clusters: int,
) -> _ProbeOutcome:
    """Embed and measure the probes, degrading to "unavailable" if the gateway is down."""
    step: dict[str, object] = {
        "requested": True,
        "attempted": True,
        "available": False,
        "reason": None,
        "sources": sources,
        "probe_count": len(probes),
        "model": None,
        "endpoint": None,
    }
    try:
        probe_vectors, client = _embed_probes(probes, args, atlas)
    except CliError as err:
        if err.code != EXIT_ENV_ERROR:
            # A user error (a bad --embed-url, say) is the operator's to fix and
            # must not be laundered into "the gateway happened to be down".
            raise
        step["reason"] = err.message
        return _ProbeOutcome(
            step=step,
            diagnostic=(
                f"probe step unavailable: {err.message}\n"
                f"hint: {err.remediation}\n"
                "corpus-coverage findings are still reported below; no model-representation "
                "finding can be made without embedding probe inputs at probe time."
            ),
        )

    if probe_vectors.shape[1] != data.shape[1]:
        step["reason"] = (
            f"probe vectors have dimension {probe_vectors.shape[1]} but this atlas's vectors "
            f"have dimension {data.shape[1]}"
        )
        step["model"] = client.model
        step["endpoint"] = client.base_url
        return _ProbeOutcome(
            step=step,
            diagnostic=(
                f"probe step unavailable: {step['reason']}\n"
                f"hint: pass {MODEL_FLAG} to match the model this atlas was surveyed with "
                f"({atlas.manifest.model!r})"
            ),
        )

    step["available"] = True
    step["model"] = client.model
    step["endpoint"] = client.base_url
    findings, inconclusive = _measure_probes(
        probes=probes,
        probe_vectors=probe_vectors,
        partition=partition,
        data=data,
        ids=ids,
        sparse_regions=sparse_regions,
        baseline_spread=baseline_spread,
        coherence_factor=coherence_factor,
        clusters=clusters,
        client=client,
    )
    return _ProbeOutcome(step=step, findings=findings, inconclusive=inconclusive)


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _render(payload: dict) -> str:
    """Text-mode rendering of the same payload ``--json`` emits.

    The two halves get their own headings and their own counts, and the model
    half always renders — announcing that it is empty and *why* — because an
    absent section reads as "nothing to worry about" rather than "not measured".
    """
    params = payload["params"]
    step = payload["probe_step"]
    lines = [
        "# gaps",
        "",
        f"atlas:         {payload['atlas']}",
        f"model:         {payload['model']}",
        f"items:         {payload['items_analyzed']} analysed of {payload['items_total']}",
        f"params:        k={params['clusters']}, sparsity_quantile={params['sparsity_quantile']}, "
        f"void_factor={params['void_factor']}, coherence_factor={params['coherence_factor']}",
        f"seed:          {payload['seed']}",
        f"partition:     {payload['stability']['verdict']}",
        "",
        payload["disclaimer"],
        "",
        f"## corpus coverage — {payload['counts']['corpus_coverage']} finding(s)",
        "   what the surveyed corpus does and does not contain. Not about the model.",
        "",
    ]
    for finding in payload["corpus_coverage_findings"]:
        lines.append(f"- [{finding['type']}] {finding['shows']}")
        lines.append(f"  does not show: {finding['does_not_show']}")
        lines.append(f"  asserted: {finding['asserted']} ({finding['stability_verdict']})")
        lines.append("")
    if not payload["corpus_coverage_findings"]:
        lines += ["(no area of this atlas met the sparsity or void thresholds above)", ""]

    lines.append(
        f"## model representation — {payload['counts']['model_representation']} finding(s)"
    )
    lines.append("   what probing actually showed. Only ever produced by a live probe step.")
    lines.append("")
    if not step["requested"]:
        lines += [
            "no probe step ran (probing is off by default), so no claim about the model is "
            "made here.",
            "run again with --probe and --probes <file> (or --holdout <n>) to test the model.",
            "",
        ]
    elif not step["available"]:
        lines += [
            f"probe step unavailable: {step['reason']}",
            "the corpus-coverage findings above still stand; no model-representation finding "
            "can be made without embedding probe inputs at probe time.",
            "",
        ]
    else:
        for finding in payload["model_representation_findings"]:
            lines.append(f"- [{finding['verdict']}] {finding['shows']}")
            lines.append(f"  does not show: {finding['does_not_show']}")
            lines.append("")
        if not payload["model_representation_findings"]:
            lines += ["(the probe step ran but produced no finding; see inconclusive below)", ""]

    for entry in payload["inconclusive_probes"]:
        lines.append(f"inconclusive [{entry['verdict']}]: {entry['shows']}")
    if payload["inconclusive_probes"]:
        lines.append("")

    uncertain = payload["uncertain"]
    if uncertain:
        lines.append("uncertain (belongs in the atlas manifest's uncertain[]):")
        lines += [f"- {entry}" for entry in uncertain]
    lines.append("")
    lines.append(reproduce_line(payload[REPRODUCED_BY_KEY]))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def cmd_gaps(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    clusters = int(args.clusters)
    runs = int(args.runs)
    resamples = int(args.resamples)
    seed = int(args.seed)
    sparsity_quantile = float(args.sparsity_quantile)
    void_factor = float(args.void_factor)
    coherence_factor = float(args.coherence_factor)
    max_iterations = int(args.max_iterations)
    tolerance = float(args.tolerance)

    _validate_flags(args)
    _validate_numbers(
        clusters=clusters,
        runs=runs,
        resamples=resamples,
        sparsity_quantile=sparsity_quantile,
        void_factor=void_factor,
        coherence_factor=coherence_factor,
    )

    atlas = load_atlas(args)

    # Held-out items leave the analysis entirely: they are excluded from the
    # clustering, from the sparsity statistics, and from the reference set the
    # probes are measured against. Leaving them in the reference set would make
    # every held-out probe its own nearest neighbour at distance ~0, and every
    # probe would "land coherently" by construction.
    probes: list[_Probe] = []
    sources: list[dict[str, object]] = []
    excluded: set[int] = set()
    holdout = int(args.holdout)
    if args.probe and holdout > 0:
        positions = _holdout_positions(atlas, holdout, seed)
        excluded = {int(position) for position in positions}
        probes += [
            _Probe(
                text=atlas.records[int(position)].text,
                provenance={
                    "source": PROVENANCE_HOLDOUT,
                    "atlas_id": atlas.records[int(position)].id,
                    "note": (
                        "held out of this run's clustering, sparsity statistics and probe "
                        "reference set, then re-embedded at probe time"
                    ),
                },
            )
            for position in positions
        ]
        sources.append({"source": PROVENANCE_HOLDOUT, "count": holdout, "seed": seed})
    if args.probe and args.probes:
        file_probes = _read_probes_file(str(args.probes))
        probes += file_probes
        sources.append(
            {"source": PROVENANCE_FILE, "path": str(args.probes), "count": len(file_probes)}
        )

    data, positions = _active_matrix(atlas, excluded, clusters)
    ids = [atlas.records[position].id for position in positions]

    partition = _partition(
        data,
        seed,
        clusters=clusters,
        sparsity_quantile=sparsity_quantile,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    baseline_spread = _mean_pairwise_distance(data)

    params: dict[str, object] = {
        "metric": METRIC,
        "algorithm": "kmeans",
        "clusters": clusters,
        "sparsity_quantile": sparsity_quantile,
        "void_factor": void_factor,
        "coherence_factor": coherence_factor,
        "max_voids": int(args.max_voids),
        "max_iterations": max_iterations,
        "tolerance": tolerance,
        "runs": runs,
        "resamples": resamples,
        "holdout": holdout if args.probe else 0,
        "holdout_ids": [atlas.records[position].id for position in sorted(excluded)],
    }

    # The structural claim under test is "these items are the sparse ones", so
    # that binary split — not the k-way partition it came from — is what the
    # stability apparatus measures. A sparse set that only appears at one seed is
    # a finding about the seed (design note SS3).
    def _sparse_partition(matrix: np.ndarray, run_seed: int) -> np.ndarray:
        return (
            _partition(
                np.asarray(matrix, dtype=np.float64),
                run_seed,
                clusters=clusters,
                sparsity_quantile=sparsity_quantile,
                max_iterations=max_iterations,
                tolerance=tolerance,
            )
            .sparse_mask()
            .astype(np.int64)
        )

    stability: StabilityResult = measure_stability(
        data, _sparse_partition, params=params, seed=seed, runs=runs, resamples=resamples
    )
    asserted = not stability.is_uncertain

    # One reported numbering covering *every* region in the partition, most
    # isolated first — so the sparsest area a reader is likely to act on is the
    # one they read first, and so a sparse-region finding and a void finding
    # that both say "region 2" mean the same region. k-means' own label
    # numbering is arbitrary per run and is never surfaced.
    ordered_clusters = sorted(
        partition.isolation, key=lambda cluster: (-partition.isolation[cluster], cluster)
    )
    region_ids = {cluster: index for index, cluster in enumerate(ordered_clusters)}

    sparse_regions: dict[int, tuple[int, list[str]]] = {}
    corpus_findings: list[dict[str, object]] = []
    for cluster in ordered_clusters:
        if cluster not in partition.sparse_clusters:
            continue
        region_id = region_ids[cluster]
        member_positions = np.flatnonzero(partition.labels == cluster)
        members = [ids[int(position)] for position in member_positions]
        sparse_regions[cluster] = (region_id, members)
        corpus_findings.append(
            _sparse_region_finding(
                region_id=region_id,
                members=members,
                isolation=partition.isolation[cluster],
                partition=partition,
                region_spread=_mean_pairwise_distance(data[member_positions]),
                baseline_spread=baseline_spread,
                asserted=asserted,
                stability_verdict=stability.verdict,
            )
        )

    corpus_findings += _void_findings(
        partition=partition,
        region_ids=region_ids,
        data=data,
        ids=ids,
        void_factor=void_factor,
        baseline_spread=baseline_spread,
        max_voids=int(args.max_voids),
        asserted=asserted,
        stability_verdict=stability.verdict,
    )

    if args.probe:
        outcome = _run_probe_step(
            args=args,
            atlas=atlas,
            probes=probes,
            sources=sources,
            partition=partition,
            data=data,
            ids=ids,
            sparse_regions=sparse_regions,
            baseline_spread=baseline_spread,
            coherence_factor=coherence_factor,
            clusters=clusters,
        )
    else:
        outcome = _ProbeOutcome(
            step={
                "requested": False,
                "attempted": False,
                "available": False,
                "reason": (
                    "probing is off by default; it is the only path in this verb that reaches "
                    "the embedding backend, so it must be asked for with --probe"
                ),
                "sources": [],
                "probe_count": 0,
                "model": None,
                "endpoint": None,
            }
        )

    uncertain: list[str] = []
    if stability.is_uncertain and (corpus_findings or outcome.findings):
        uncertain.append(stability.uncertain_note("the sparse/dense split reported here"))

    payload: dict[str, object] = {
        "atlas": str(atlas.path),
        "model": atlas.manifest.model,
        "metric": METRIC,
        "items_total": len(atlas.records),
        "items_analyzed": int(data.shape[0]),
        "dimensions": int(data.shape[1]),
        "disclaimer": DISCLAIMER,
        # Two keys, never one list. See the module docstring: a shared list with
        # a `kind` field is one `+` away from erasing the distinction.
        "corpus_coverage_findings": corpus_findings,
        "model_representation_findings": outcome.findings,
        "inconclusive_probes": outcome.inconclusive,
        "counts": {
            "corpus_coverage": len(corpus_findings),
            "model_representation": len(outcome.findings),
            "inconclusive_probes": len(outcome.inconclusive),
        },
        "probe_step": outcome.step,
        "params": params,
        "seed": seed,
        "stability": stability.to_dict(),
        "uncertain": uncertain,
        REPRODUCED_BY_KEY: reproducing_command(),
    }

    if outcome.diagnostic:
        emit_diagnostic(outcome.diagnostic)
    emit_result(payload if json_mode else _render(payload), json_mode=json_mode)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "gaps",
        help="Find under-covered areas, labeled corpus-coverage or model-representation.",
        description=(
            "Report areas of a surveyed atlas the corpus covers thinly, and — only when "
            "--probe actively embeds new inputs at probe time — what probing those areas "
            "shows about the model. The two kinds are emitted as separate keys and are never "
            "merged: corpus sparsity alone says nothing about model capability."
        ),
    )
    add_atlas_argument(p)
    p.add_argument(
        "--clusters",
        "-k",
        type=int,
        default=DEFAULT_CLUSTERS,
        help=f"Number of regions to partition into before measuring sparsity (default: "
        f"{DEFAULT_CLUSTERS}). Which areas read as sparse depends on this; vary it and "
        "compare the stability reported back.",
    )
    p.add_argument(
        "--sparsity-quantile",
        type=float,
        default=DEFAULT_SPARSITY_QUANTILE,
        help=f"Share of regions, most isolated first, reported as sparse (default: "
        f"{DEFAULT_SPARSITY_QUANTILE}). Isolation is a region's mean cosine distance from "
        "its members to their nearest neighbour. A threshold, not a discovered truth.",
    )
    p.add_argument(
        "--void-factor",
        type=float,
        default=DEFAULT_VOID_FACTOR,
        help=f"Multiple of the corpus's own median nearest-neighbour distance above which a "
        f"midpoint between two region centroids is reported as a void (default: "
        f"{DEFAULT_VOID_FACTOR}). Relative to the corpus because 'far' only means "
        "something against how tightly this model packs this corpus.",
    )
    p.add_argument(
        "--max-voids",
        type=int,
        default=DEFAULT_MAX_VOIDS,
        help=f"Cap on reported inter-region voids, largest first (default: "
        f"{DEFAULT_MAX_VOIDS}). Region pairs grow quadratically in k.",
    )
    p.add_argument(
        "--probe",
        dest="probe",
        action="store_true",
        default=False,
        help="Run the probe step: embed probe inputs at probe time and measure where they "
        "land. OFF by default -- this is the only path in this verb that reaches the "
        "embedding backend, and a model-representation finding is impossible without it.",
    )
    p.add_argument(
        "--no-probe",
        dest="probe",
        action="store_false",
        help="Explicitly disable the probe step (the default). Corpus-coverage findings only; "
        "the model-representation list is emitted empty rather than omitted.",
    )
    p.add_argument(
        "--probes",
        default=None,
        metavar="<file>",
        help="File of probe inputs, one text per line (blank lines and # comments skipped). "
        "Requires --probe. Each probe's file and line number is recorded on the finding so "
        "a reader can audit exactly what was sent to the model.",
    )
    p.add_argument(
        "--holdout",
        type=int,
        default=0,
        help="Hold out n atlas items, exclude them from the clustering, the sparsity "
        "statistics and the probe reference set, then re-embed their text at probe time "
        "(default: 0). Requires --probe. Selection is reproducible from --seed.",
    )
    p.add_argument(
        "--coherence-factor",
        type=float,
        default=DEFAULT_COHERENCE_FACTOR,
        help=f"Fraction of the corpus-wide baseline spread below which probes count as "
        f"cohering rather than scattering (default: {DEFAULT_COHERENCE_FACTOR}). Probes at "
        "or above it are no tighter than this model places arbitrary text.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Base RNG seed (default: {DEFAULT_SEED}). The reported partition, every "
        "stability rerun, and the holdout selection all derive from this one value.",
    )
    p.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Seed reruns used to measure whether the sparse/dense split reproduces "
        f"(default: {DEFAULT_RUNS}). Reruns cost clustering work but no re-embedding. "
        "--runs 1 with --resamples 0 is reported as 'stability unverified'.",
    )
    p.add_argument(
        "--resamples",
        type=int,
        default=DEFAULT_RESAMPLES,
        help=f"Bootstrap resamples of the item set (default: {DEFAULT_RESAMPLES}); 0 skips "
        "the bootstrap. This is the check that catches a sparse area held up by a "
        "handful of items.",
    )
    p.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"Cap on Lloyd iterations per clustering (default: {DEFAULT_MAX_ITERATIONS}).",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"Centroid movement below which a clustering is called converged "
        f"(default: {DEFAULT_TOLERANCE}).",
    )
    # Flag names come from the backend constants, matching survey.py and
    # neighbors.py, so the flag, its help text, and the client's own
    # unreachable-endpoint hint can never drift apart. Both are only consulted
    # on the probe path.
    p.add_argument(ENDPOINT_FLAG, default=None, help=ENDPOINT_HELP + " Probe step only.")
    p.add_argument(
        MODEL_FLAG,
        default=None,
        help=MODEL_HELP + " Probe step only; defaults to the atlas's own model.",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_gaps)
