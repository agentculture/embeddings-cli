"""``embeddings regions`` — cluster an Atlas into semantic regions, with the parameters attached.

What this verb is allowed to claim
----------------------------------
A clustering is a *parameter-dependent artifact*. k-means returns k
well-separated-looking blobs whether or not the space has any, and the specific
partition moves with ``k``, with the item set, and with the seed alone. So the
one thing this verb must never emit is a bare "the space has three regions":
that is a statement about a ``k`` someone chose. Per
``docs/design-notes/semantic-measurement.md`` §3, every region here carries its
own stability measure, and the full parameter set plus the seed are stamped on
the payload, on the top-level stability result, and on each region's.

Low-stability regions are **still reported**. Suppressing them would hide real
ambiguity in the space, which is itself a finding. What changes is their
epistemic status: an unstable or unverified region has ``asserted: false`` and a
``finding`` string that says, in words, that it is a finding about the
parameters rather than about the model — and it is copied into the payload's
``uncertain[]``, the section
:class:`~embeddings.atlas.schema.AtlasManifest` reserves for exactly this.

Why k-means, and why that choice is load-bearing
------------------------------------------------
The algorithm is Lloyd's k-means with k-means++ seeding, implemented here on
numpy alone. It was chosen because it (a) adds no dependency — HDBSCAN would,
and this CLI's dependency surface is deliberately one library wide; (b) is
genuinely seed-sensitive through its initialization, so the stability apparatus
has something real to measure rather than a pure function that would score 1.0
by construction; and (c) is the algorithm the design note names, so its known
failure mode (it always returns exactly ``k`` non-empty clusters, spherical and
equal-ish, structure or no structure) is the failure mode §3's labeling exists
to catch.

That choice is recorded in every payload as ``algorithm`` and inside ``params``.
Changing it later would reshape committed study evidence: an Atlas region set
produced by k-means and one produced by a density-based method are not the same
claim, and a reader comparing them across a version boundary must be able to
see which produced what.

What is deterministic here
--------------------------
Everything. This is a reader verb over a fixed Atlas — design note §4's
guaranteed side of the line. The reported partition is the clustering at the
base ``--seed``; the rerun and resample streams are derived from that same seed
through :class:`numpy.random.SeedSequence`, never from process-global RNG
state. Same Atlas, same flags, same bytes out. No socket is opened: vectors come
from the Atlas on disk, so ``regions`` works with the gateway down.

The two stability measures, and why they are different measures
---------------------------------------------------------------
* **Whole-partition stability** (top level) uses the Adjusted Rand Index across
  seed reruns and bootstrap resamples, via
  :func:`embeddings.measure.measure_stability`. It answers "does this
  *partition* reproduce?"
* **Per-region stability** uses maximum Jaccard overlap between the region's
  member set and the best-matching cluster in each rerun/resample. It answers
  "does *this particular group of items* keep being found?" — a question ARI
  cannot answer, because ARI scores a whole partition and a single unstable
  region can hide inside an otherwise reproducible one. Both are named in the
  output (``measure``) so no reader has to guess which statistic a number is.
"""

from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np

from embeddings.atlas.io import Atlas
from embeddings.cli._commands._atlas_arg import add_atlas_argument, load_atlas
from embeddings.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from embeddings.cli._output import emit_result
from embeddings.measure import (
    AgreementDistribution,
    StabilityResult,
    measure_stability,
    seed_reruns,
)
from embeddings.measure.stability import DEFAULT_RESAMPLES, DEFAULT_RUNS

#: The clustering algorithm this verb implements, stamped on every payload.
#: See the module docstring for why a change here is not a free refactor.
ALGORITHM = "kmeans"

#: The per-region agreement statistic. Deliberately *not* the Adjusted Rand
#: Index used for the whole partition — see the module docstring.
MEASURE_JACCARD = "max_jaccard_overlap"

#: Number of regions to partition into when ``--clusters`` is omitted. There is
#: no principled default cluster count for an unknown space; this is a starting
#: point for a Surveyor to vary, which is exactly why it is reported back.
DEFAULT_CLUSTERS = 5
#: Base seed. Every derived seed in the run comes from this one value.
DEFAULT_SEED = 0
#: Lloyd-iteration cap. Reached only when the partition is still churning.
DEFAULT_MAX_ITERATIONS = 100
#: Centroid-movement threshold below which the iteration is called converged.
DEFAULT_TOLERANCE = 1e-6
#: Representative items reported per region, nearest-centroid first.
DEFAULT_REPRESENTATIVES = 3


# --------------------------------------------------------------------------
# k-means on numpy alone
# --------------------------------------------------------------------------


def _squared_distances(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    """``(n_items, n_centroids)`` squared euclidean distances, clipped at zero.

    Computed through the ``|a|^2 - 2ab + |b|^2`` expansion, which can go a hair
    negative on float round-off; the clip keeps ``argmin`` and any later square
    root well defined.
    """
    item_norms = np.einsum("ij,ij->i", data, data)[:, None]
    centroid_norms = np.einsum("ij,ij->i", centroids, centroids)[None, :]
    cross = data @ centroids.T
    return np.maximum(item_norms - 2.0 * cross + centroid_norms, 0.0)


def _initial_centroids(data: np.ndarray, clusters: int, rng: np.random.Generator) -> np.ndarray:
    """k-means++ seeding: spread the initial centroids by squared-distance sampling.

    This is where the seed enters the algorithm. Plain random-point
    initialization would work too, but k-means++ converges from a better start
    while remaining genuinely seed-dependent — the property the stability
    apparatus needs in order to be measuring anything at all.
    """
    chosen = [int(rng.integers(0, data.shape[0]))]
    while len(chosen) < clusters:
        nearest = _squared_distances(data, data[chosen]).min(axis=1)
        total = float(nearest.sum())
        if total <= 0.0:
            # Every remaining point coincides with a chosen centroid; sampling
            # proportional to zero is undefined, so fall back to uniform.
            chosen.append(int(rng.integers(0, data.shape[0])))
            continue
        chosen.append(int(rng.choice(data.shape[0], p=nearest / total)))
    return np.asarray(data[chosen], dtype=np.float64)


def _fill_empty_clusters(labels: np.ndarray, distances: np.ndarray, clusters: int) -> np.ndarray:
    """Guarantee ``clusters`` non-empty groups by donating worst-fitting items.

    An empty cluster would make the reported region count disagree with the
    ``k`` in ``params``, and a reader comparing two runs would have no way to
    tell "k=5 found 4 regions" from "k=4 was requested". The donor is the item
    farthest from its own centroid among clusters that can spare one, so the
    choice is deterministic and takes the least confidently placed item.
    """
    labels = labels.copy()
    own_distance = distances[np.arange(labels.shape[0]), labels]
    for cluster in range(clusters):
        if np.any(labels == cluster):
            continue
        sizes = np.bincount(labels, minlength=clusters)
        spare = sizes[labels] > 1
        if not np.any(spare):  # pragma: no cover - needs clusters > n_items
            break
        candidate_distance = np.where(spare, own_distance, -np.inf)
        labels[int(np.argmax(candidate_distance))] = cluster
    return labels


def kmeans(
    data: np.ndarray,
    seed: int,
    *,
    clusters: int,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> tuple[np.ndarray, np.ndarray]:
    """Lloyd's k-means under ``seed``; returns ``(labels, centroids)``.

    Deterministic given ``(data, seed, clusters, max_iterations, tolerance)``:
    the only randomness is the k-means++ initialization, drawn from a
    :class:`numpy.random.Generator` seeded here and nowhere else.
    """
    array = np.asarray(data, dtype=np.float64)
    effective = min(clusters, array.shape[0])
    centroids = _initial_centroids(array, effective, np.random.default_rng(seed))

    for _ in range(max_iterations):
        distances = _squared_distances(array, centroids)
        labels = np.argmin(distances, axis=1)
        updated = centroids.copy()
        for cluster in range(effective):
            members = array[labels == cluster]
            if members.shape[0]:
                updated[cluster] = members.mean(axis=0)
        shift = float(np.max(np.abs(updated - centroids))) if effective else 0.0
        centroids = updated
        if shift <= tolerance:
            break

    distances = _squared_distances(array, centroids)
    labels = _fill_empty_clusters(np.argmin(distances, axis=1), distances, effective)
    # Recompute centroids from the final labeling so the representatives a
    # region reports are nearest to the centroid of the region as *reported*,
    # not to a centroid from a superseded iteration.
    for cluster in range(effective):
        members = array[labels == cluster]
        if members.shape[0]:
            centroids[cluster] = members.mean(axis=0)
    return labels, centroids


def cluster_labels(
    data: np.ndarray,
    seed: int,
    *,
    clusters: int,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    tolerance: float = DEFAULT_TOLERANCE,
) -> np.ndarray:
    """:data:`~embeddings.measure.stability.ClusterFn`-shaped view of :func:`kmeans`."""
    labels, _ = kmeans(
        data,
        seed,
        clusters=clusters,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )
    return labels


# --------------------------------------------------------------------------
# Per-region stability
# --------------------------------------------------------------------------


def _max_jaccard(reference: np.ndarray, labels: np.ndarray) -> float:
    """Best Jaccard overlap between ``reference`` (a boolean mask) and any cluster."""
    best = 0.0
    for value in np.unique(labels):
        candidate = labels == value
        union = int(np.count_nonzero(reference | candidate))
        if union == 0:  # pragma: no cover - a cluster is never empty here
            continue
        best = max(best, int(np.count_nonzero(reference & candidate)) / union)
    return best


def _bootstrap_labelings(
    data: np.ndarray,
    seed: int,
    *,
    resamples: int,
    clusters: int,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """``(indices, labels)`` for each bootstrap resample, seeded from ``seed`` alone.

    Mirrors :func:`embeddings.measure.bootstrap_agreement`'s scheme — draw
    ``n`` indices with replacement, re-cluster that matrix — but keeps the
    labelings rather than collapsing them to one ARI, because per-region
    overlap has to be read off the labelings themselves.
    """
    n_items = data.shape[0]
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for child in np.random.SeedSequence(seed).spawn(resamples):
        index_sequence, cluster_sequence = child.spawn(2)
        indices = np.random.default_rng(index_sequence).integers(0, n_items, size=n_items)
        cluster_seed = int(cluster_sequence.generate_state(1, dtype=np.uint32)[0])
        out.append((indices, cluster_labels(data[indices], cluster_seed, clusters=clusters)))
    return out


def _region_stability(
    mask: np.ndarray,
    *,
    rerun_labelings: Sequence[np.ndarray],
    bootstrap: Sequence[tuple[np.ndarray, np.ndarray]],
    params: dict[str, object],
    seed: int,
    runs: int,
    resamples: int,
) -> StabilityResult:
    """Stability of one region: does *this group of items* keep being found?

    Seed evidence is gated at ``runs >= 2``, matching
    :func:`embeddings.measure.measure_stability`, so "one run" means the same
    thing — :data:`~embeddings.measure.VERDICT_UNVERIFIED` — at both levels of
    the payload rather than a region looking measured while the partition
    containing it does not.
    """
    seed_distribution = None
    if runs >= 2:
        seed_distribution = AgreementDistribution.from_values(
            MEASURE_JACCARD, [_max_jaccard(mask, labels) for labels in rerun_labelings]
        )

    bootstrap_values = [
        _max_jaccard(mask[indices], labels)
        for indices, labels in bootstrap
        if bool(np.any(mask[indices]))
    ]
    bootstrap_distribution = (
        AgreementDistribution.from_values(MEASURE_JACCARD, bootstrap_values)
        if bootstrap_values
        else None
    )

    notes: list[str] = []
    if seed_distribution is None and bootstrap_distribution is None:
        notes.append(
            "single run with no bootstrap resampling: this region is reported as "
            "'stability unverified', not as a measured finding"
        )
    elif seed_distribution is None:
        notes.append("no seed-rerun evidence: this region rests on bootstrap resampling alone")
    elif bootstrap_distribution is None:
        notes.append("no bootstrap evidence: this region rests on seed reruns alone")

    return StabilityResult(
        measure=MEASURE_JACCARD,
        params=dict(params),
        seed=seed,
        runs=runs,
        resamples=resamples,
        seed_agreement=seed_distribution,
        bootstrap_agreement=bootstrap_distribution,
        notes=tuple(notes),
    )


def _finding(region_id: int, stability: StabilityResult, params: dict[str, object]) -> str:
    """The sentence an unstable or unverified region is reported *as*.

    Deliberately phrased as a claim about the run rather than about the space:
    §3's rule is that a region surviving only one parameter setting is a finding
    about the parameters, and the output has to say so in words an agent will
    copy into an Atlas rather than leave it implied by a numeric field.
    """
    return (
        f"region {region_id} is not asserted as structure in the space: "
        f"{stability.describe()}. At these parameters "
        f"(algorithm={params['algorithm']}, k={params['clusters']}, seed={stability.seed}) "
        "it is a finding about the parameters, not about the model — vary k, the seed, "
        "and the item set before treating it as real."
    )


# --------------------------------------------------------------------------
# Assembly
# --------------------------------------------------------------------------


def _validate(
    *, clusters: int, runs: int, resamples: int, representatives: int, max_iterations: int
) -> None:
    if clusters < 2:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"--clusters must be at least 2, got {clusters}",
            remediation=(
                "pass --clusters 2 or higher; a single cluster is not a partition and "
                f"says nothing about the space (default: {DEFAULT_CLUSTERS})"
            ),
        )
    if runs < 1:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"--runs must be at least 1, got {runs}",
            remediation=(
                "pass --runs 2 or higher to obtain seed-rerun evidence; --runs 1 is "
                "reported as 'stability unverified'"
            ),
        )
    if resamples < 0:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"--resamples must be zero or more, got {resamples}",
            remediation="pass --resamples 0 to skip bootstrap resampling",
        )
    if representatives < 1:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"--representatives must be at least 1, got {representatives}",
            remediation=f"pass --representatives 1 or higher (default: {DEFAULT_REPRESENTATIVES})",
        )
    if max_iterations < 1:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"--max-iterations must be at least 1, got {max_iterations}",
            remediation=f"pass --max-iterations 1 or higher (default: {DEFAULT_MAX_ITERATIONS})",
        )


def _item_matrix(atlas: Atlas, clusters: int) -> np.ndarray:
    """One row per Atlas record, in record order, as float64.

    Records may share a ``vector_index`` (identical text is embedded once), so
    this is a gather rather than a slice of ``atlas.vectors``.
    """
    if not atlas.records:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"the atlas at {atlas.path} contains no records — there is nothing to cluster",
            remediation="re-run `embeddings survey <corpus> --apply` with a non-empty corpus",
        )
    if atlas.vectors.ndim != 2 or atlas.vectors.shape[1] == 0:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=f"the atlas at {atlas.path} has no vectors to cluster",
            remediation=(
                "vectors.npy is regenerable bulk and is git-ignored; re-run the survey "
                "that produced this atlas to rebuild it"
            ),
        )

    indices = [record.vector_index for record in atlas.records]
    if min(indices) < 0 or max(indices) >= atlas.vectors.shape[0]:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=(
                f"the atlas at {atlas.path} has a record pointing outside its vector array "
                f"(indices {min(indices)}..{max(indices)}, {atlas.vectors.shape[0]} vectors)"
            ),
            remediation="re-run the survey that produced this atlas to rebuild it",
        )
    if clusters > len(indices):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"--clusters {clusters} exceeds the {len(indices)} items in this atlas",
            remediation=f"pass --clusters {len(indices)} or fewer, or survey a larger corpus",
        )
    return np.asarray(atlas.vectors[indices], dtype=np.float64)


def _region_payload(
    *,
    region_id: int,
    member_positions: np.ndarray,
    atlas: Atlas,
    data: np.ndarray,
    centroid: np.ndarray,
    representatives: int,
    stability: StabilityResult,
    params: dict[str, object],
) -> dict[str, object]:
    distances = np.sqrt(_squared_distances(data[member_positions], centroid[None, :])[:, 0])
    order = np.argsort(distances, kind="stable")[:representatives]
    asserted = not stability.is_uncertain
    return {
        "id": region_id,
        "size": int(member_positions.shape[0]),
        "members": [atlas.records[int(position)].id for position in member_positions],
        "representatives": [
            {
                "id": atlas.records[int(member_positions[position])].id,
                "text": atlas.records[int(member_positions[position])].text,
                "distance": round(float(distances[position]), 6),
            }
            for position in order
        ],
        "stability": stability.to_dict(),
        "asserted": asserted,
        "finding": None if asserted else _finding(region_id, stability, params),
    }


def _render(payload: dict) -> str:
    """Text-mode rendering of the same payload ``--json`` emits.

    Deliberately untyped in its parameter: this reads a dict this module built
    a few lines earlier, and narrowing every lookup would add noise without
    adding a check.
    """
    params = payload["params"]
    lines = [
        "# regions",
        "",
        f"atlas:         {payload['atlas']}",
        f"model:         {payload['model']}",
        f"items:         {payload['items']} x {payload['dimensions']}",
        f"algorithm:     {payload['algorithm']}",
        f"params:        k={params['clusters']}, runs={params['runs']}, "
        f"resamples={params['resamples']}, max_iterations={params['max_iterations']}, "
        f"tolerance={params['tolerance']}",
        f"seed:          {payload['seed']}",
        f"partition:     {payload['verdict']}",
        "",
    ]
    for region in payload["regions"]:
        names = ", ".join(item["id"] for item in region["representatives"])
        lines.append(f"region {region['id']} — {region['size']} items")
        lines.append(f"  representatives: {names}")
        lines.append(f"  stability:       {region['stability']['verdict']}")
        if region["finding"]:
            lines.append(f"  finding:         {region['finding']}")
        lines.append("")

    uncertain = payload["uncertain"]
    if uncertain:
        lines.append("uncertain (belongs in the atlas manifest's uncertain[]):")
        lines += [f"- {entry}" for entry in uncertain]
    else:
        lines.append("every region above survived its stability check at these parameters.")
    return "\n".join(lines)


def cmd_regions(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    clusters = int(args.clusters)
    runs = int(args.runs)
    resamples = int(args.resamples)
    representatives = int(args.representatives)
    max_iterations = int(args.max_iterations)
    seed = int(args.seed)
    _validate(
        clusters=clusters,
        runs=runs,
        resamples=resamples,
        representatives=representatives,
        max_iterations=max_iterations,
    )

    atlas = load_atlas(args)
    data = _item_matrix(atlas, clusters)

    params: dict[str, object] = {
        "algorithm": ALGORITHM,
        "clusters": clusters,
        "max_iterations": max_iterations,
        "tolerance": float(args.tolerance),
        "runs": runs,
        "resamples": resamples,
        "representatives": representatives,
    }

    def _cluster(matrix: np.ndarray, run_seed: int) -> np.ndarray:
        return cluster_labels(
            matrix,
            run_seed,
            clusters=clusters,
            max_iterations=max_iterations,
            tolerance=float(args.tolerance),
        )

    # The *reported* partition is the clustering at the base seed. Everything
    # below measures how much that partition reproduces; it never replaces it,
    # so the regions a reader sees are the ones the stated seed produced.
    labels, centroids = kmeans(
        data,
        seed,
        clusters=clusters,
        max_iterations=max_iterations,
        tolerance=float(args.tolerance),
    )

    overall = measure_stability(
        data, _cluster, params=params, seed=seed, runs=runs, resamples=resamples
    )
    rerun_labelings = seed_reruns(data, _cluster, runs=runs, seed=seed) if runs >= 2 else []
    bootstrap = (
        _bootstrap_labelings(data, seed, resamples=resamples, clusters=clusters)
        if resamples
        else []
    )

    # Largest region first, ties broken by first member position: a stable
    # order that does not depend on k-means' arbitrary internal label numbering.
    groups = sorted(
        (np.flatnonzero(labels == cluster) for cluster in np.unique(labels)),
        key=lambda positions: (-positions.shape[0], int(positions[0])),
    )

    uncertain: list[str] = []
    region_payloads: list[dict[str, object]] = []
    for region_id, positions in enumerate(groups):
        mask = np.zeros(labels.shape[0], dtype=bool)
        mask[positions] = True
        stability = _region_stability(
            mask,
            rerun_labelings=rerun_labelings,
            bootstrap=bootstrap,
            params=params,
            seed=seed,
            runs=runs,
            resamples=resamples,
        )
        region_payloads.append(
            _region_payload(
                region_id=region_id,
                member_positions=positions,
                atlas=atlas,
                data=data,
                centroid=centroids[int(labels[positions[0]])],
                representatives=representatives,
                stability=stability,
                params=params,
            )
        )
        if stability.is_uncertain:
            uncertain.append(stability.uncertain_note(f"region {region_id}"))

    if overall.is_uncertain:
        uncertain.append(overall.uncertain_note(f"the {clusters}-region partition itself"))

    payload: dict[str, object] = {
        "atlas": str(atlas.path),
        "model": atlas.manifest.model,
        "items": int(data.shape[0]),
        "dimensions": int(data.shape[1]),
        "algorithm": ALGORITHM,
        "params": params,
        "seed": seed,
        "verdict": overall.verdict,
        "stability": overall.to_dict(),
        "regions": region_payloads,
        "uncertain": uncertain,
    }
    emit_result(payload if json_mode else _render(payload), json_mode=json_mode)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "regions",
        help="Cluster an atlas into semantic regions, with parameters, seed, and stability.",
        description=(
            "Partition a surveyed atlas into semantic regions and report each region's "
            "membership, size, representative items, and stability across seed reruns and "
            "bootstrap resamples. A region that survives only one parameter setting is "
            "reported as a finding about the parameters, not as a fact about the model."
        ),
    )
    add_atlas_argument(p)
    p.add_argument(
        "--clusters",
        "-k",
        type=int,
        default=DEFAULT_CLUSTERS,
        help=f"Number of regions to partition into (default: {DEFAULT_CLUSTERS}). There is no "
        "principled default for an unknown space — vary this and compare the stability "
        "reported back before treating any region count as a property of the model.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Base RNG seed (default: {DEFAULT_SEED}). The reported partition is the "
        "clustering at this seed; every rerun and resample seed is derived from it, so the "
        "whole run replays exactly.",
    )
    p.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Seed reruns used to measure stability (default: {DEFAULT_RUNS}). Reruns cost "
        "clustering work but no re-embedding. --runs 1 with --resamples 0 yields no evidence "
        "at all and is reported as 'stability unverified'.",
    )
    p.add_argument(
        "--resamples",
        type=int,
        default=DEFAULT_RESAMPLES,
        help=f"Bootstrap resamples of the item set (default: {DEFAULT_RESAMPLES}); 0 skips the "
        "bootstrap. This is the check that catches a region held up by a handful of items.",
    )
    p.add_argument(
        "--representatives",
        type=int,
        default=DEFAULT_REPRESENTATIVES,
        help=f"Items reported per region, nearest-centroid first (default: "
        f"{DEFAULT_REPRESENTATIVES}). Capped at the region's size.",
    )
    p.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"Cap on Lloyd iterations per clustering (default: {DEFAULT_MAX_ITERATIONS}). "
        "Reached only when the partition is still churning.",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"Centroid movement below which a clustering is called converged "
        f"(default: {DEFAULT_TOLERANCE}).",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_regions)
