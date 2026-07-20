"""``embeddings compare`` — what two models agree is near what, and nothing more.

The rule this verb exists to obey (``docs/design-notes/semantic-measurement.md``
§2): **no output value may assume, imply, or be derived from a shared
coordinate frame.**

Why that rule is not caution
----------------------------
Two embedding models share no basis. They differ in dimensionality, in scale,
in the arbitrary rotation and reflection their training landed on, and in the
shape of the distance distribution they induce. A coordinate from model A and a
coordinate from model B are numbers in different units of different things, and
subtracting them would be arithmetic on incommensurable quantities presented
with the full authority of a number.

Cosine similarity is the specific trap, because it *looks* model-independent:
it is normalized, bounded, and reported the same way everywhere. It is still
not on a common ruler — models differ systematically in how tightly they pack
unrelated text, so ``0.82`` from one model and ``0.82`` from another are not the
same statement about similarity, and "the distance changed by 0.1 between
models" is not a measurement. This module therefore computes cosine similarity
**only inside each model's own frame**, uses it only to *order* that model's own
neighbors, and never emits it. What leaves this module is item ids, ranks,
counts, set-overlap fractions, and rank correlations.

Why relational measures survive
--------------------------------
Ranks and set membership are invariant under exactly the transformations that
differ between models: any rotation, any reflection, any monotone rescaling of
distance. "Which items are this item's ``k`` nearest neighbors" is a question
each model answers in its own frame, and the two answers are directly
comparable because they are drawn from the *same shared item set*. That is why
the shared probe set is a requirement rather than a convenience, and why this
verb refuses (exit 1) when the two atlases have no items in common.

The probe set is built by intersecting the two atlases on record ``id`` **and**
requiring identical ``text`` — assumption 1 is enforced here, not merely
asserted in the output. An id that carries different text in the two atlases is
not the same probe item, so it cannot enter the common index; it is excluded and
counted.

What is measured
-----------------
* **k-NN set overlap (Jaccard)** per probe item, at a stated ``k``. This is the
  measure conditioned on ``k``.
* **Spearman rank correlation** per probe item, over each model's *complete*
  ordering of every other probe item. This one is deliberately **not**
  restricted to the top-``k`` neighbors — see below.

Both are reported as a distribution (mean, min, max, spread, stdev, and the raw
per-item values via :class:`~embeddings.measure.AgreementDistribution`), never
as a bare mean — four items agreeing perfectly plus one disagreeing completely
averages to something reassuring.

Why the rank correlation is not restricted to the top-``k`` union
------------------------------------------------------------------
The obvious-looking design — correlate ranks over the union of the two top-``k``
neighbor sets — is **structurally biased toward disagreement**, and was rejected
here after measuring it. Model A's own ``k`` picks occupy ranks 0..k-1 in A's
ordering but land at effectively random (mostly high) ranks in B's, and model
B's picks do the mirror image. So the union is, by construction, a set on which
the two rank vectors run in opposite directions. Two *independent* random
orderings score about **-0.63** under that domain rather than 0.0, which means
"strongly negative" would have read as real disagreement while actually being an
artifact of the domain, and the chance baseline for it is not analytic.

Over each model's complete ordering the same independent-orderings check scores
**0.002** — the chance value genuinely is 0.0, so the number is interpretable on
its own. The cost is that this measure is not conditioned on ``k``, which is
stated in the output rather than glossed: ``k`` conditions the Jaccard, not the
Spearman.

The chance baseline
--------------------
Expected random overlap scales with ``k/N``: a small probe set with a large
``k`` yields high agreement that means nothing. So every run reports, next to
the observed value, the Jaccard two *independent uniform* choosers would score
at the same ``k`` and ``N`` — see :func:`chance_baseline` for the derivation and
its one approximation. Without it a Jaccard number is uninterpretable.

What ``compare`` does not say
------------------------------
It never ranks the two models. Agreement is symmetric and carries no ground
truth: neither model is shown to be better, more accurate, or more semantically
correct by agreeing or disagreeing with the other. Where they disagree, that is
a located, re-runnable finding an operator can inspect — which is the point.
:data:`NOT_CLAIMED` is emitted verbatim in every payload so a reader cannot
arrive at the number without the disclaimer.

No gateway
-----------
``compare`` reads two already-surveyed atlases and opens no socket;
``tests/test_compare.py`` proves it by running the real entry point inside
:func:`tests._fakes.blocked_sockets`. It is also fully deterministic: the same
two atlases at the same ``k`` produce byte-identical output.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from embeddings._numeric import MIN_NORM
from embeddings.atlas.io import Atlas
from embeddings.atlas.schema import AtlasRecord
from embeddings.cli._commands._atlas_arg import load_atlas
from embeddings.cli._errors import EXIT_USER_ERROR, CliError
from embeddings.cli._output import emit_result
from embeddings.cli._provenance import (
    REPRODUCED_BY_KEY,
    reproduce_line,
    reproducing_command,
)
from embeddings.measure import AgreementDistribution

#: Names stamped on every distribution, so a reader never has to guess which
#: statistic a number is.
MEASURE_JACCARD = "knn_jaccard"
MEASURE_SPEARMAN = "spearman_rank_correlation"

#: Default neighborhood size. Agreement typically falls as ``k`` grows, so this
#: is a slice, not a summary — which is why ``k`` is stamped on every output.
DEFAULT_TOP_K = 10
#: Default seed recorded for provenance. See :data:`SEED_NOTE`.
DEFAULT_SEED = 0

#: Above this expected-Jaccard value the chance baseline dominates, and the
#: observed number says more about ``k`` and ``N`` than about the two models.
CHANCE_DOMINATED_THRESHOLD = 0.25

#: How many worst-agreeing probe items text mode lists as located findings.
_TEXT_DISAGREEMENT_EXAMPLES = 5

#: Guard against dividing a zero vector by its zero norm.
#: Shared with the other cosine call sites so all four agree on what counts
#: as a degenerate magnitude. See embeddings/_numeric.py for the derivation.
_EPSILON = MIN_NORM

#: Digits kept when serializing, matching ``embeddings.measure.stability``.
_ROUND_DIGITS = 6

#: The frame disclosure emitted in every payload. Names cosine explicitly
#: because that is the value a reader is most likely to assume is portable.
NO_SHARED_FRAME_STATEMENT = (
    "These two models share no coordinate frame: no common basis, no common scale, and "
    "no common distance distribution. Nothing in this output is a coordinate, a distance, "
    "or a cosine, and no value here is comparable across the two models as a magnitude. "
    "Cosine similarity in particular is NOT on a common ruler between models even though it "
    "is normalized and bounded — it was used only to order each model's own neighbor list "
    "inside its own frame, and no cosine value is reported. Every number below is a "
    "set-overlap fraction or a rank correlation, both invariant under the rotation, "
    "reflection, and monotone rescaling of distance that differ between models."
)

#: What agreement does not establish. Emitted verbatim in every payload.
NOT_CLAIMED: tuple[str, ...] = (
    "compare does not rank the two models. Agreement is symmetric and carries no ground "
    "truth, so neither model is shown to be better, more accurate, or more semantically "
    "correct by agreeing or disagreeing with the other.",
    "A disagreement is a located, re-runnable finding an operator can inspect — not an "
    "error by either model, and not evidence that one of them got the item wrong.",
    "No cross-model distance, cosine, or coordinate is reported, because such a value would "
    "be arithmetic on incommensurable quantities.",
    "Agreement is not accuracy. Two models can agree closely and both be wrong about the "
    "domain, which this comparison has no way to detect.",
)

#: Recorded because §3 requires a seed on every measurement output; stated
#: honestly because v1 consumes no randomness.
SEED_NOTE = (
    "seed is recorded for provenance; the v1 comparison is fully deterministic and "
    "consumes no randomness, so changing the seed cannot change these numbers"
)

#: The exact remediation for "these two atlases share no probe items".
NO_SHARED_PROBE_SET_HINT = (
    "compare needs a shared probe set: both atlases must contain the same items, "
    "identified by the same record ids and carrying identical text. Survey the same "
    "corpus with each model (embeddings survey <corpus> --apply --atlas <path>), then "
    "compare the two resulting atlases."
)


def _user_error(message: str, remediation: str) -> CliError:
    return CliError(code=EXIT_USER_ERROR, message=message, remediation=remediation)


def _round(value: float) -> float:
    return float(round(float(value), _ROUND_DIGITS))


# --------------------------------------------------------------------------
# The relational primitives
# --------------------------------------------------------------------------


def jaccard(left: Iterable[int], right: Iterable[int]) -> float:
    """Set-overlap fraction ``|A n B| / |A u B|`` — symmetric, frame-free.

    Set membership survives every transformation that differs between two
    models, which is the entire reason this is the headline measure. Two empty
    sets are treated as identical (``1.0``); the CLI path never produces them,
    since ``k >= 1``.
    """
    a = set(left)
    b = set(right)
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def _pearson(left: np.ndarray, right: np.ndarray) -> float | None:
    centered_left = left - left.mean()
    centered_right = right - right.mean()
    denominator = float(np.sqrt((centered_left**2).sum() * (centered_right**2).sum()))
    if denominator <= _EPSILON:
        return None
    return float((centered_left * centered_right).sum() / denominator)


def spearman(
    left: Sequence[float] | np.ndarray, right: Sequence[float] | np.ndarray
) -> float | None:
    """Spearman rank correlation between two orderings of the same items.

    Implemented directly on numpy (this project does not depend on scipy): both
    inputs are converted to their within-sample ranks and correlated with
    Pearson. Ties are broken by position, which is deterministic and adequate
    here because the inputs are already distinct rank positions drawn from each
    model's own complete ordering.

    The property that makes this the right cross-model measure: it is invariant
    under any monotone rescaling of its inputs, which is exactly the freedom two
    models have relative to each other. ``None`` when fewer than two points are
    supplied, or when a rank vector is constant — an undefined correlation is
    reported as absent, never as ``0.0``, which a caller could mistake for
    measured independence.
    """
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    if a.size != b.size:
        raise _user_error(
            f"cannot correlate orderings of different lengths ({a.size} vs {b.size})",
            "both orderings must cover the same items, in the same order",
        )
    if a.size < 2:
        return None
    rank_a = np.argsort(np.argsort(a, kind="stable"), kind="stable").astype(np.float64)
    rank_b = np.argsort(np.argsort(b, kind="stable"), kind="stable").astype(np.float64)
    return _pearson(rank_a, rank_b)


def chance_baseline(*, probe_items: int, k: int) -> dict[str, object]:
    """Expected k-NN agreement between two *independent uniform* choosers.

    Without this, a Jaccard number is uninterpretable: a small probe set with a
    large ``k`` forces high overlap no matter what the models represent. At the
    extreme ``k == N - 1`` both models must pick every other item and agreement
    is a forced ``1.0``, which this returns exactly.

    Derivation. Each model picks ``k`` of the ``N - 1`` candidates other than
    the item itself. Under independent uniform selection the expected
    intersection is ``k * k / (N - 1)`` (each of A's ``k`` picks is in B's set
    with probability ``k / (N - 1)``), and the expected union is
    ``2k - E|A n B|``.

    The one approximation, stated rather than hidden: the reported value is the
    *ratio of expectations*, ``E|A n B| / E|A u B|``, not the expectation of the
    ratio. They differ slightly because Jaccard is non-linear. The ratio of
    expectations is the standard, conservative reading and is what the
    ``method`` field in the returned payload says it is.

    The expected Spearman correlation of two independent *complete* orderings is
    exactly ``0.0``, so that measure needs no such approximation — but only
    because it is computed over each model's full ordering. The seemingly
    natural top-``k``-union variant has a chance value near ``-0.63``, not
    ``0.0``, which is why this module does not use it (see the module docstring).
    """
    pool = probe_items - 1
    expected_intersection = (k * k) / pool
    expected_union = (2 * k) - expected_intersection
    expected_jaccard = expected_intersection / expected_union
    return {
        "k": k,
        "probe_items": probe_items,
        "candidate_pool": pool,
        "expected_knn_jaccard": _round(expected_jaccard),
        "expected_spearman_rank_correlation": 0.0,
        "method": (
            "two independent uniform choosers of k items from the N-1 candidates other than "
            "the item itself: E|A n B| = k*k/(N-1), E|A u B| = 2k - E|A n B|, reported as the "
            "ratio of expectations (not the expectation of the ratio — Jaccard is non-linear, "
            "so the two differ slightly). The Spearman baseline is exactly 0.0 because that "
            "measure runs over each model's complete ordering; it would NOT be 0.0 over the "
            "top-k union, which is structurally anti-correlated and is therefore not used."
        ),
    }


# --------------------------------------------------------------------------
# The shared probe set
# --------------------------------------------------------------------------


def _first_record_by_id(atlas: Atlas) -> dict[str, AtlasRecord]:
    """Map id -> first record carrying it, so a duplicated id resolves stably."""
    by_id: dict[str, AtlasRecord] = {}
    for record in atlas.records:
        by_id.setdefault(record.id, record)
    return by_id


def _build_probe_set(
    atlas_a: Atlas, atlas_b: Atlas
) -> tuple[list[str], np.ndarray, np.ndarray, int]:
    """Intersect the two atlases into a common index.

    Returns ``(ids, rows_a, rows_b, text_mismatches)``. Ids are sorted, so the
    probe set — and therefore every number derived from it — is identical no
    matter which atlas was passed as A.
    """
    records_a = _first_record_by_id(atlas_a)
    records_b = _first_record_by_id(atlas_b)

    ids: list[str] = []
    rows_a: list[int] = []
    rows_b: list[int] = []
    mismatches = 0
    for item_id in sorted(set(records_a) & set(records_b)):
        record_a = records_a[item_id]
        record_b = records_b[item_id]
        # Assumption 1 enforced: same id + different text is not the same probe
        # item, and a metric defined over a broken common index measures nothing.
        if record_a.text != record_b.text:
            mismatches += 1
            continue
        ids.append(item_id)
        rows_a.append(record_a.vector_index)
        rows_b.append(record_b.vector_index)

    return ids, np.asarray(rows_a, dtype=np.int64), np.asarray(rows_b, dtype=np.int64), mismatches


def _neighbor_order(vectors: np.ndarray) -> np.ndarray:
    """Each row's ordering of every other probe item, most similar first.

    Cosine similarity is computed here and used *only* here: it orders one
    model's own neighbors inside that model's own frame. No similarity value
    escapes this function — only the ordering does, and an ordering is exactly
    what survives the differences between models.
    """
    data = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(data, axis=1, keepdims=True)
    unit = data / np.maximum(norms, _EPSILON)
    within_frame = unit @ unit.T
    np.fill_diagonal(within_frame, -np.inf)  # an item is never its own neighbor
    # Stable sort => ties broken by index, so the ordering is deterministic.
    return np.argsort(-within_frame, axis=1, kind="stable")


def _rank_positions(order: np.ndarray) -> np.ndarray:
    """``positions[i, j]`` = where item ``j`` sits in item ``i``'s ordering."""
    count = order.shape[0]
    positions = np.empty_like(order)
    np.put_along_axis(positions, order, np.tile(np.arange(count), (count, 1)), axis=1)
    return positions


# --------------------------------------------------------------------------
# Stated assumptions (design note §2)
# --------------------------------------------------------------------------


def _assumptions(
    *, probe_items: int, mismatches: int, k: int, baseline: dict[str, object]
) -> list[dict[str, str]]:
    """The five §2 assumptions, each with what this run can and cannot verify.

    Recording *status* alongside the assumption is deliberate: two of the five
    are enforced by construction here, two are reported as values, and one is
    simply not checkable by this tool. Flattening them into uniform boilerplate
    would hide which is which.
    """
    return [
        {
            "id": "common-index",
            "assumption": (
                "Both models embed the same probe items, identified identically. The metric "
                "is defined over a common index; without one there is nothing to intersect."
            ),
            "status": (
                f"enforced: {probe_items} item(s) matched on record id with identical text; "
                f"{mismatches} shared id(s) excluded for carrying different text in the two "
                "atlases"
            ),
        },
        {
            "id": "representative-probe-set",
            "assumption": (
                "The probe set is representative of the region being compared. Jaccard over "
                "an unrepresentative sample measures the sample, not the models."
            ),
            "status": (
                "not verifiable by this tool: the probe set is whatever both atlases happen "
                "to contain. Judge it against the corpora recorded in each atlas manifest's "
                "params before generalizing this result beyond them."
            ),
        },
        {
            "id": "chance-baseline",
            "assumption": (
                "The probe set is large enough that overlap at k is not dominated by chance. "
                "Expected random overlap scales with k/N, so small N with large k yields high "
                "agreement that means nothing."
            ),
            "status": (
                f"reported: expected Jaccard {baseline['expected_knn_jaccard']} for two "
                f"independent uniform choosers at k={k}, N={probe_items}. Read every observed "
                "value against it."
            ),
        },
        {
            "id": "conditioned-on-k",
            "assumption": (
                "Results are conditioned on k. Agreement typically falls as k grows, so a "
                "single k is a slice, not a summary."
            ),
            "status": (
                f"k={k}, stamped on every output. It conditions the k-NN Jaccard only: the "
                "Spearman rank correlation runs over each model's complete ordering of the "
                "probe set and is NOT conditioned on k, because the top-k-union variant is "
                "structurally anti-correlated (chance is about -0.63 there, not 0.0). Re-run "
                "at other k before describing these models as agreeing or disagreeing in "
                "general."
            ),
        },
        {
            "id": "model-preprocessing",
            "assumption": (
                "Each model's own preprocessing was applied as that model intends — "
                "normalization, prompt or instruction prefixes, truncation. Comparing model A "
                "used correctly against model B used incorrectly measures the harness."
            ),
            "status": (
                "not verifiable by this tool: each atlas was produced by its own survey run, "
                "and this verb only reads the result. Check the endpoint and params recorded "
                "in each atlas manifest."
            ),
        },
    ]


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _render(payload: dict[str, object]) -> str:
    atlas_a = payload["atlas_a"]
    atlas_b = payload["atlas_b"]
    probe = payload["probe_set"]
    measures = payload["measures"]
    baseline = payload["chance_baseline"]
    assert isinstance(atlas_a, dict) and isinstance(atlas_b, dict)
    assert isinstance(probe, dict) and isinstance(measures, dict) and isinstance(baseline, dict)

    jaccard_stats = measures[MEASURE_JACCARD]
    rank_stats = measures[MEASURE_SPEARMAN]
    assert isinstance(jaccard_stats, dict)

    lines = [
        f"# compare (relational only — Jaccard conditioned on k={payload['k']})",
        "",
        f"model A:       {atlas_a['model']}  [{atlas_a['path']}]",
        f"model B:       {atlas_b['model']}  [{atlas_b['path']}]",
        f"probe set:     {probe['size']} shared item(s)"
        + (
            f", {probe['excluded_id_text_mismatch']} excluded (same id, different text)"
            if probe["excluded_id_text_mismatch"]
            else ""
        ),
        f"k:             {payload['k']}",
        f"seed:          {payload['seed']}",
        "",
        "## agreement",
        "",
        f"k-NN Jaccard:  mean {jaccard_stats['mean']:.3f}  min {jaccard_stats['min']:.3f}  "
        f"max {jaccard_stats['max']:.3f}  spread {jaccard_stats['spread']:.3f}   "
        f"[top-{payload['k']} sets]",
    ]
    if isinstance(rank_stats, dict):
        lines.append(
            f"Spearman rank: mean {rank_stats['mean']:.3f}  min {rank_stats['min']:.3f}  "
            f"max {rank_stats['max']:.3f}  spread {rank_stats['spread']:.3f}   "
            f"[full ordering of all {probe['size']} probe items — NOT conditioned on k]"
        )
    else:
        lines.append(
            "Spearman rank: not computable — a probe set of "
            f"{probe['size']} leaves fewer than 2 candidates to rank"
        )

    lines += [
        "",
        f"chance baseline at k={payload['k']}, N={probe['size']}: expected Jaccard "
        f"{baseline['expected_knn_jaccard']:.3f}, expected Spearman 0.000 "
        f"(observed is {baseline['observed_above_chance']:+.3f} vs chance)",
    ]

    per_item = payload["per_item"]
    assert isinstance(per_item, list)
    worst = sorted(per_item, key=lambda entry: (entry["jaccard"], entry["id"]))
    worst = worst[:_TEXT_DISAGREEMENT_EXAMPLES]
    if worst:
        lines += ["", "## where they disagree most (located, re-runnable)", ""]
        for entry in worst:
            lines.append(
                f"- {entry['id']}: Jaccard {entry['jaccard']:.3f} "
                f"({entry['shared_neighbors']}/{entry['union_size']} shared); "
                f"only A: {', '.join(entry['only_a']) or '-'}; "
                f"only B: {', '.join(entry['only_b']) or '-'}"
            )

    lines += ["", "## assumptions", ""]
    for entry in payload["assumptions"]:  # type: ignore[union-attr]
        lines.append(f"- [{entry['id']}] {entry['assumption']}")
        lines.append(f"  status: {entry['status']}")

    lines += ["", "## what this does not say", ""]
    lines += [f"- {claim}" for claim in payload["not_claimed"]]  # type: ignore[union-attr]

    frame = payload["coordinate_frame"]
    assert isinstance(frame, dict)
    lines += [
        "",
        "## coordinate frame",
        "",
        f"shared: no (model A is {frame['dimensions_a']}-dimensional, "
        f"model B is {frame['dimensions_b']}-dimensional)",
        frame["statement"],
    ]

    notes = payload["notes"]
    if isinstance(notes, list) and notes:
        lines += ["", "## notes", ""]
        lines += [f"- {note}" for note in notes]

    lines += ["", reproduce_line(payload[REPRODUCED_BY_KEY])]
    return "\n".join(lines)


# --------------------------------------------------------------------------
# The handler
# --------------------------------------------------------------------------


def _load(flag: str, raw_path: str) -> Atlas:
    """Load one side, reusing the shared reader-verb entry path verbatim.

    The message is prefixed with the flag so a caller knows *which* atlas was
    missing, but the code and the remediation hint pass through unchanged —
    every reader verb in this CLI reports the same fix for the same problem.
    """
    try:
        return load_atlas(argparse.Namespace(atlas=raw_path))
    except CliError as err:
        raise CliError(
            code=err.code,
            message=f"{flag}: {err.message}",
            remediation=err.remediation,
        ) from err


def _validate_top_k(k: int) -> None:
    if k < 1:
        raise _user_error(
            f"--top-k must be a positive integer, got {k}",
            f"pass --top-k 1 or higher (default: {DEFAULT_TOP_K})",
        )


def _validate_probe_set(
    *, probe_items: int, mismatches: int, k: int, path_a: Path, path_b: Path
) -> None:
    """The three ways a probe set can be unusable for a k-NN agreement measure.

    Split from :func:`cmd_compare` because these are the same kind of check —
    "is there enough shared, well-formed evidence to measure agreement at all"
    — applied in sequence, and grouping them keeps that intent in one place.
    """
    if probe_items == 0:
        raise _user_error(
            f"the two atlases share no probe items ({path_a} and {path_b} have no record ids "
            f"in common carrying identical text; {mismatches} id(s) matched but differ in text)",
            NO_SHARED_PROBE_SET_HINT,
        )
    if probe_items < 2:
        raise _user_error(
            f"the shared probe set has {probe_items} item(s); neighborhood agreement needs at "
            "least 2 so an item has something to be near",
            NO_SHARED_PROBE_SET_HINT,
        )
    if k > probe_items - 1:
        raise _user_error(
            f"--top-k {k} exceeds the {probe_items - 1} candidate(s) available in a shared "
            f"probe set of {probe_items} item(s)",
            f"pass --top-k {probe_items - 1} or lower, or compare atlases over a larger "
            "shared corpus",
        )


def _agreement_per_item(
    *,
    ids: Sequence[str],
    order_a: np.ndarray,
    order_b: np.ndarray,
    positions_a: np.ndarray,
    positions_b: np.ndarray,
    k: int,
) -> tuple[list[dict[str, object]], list[float], list[float]]:
    """Per-probe-item Jaccard and Spearman agreement — the located, re-runnable findings.

    Returns ``(per_item, jaccard_values, spearman_values)``: the located
    per-item records for the payload, plus the two raw value lists the
    distribution summaries in :func:`cmd_compare` are built from.
    """
    probe_items = len(ids)
    all_indices = np.arange(probe_items, dtype=np.int64)
    per_item: list[dict[str, object]] = []
    jaccard_values: list[float] = []
    spearman_values: list[float] = []

    for index, item_id in enumerate(ids):
        top_a = order_a[index, :k]
        top_b = order_b[index, :k]
        set_a = {int(value) for value in top_a}
        set_b = {int(value) for value in top_b}
        union = sorted(set_a | set_b)
        overlap = jaccard(set_a, set_b)
        jaccard_values.append(overlap)

        # Ranks come from each model's *complete* ordering of every other probe
        # item, deliberately NOT restricted to the top-k union: that restriction
        # is structurally anti-correlated (independent orderings score about
        # -0.63 on it rather than 0.0), which would make the number an artifact
        # of the domain. See the module docstring.
        others = all_indices[all_indices != index]
        correlation = spearman(positions_a[index, others], positions_b[index, others])
        if correlation is not None:
            spearman_values.append(correlation)

        per_item.append(
            {
                "id": item_id,
                "jaccard": _round(overlap),
                "shared_neighbors": len(set_a & set_b),
                "union_size": len(union),
                "spearman": None if correlation is None else _round(correlation),
                "only_a": [ids[value] for value in sorted(set_a - set_b)],
                "only_b": [ids[value] for value in sorted(set_b - set_a)],
            }
        )

    return per_item, jaccard_values, spearman_values


def _build_notes(
    *,
    atlas_a: Atlas,
    same_model: bool,
    path_a: Path,
    path_b: Path,
    k: int,
    probe_items: int,
    expected_jaccard: float,
    rank_distribution: AgreementDistribution | None,
    mismatches: int,
) -> list[str]:
    """The caveats worth surfacing next to this run's numbers, in a fixed order.

    Each condition below is independent of the others, so this is a flat
    sequence of checks rather than nested logic — extracted purely to keep
    that sequence (and the seed note it always starts with) out of the
    handler's own control flow.
    """
    notes: list[str] = [SEED_NOTE]
    if same_model:
        notes.append(
            "both atlases were written by the same model "
            f"({atlas_a.manifest.model}): this is a self-consistency sanity check, not a "
            "cross-model comparison, and agreement near 1.0 is the expected result rather "
            "than a finding"
        )
    if path_a == path_b:
        notes.append("both --atlas-a and --atlas-b point at the same atlas directory")
    if expected_jaccard > CHANCE_DOMINATED_THRESHOLD:
        notes.append(
            f"chance-dominated: at k={k} over {probe_items} probe item(s) two independent "
            f"uniform choosers already score {expected_jaccard:.3f}, so the observed overlap "
            "says more about k and N than about the two models. Lower k, or compare over a "
            "larger shared probe set"
        )
    if rank_distribution is None:
        notes.append(
            f"no rank correlation computed: a probe set of {probe_items} leaves fewer than 2 "
            "other items to rank, which leaves Spearman undefined"
        )
    if mismatches:
        notes.append(
            f"{mismatches} record id(s) present in both atlases were excluded because their "
            "text differs — the same id over different text is not the same probe item"
        )
    return notes


def cmd_compare(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    k = int(args.top_k)
    seed = int(args.seed)
    _validate_top_k(k)

    path_a = Path(args.atlas_a)
    path_b = Path(args.atlas_b)
    atlas_a = _load("--atlas-a", str(path_a))
    atlas_b = _load("--atlas-b", str(path_b))

    ids, rows_a, rows_b, mismatches = _build_probe_set(atlas_a, atlas_b)
    probe_items = len(ids)
    _validate_probe_set(
        probe_items=probe_items, mismatches=mismatches, k=k, path_a=path_a, path_b=path_b
    )

    order_a = _neighbor_order(atlas_a.vectors[rows_a])
    order_b = _neighbor_order(atlas_b.vectors[rows_b])
    positions_a = _rank_positions(order_a)
    positions_b = _rank_positions(order_b)

    per_item, jaccard_values, spearman_values = _agreement_per_item(
        ids=ids,
        order_a=order_a,
        order_b=order_b,
        positions_a=positions_a,
        positions_b=positions_b,
        k=k,
    )

    jaccard_distribution = AgreementDistribution.from_values(MEASURE_JACCARD, jaccard_values)
    rank_distribution = (
        AgreementDistribution.from_values(MEASURE_SPEARMAN, spearman_values)
        if spearman_values
        else None
    )

    baseline = chance_baseline(probe_items=probe_items, k=k)
    expected_jaccard = float(baseline["expected_knn_jaccard"])  # type: ignore[arg-type]
    baseline["observed_above_chance"] = _round(jaccard_distribution.mean - expected_jaccard)

    same_model = atlas_a.manifest.model == atlas_b.manifest.model
    notes = _build_notes(
        atlas_a=atlas_a,
        same_model=same_model,
        path_a=path_a,
        path_b=path_b,
        k=k,
        probe_items=probe_items,
        expected_jaccard=expected_jaccard,
        rank_distribution=rank_distribution,
        mismatches=mismatches,
    )

    params: dict[str, object] = {
        "k": k,
        "seed": seed,
        "atlas_a": str(path_a),
        "atlas_b": str(path_b),
        "model_a": atlas_a.manifest.model,
        "model_b": atlas_b.manifest.model,
        "probe_items": probe_items,
        "excluded_id_text_mismatch": mismatches,
        "measures": [MEASURE_JACCARD, MEASURE_SPEARMAN],
        "neighbor_ordering": (
            "cosine similarity computed within each model's own frame, used only to order "
            "that model's own neighbor list; never compared across models and never emitted"
        ),
        "rank_correlation_domain": (
            "each model's complete ordering of every other probe item — NOT the top-k union, "
            "which is structurally anti-correlated (chance is about -0.63 there rather than "
            "0.0). k conditions the Jaccard only"
        ),
    }

    payload: dict[str, object] = {
        "command": "compare",
        "atlas_a": {
            "path": str(path_a),
            "model": atlas_a.manifest.model,
            "endpoint": atlas_a.manifest.endpoint,
            "seed": atlas_a.manifest.seed,
            "items": len(atlas_a.records),
        },
        "atlas_b": {
            "path": str(path_b),
            "model": atlas_b.manifest.model,
            "endpoint": atlas_b.manifest.endpoint,
            "seed": atlas_b.manifest.seed,
            "items": len(atlas_b.records),
        },
        "same_model": same_model,
        "probe_set": {
            "size": probe_items,
            "excluded_id_text_mismatch": mismatches,
            "source": "intersection of both atlases on record id, requiring identical text",
        },
        "k": k,
        "seed": seed,
        "params": params,
        "measures": {
            MEASURE_JACCARD: jaccard_distribution.to_dict(),
            MEASURE_SPEARMAN: (None if rank_distribution is None else rank_distribution.to_dict()),
        },
        "chance_baseline": baseline,
        "per_item": per_item,
        "assumptions": _assumptions(
            probe_items=probe_items, mismatches=mismatches, k=k, baseline=baseline
        ),
        "not_claimed": list(NOT_CLAIMED),
        "coordinate_frame": {
            "shared": False,
            "dimensions_a": int(atlas_a.vectors.shape[1]) if atlas_a.vectors.ndim == 2 else 0,
            "dimensions_b": int(atlas_b.vectors.shape[1]) if atlas_b.vectors.ndim == 2 else 0,
            "statement": NO_SHARED_FRAME_STATEMENT,
        },
        "notes": notes,
        REPRODUCED_BY_KEY: reproducing_command(),
    }

    emit_result(payload if json_mode else _render(payload), json_mode=json_mode)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "compare",
        help="Compare two surveyed models by k-NN overlap and rank agreement on a shared "
        "probe set (never by coordinates or cosines).",
        description=(
            "Measure how much two models agree about what is near what, on the probe items "
            "both of their atlases contain. Reports k-NN set overlap (Jaccard) and Spearman "
            "rank correlation, both with a chance baseline for the same k and N. Emits no "
            "coordinate, distance, or cosine: the two models share no frame, so such a value "
            "would not be comparable. Never ranks the two models."
        ),
    )
    p.add_argument(
        "--atlas-a",
        required=True,
        metavar="<path>",
        help="Path to the first model's atlas directory, as written by "
        "`embeddings survey <corpus> --apply --atlas <path>`.",
    )
    p.add_argument(
        "--atlas-b",
        required=True,
        metavar="<path>",
        help="Path to the second model's atlas directory. Must share probe items with "
        "--atlas-a: the same record ids carrying identical text.",
    )
    p.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        metavar="<n>",
        help=f"Neighborhood size the comparison is conditioned on (default: {DEFAULT_TOP_K}). "
        "Agreement typically falls as k grows, so one k is a slice, not a summary; k is "
        "stamped on every output value and must not exceed (probe items - 1).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        metavar="<n>",
        help=f"Seed recorded in the output for provenance (default: {DEFAULT_SEED}). The "
        "comparison itself is fully deterministic and consumes no randomness.",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_compare)
