"""``embeddings directions`` — propose candidate semantic directions and test whether they hold.

A *direction* here is a vector in the embedding space, derived from the
differences between paired items. It is emphatically **not** a coordinate index:
nothing in this module names, ranks, or interprets an individual dimension. Per
``docs/design-notes/semantic-measurement.md``, per-dimension interpretation is
unstable across seeds, training runs, and model versions, and it is misleading
precisely because it reads well.

Why the held-out split is the whole verb
----------------------------------------
Fit a direction to a handful of item pairs and it will always look like
something: the top principal component of *any* small set of difference vectors
is a clean, confident-looking axis, whether or not the pairs share a concept.
The claim only carries information if the direction transfers to pairs it was
not fitted on. So ``discover`` splits its evidence:

* **derive** candidate directions from the *training* pairs only — the singular
  vectors of the matrix of training difference vectors, oriented so each points
  from ``a`` toward ``b`` on average;
* **measure** each candidate against the *held-out* pairs it never saw — mean
  cosine between each held-out difference vector and the candidate, plus the
  fraction of held-out pairs whose difference points the same way at all;
* **report both**, alongside the training-side alignment, so the overfit case is
  visible rather than inferred: a candidate with high training alignment and
  near-zero held-out alignment fitted noise.

A candidate that fails to generalize is reported as a **negative result** —
listed, marked ``DOES NOT GENERALIZE``, and routed into ``uncertain[]``. It is
never dropped: suppressing it would hide the fact that the pair set does not
encode a transferable direction, which is itself a finding.

Where the pairs come from
-------------------------
Two sources, one code path afterwards:

* ``--pairs <path>`` — a file of ``(a, b)`` **Atlas item ids**, one pair per
  line, as either a JSON object (``{"a": "id1", "b": "id2"}``) or a bare
  ``id1,id2`` CSV row. Ids, not texts: an id resolves against the Atlas's own
  records, so ``discover`` needs no embedding gateway. Text inputs would require
  embedding at read time, which is exactly the dependency reader verbs are
  supposed to avoid (design note §4); ``tests/test_directions.py`` proves the
  gateway-free property with :func:`tests._fakes.blocked_sockets`.
* no ``--pairs`` — unsupervised: sample ``--sample-pairs`` random distinct item
  pairs from the Atlas under ``--seed`` and look for principal directions among
  their differences. Useful for asking "does *any* recurring direction exist
  here?" without having a hypothesis first; the held-out test applies unchanged,
  and typically answers "no", which is the honest answer.

Stability (design note §3)
--------------------------
Every candidate also carries a stability measurement routed through
:func:`embeddings.measure.measure_stability`. The structure measured is the
**bipartition of the surveyed items the direction induces** (which side of the
median projection each item falls on), compared with the Adjusted Rand Index
across seeds and under bootstrap resampling. Each rerun re-splits the pair set
under a derived seed and re-derives the candidate, so the question the score
answers is: *would a different split of these pairs have carved the space the
same way?* A single run with no resampling is labeled ``stability unverified``
with ``score: null`` — never a zero a caller could compare against a threshold.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from embeddings._numeric import safe_norms
from embeddings.atlas.io import Atlas
from embeddings.cli._commands._atlas_arg import add_atlas_argument, load_atlas, resolve_atlas_path
from embeddings.cli._commands.overview import emit_overview
from embeddings.cli._errors import EXIT_USER_ERROR, CliError
from embeddings.cli._output import emit_result
from embeddings.cli._provenance import (
    REPRODUCED_BY_KEY,
    reproduce_line,
    reproducing_command,
)
from embeddings.measure import StabilityResult, measure_stability
from embeddings.measure.stability import DEFAULT_RESAMPLES, DEFAULT_RUNS

#: Name of the generalization statistic, stamped on output so a reader never has
#: to guess what a number is. The design note allows "neighborhood or rank
#: agreement" for direction claims; cosine alignment of held-out difference
#: vectors is the direction-shaped member of that family.
MEASURE_HELDOUT_COSINE = "heldout_mean_cosine"

#: Defaults. ``--holdout`` is a *fraction* of the pair set; the split always
#: leaves at least one pair on each side, or the run fails rather than pretending
#: it measured generalization.
DEFAULT_HOLDOUT = 0.3
DEFAULT_TOP = 3
DEFAULT_SAMPLE_PAIRS = 64
DEFAULT_SEED = 0

#: Thresholds a candidate must clear on the held-out pairs to be called
#: generalizing. Reported verbatim in the payload so a reader can apply their own
#: bar to the raw numbers instead of inheriting these. Both are needed: a high
#: mean cosine carried by a few strong pairs while the rest point the other way
#: is not a direction, and unanimous-but-weak agreement is not one either. The
#: chance baselines are 0.0 and 0.5 respectively, for random vectors.
MIN_HELDOUT_MEAN_COSINE = 0.2
MIN_HELDOUT_AGREEMENT_RATE = 0.75

_ROUND_DIGITS = 6

_FRAMING_LIMITATION = (
    "a candidate direction is a vector in the embedding space, not a coordinate: this "
    "output never names or interprets an individual dimension"
)
_PAIR_SET_LIMITATION = (
    "generalization is measured only against the held-out pairs listed here; a direction "
    "that transfers across this split may still not transfer to a different pair set, and "
    "the split is conditioned on --seed"
)


def _user_error(message: str, remediation: str) -> CliError:
    return CliError(code=EXIT_USER_ERROR, message=message, remediation=remediation)


def _round(value: float) -> float:
    return float(round(float(value), _ROUND_DIGITS))


# --------------------------------------------------------------------------
# Pair sourcing — ids only, so this verb never needs the gateway
# --------------------------------------------------------------------------


def _parse_pairs_file(path: Path) -> list[tuple[str, str]]:
    """Read ``(a, b)`` item-id pairs from a JSONL or two-column CSV file.

    Blank lines and ``#`` comments are skipped. Both line shapes are accepted on
    the same file, decided per line, because the two spellings cost nothing to
    support and a caller generating one of them should not have to find out which
    this verb wanted.
    """
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError as err:
        raise _user_error(
            f"no pairs file at {path}",
            "pass --pairs <path> pointing at a readable file, or omit --pairs to sample "
            "candidate pairs from the atlas instead",
        ) from err
    except OSError as err:
        raise _user_error(
            f"cannot read the pairs file at {path}: {err}",
            "check the path and its permissions",
        ) from err

    pairs: list[tuple[str, str]] = []
    for number, line in enumerate(raw.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.startswith("{"):
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as err:
                raise _user_error(
                    f"{path} line {number} is not valid JSON: {err}",
                    'each line must be {"a": "<item id>", "b": "<item id>"} or a bare '
                    "'<item id>,<item id>' row",
                ) from err
            if not isinstance(payload, dict) or "a" not in payload or "b" not in payload:
                raise _user_error(
                    f"{path} line {number} has no 'a'/'b' keys",
                    'each JSON line must carry both ids, e.g. {"a": "item-1", "b": "item-2"}',
                )
            pairs.append((str(payload["a"]), str(payload["b"])))
            continue
        fields = [field.strip() for field in stripped.split(",")]
        if len(fields) != 2 or not all(fields):
            raise _user_error(
                f"{path} line {number} is not a two-column '<item id>,<item id>' row",
                'use two comma-separated item ids per line, or JSON objects with "a"/"b"',
            )
        pairs.append((fields[0], fields[1]))
    return pairs


def _sample_pairs(records: Sequence[object], count: int, seed: int) -> list[tuple[str, str]]:
    """Sample ``count`` distinct ordered item-id pairs from the Atlas under ``seed``."""
    ids = [str(getattr(record, "id")) for record in records]
    if len(ids) < 2:
        raise _user_error(
            f"cannot sample item pairs from an atlas holding {len(ids)} item(s)",
            "survey a corpus with at least two items, or pass --pairs with explicit ids",
        )
    rng = np.random.default_rng(seed)
    seen: set[tuple[str, str]] = set()
    ordered: list[tuple[str, str]] = []
    # Bounded attempts: with few items the distinct-pair space runs out, and the
    # honest outcome is "fewer pairs than asked for", not a spin.
    for _ in range(count * 20):
        if len(ordered) >= count:
            break
        left, right = rng.choice(len(ids), size=2, replace=False)
        pair = (ids[int(left)], ids[int(right)])
        if pair in seen:
            continue
        seen.add(pair)
        ordered.append(pair)
    return ordered


def _vector_lookup(atlas: Atlas) -> dict[str, int]:
    return {record.id: record.vector_index for record in atlas.records}


def _difference_matrix(
    atlas: Atlas, pairs: Sequence[tuple[str, str]]
) -> np.ndarray:  # (n_pairs, dim)
    """``v(b) - v(a)`` for every pair, resolved through Atlas item ids."""
    index_by_id = _vector_lookup(atlas)
    vectors = np.asarray(atlas.vectors, dtype=np.float64)
    if vectors.ndim != 2 or vectors.shape[0] == 0:
        raise _user_error(
            f"the atlas at {atlas.path} holds no vectors to derive directions from",
            "re-run `embeddings survey <corpus> --apply` to rebuild it",
        )
    rows = []
    for left, right in pairs:
        for item_id in (left, right):
            if item_id not in index_by_id:
                raise _user_error(
                    f"pair item id {item_id!r} is not in the atlas at {atlas.path}",
                    "pairs name atlas item ids (the 'id' field of records.jsonl), not texts; "
                    "check the ids or re-survey the corpus that contains them",
                )
        rows.append(vectors[index_by_id[right]] - vectors[index_by_id[left]])
    return np.asarray(rows, dtype=np.float64)


# --------------------------------------------------------------------------
# Derivation and generalization measurement
# --------------------------------------------------------------------------


def split_pairs(total: int, holdout: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """``(train_indices, holdout_indices)`` — a seeded permutation split of the pair set.

    Always leaves at least one pair on each side; a fraction that would empty
    either side is clamped rather than silently producing a "generalization"
    number measured on nothing.
    """
    rng = np.random.default_rng(seed)
    order = rng.permutation(total)
    size = int(round(total * holdout))
    size = max(1, min(size, total - 1))
    return order[size:], order[:size]


def derive_directions(diffs: np.ndarray, top: int) -> tuple[np.ndarray, np.ndarray]:
    """Candidate directions from difference vectors, plus each one's share of their energy.

    The candidates are the leading right singular vectors of the (uncentered)
    difference matrix — the directions along which the training differences vary
    most. Uncentered on purpose: the *sign* of a difference vector is the
    meaningful part of a paired comparison, and centering would subtract exactly
    the mean offset a direction claim is about.

    Each returned vector is unit length and oriented so the mean training
    difference projects positively onto it, which fixes the arbitrary sign the
    SVD returns and makes "points from ``a`` toward ``b``" a stated property
    rather than a coin flip that would flip a rerun's verdict.
    """
    _, singular, right = np.linalg.svd(diffs, full_matrices=False)
    count = max(1, min(top, right.shape[0]))
    directions = np.array(right[:count], dtype=np.float64)

    total_energy = float(np.sum(singular**2))
    if total_energy > 0.0:
        share = (singular[:count] ** 2) / total_energy
    else:
        share = np.zeros(count, dtype=np.float64)

    mean_projection = (diffs @ directions.T).mean(axis=0)
    flip = np.where(mean_projection < 0.0, -1.0, 1.0)
    return directions * flip[:, None], np.asarray(share, dtype=np.float64)


def alignment(diffs: np.ndarray, direction: np.ndarray) -> np.ndarray:
    """Cosine between each difference vector and ``direction`` (assumed unit length).

    A difference with no usable magnitude (two items sharing a vector, or
    separated only by float noise) scores 0.0 rather than NaN: it carries no
    directional evidence either way, and a NaN would poison the mean it feeds.
    Degeneracy is judged by magnitude via :func:`safe_norms`, not by ``> 0.0`` —
    a difference of norm 1e-300 is noise, not evidence, and dividing by it
    would yield a confident-looking alignment score built from nothing.
    """
    safe, degenerate = safe_norms(diffs)
    return np.where(degenerate, 0.0, (diffs @ direction) / safe)


def _bipartition_fn(diffs: np.ndarray, rank: int, holdout: float, top: int):
    """``(data, seed) -> labels``: the item bipartition this candidate induces.

    Handed to :func:`~embeddings.measure.measure_stability`. Each call re-splits
    the pair set under the supplied seed, re-derives the rank-``rank`` candidate
    from the resulting training pairs, and labels every row of ``data`` by which
    side of the median projection it falls on. Comparing those labelings across
    seeds (and under bootstrap resampling of the items) asks whether a different
    split of the same pairs would have carved the space the same way. ARI is
    invariant to label permutation, so a direction that comes back sign-flipped
    still scores as agreement — which is correct: it is the same partition.
    """

    def cluster_fn(data: np.ndarray, seed: int) -> np.ndarray:
        train_index, _ = split_pairs(diffs.shape[0], holdout, seed)
        directions, _ = derive_directions(diffs[train_index], top)
        if rank >= directions.shape[0]:
            # Fewer candidates survive this split than the base run produced;
            # a constant labeling is the honest "no structure to compare".
            return np.zeros(np.asarray(data).shape[0], dtype=np.int8)
        projection = np.asarray(data, dtype=np.float64) @ directions[rank]
        return (projection >= float(np.median(projection))).astype(np.int8)

    return cluster_fn


def _pair_entries(
    pairs: Sequence[tuple[str, str]], index: np.ndarray, cosines: np.ndarray
) -> list[dict[str, object]]:
    return [
        {"a": pairs[int(position)][0], "b": pairs[int(position)][1], "cosine": _round(cosine)}
        for position, cosine in zip(index, cosines)
    ]


# --------------------------------------------------------------------------
# The discover verb
# --------------------------------------------------------------------------


def _validate(holdout: float, top: int, runs: int, resamples: int, sample_pairs: int) -> None:
    if not 0.0 < holdout < 1.0:
        raise _user_error(
            f"--holdout must be a fraction strictly between 0 and 1, got {holdout}",
            f"pass e.g. --holdout {DEFAULT_HOLDOUT} to hold out {int(DEFAULT_HOLDOUT * 100)}% "
            "of the pairs",
        )
    if top < 1:
        raise _user_error(
            f"--top must be at least 1, got {top}",
            f"pass --top {DEFAULT_TOP} (the default) or higher",
        )
    if runs < 1:
        raise _user_error(
            f"--runs must be at least 1, got {runs}",
            "use --runs 2 or more to obtain seed-rerun evidence; --runs 1 reports "
            "'stability unverified'",
        )
    if resamples < 0:
        raise _user_error(
            f"--resamples must be zero or more, got {resamples}",
            "pass --resamples 0 to skip bootstrap resampling",
        )
    if sample_pairs < 2:
        raise _user_error(
            f"--sample-pairs must be at least 2, got {sample_pairs}",
            "a train/held-out split needs at least two pairs; pass --pairs to supply your own",
        )


def _collect_pairs(atlas: Atlas, args: argparse.Namespace, seed: int) -> tuple[list, str]:
    if getattr(args, "pairs", None):
        path = Path(args.pairs)
        return _parse_pairs_file(path), f"file:{path}"
    return _sample_pairs(atlas.records, int(args.sample_pairs), seed), "sampled"


def _candidate_payload(
    *,
    rank: int,
    direction: np.ndarray,
    share: float,
    pairs: Sequence[tuple[str, str]],
    train_index: np.ndarray,
    holdout_index: np.ndarray,
    train_cosines: np.ndarray,
    holdout_cosines: np.ndarray,
    stability: StabilityResult,
    include_vector: bool,
) -> dict[str, object]:
    """One candidate's full record: verdict, both sides of the split, and its stability."""
    agreement_rate = float(np.mean(holdout_cosines > 0.0))
    mean_holdout = float(np.mean(holdout_cosines))
    generalizes = bool(
        mean_holdout >= MIN_HELDOUT_MEAN_COSINE and agreement_rate >= MIN_HELDOUT_AGREEMENT_RATE
    )
    payload: dict[str, object] = {
        "rank": rank,
        "generalizes": generalizes,
        "measure": MEASURE_HELDOUT_COSINE,
        "train": {
            "pairs": int(train_index.shape[0]),
            "mean_cosine": _round(float(np.mean(train_cosines))),
            "energy_share": _round(share),
            "pairs_used": _pair_entries(pairs, train_index, train_cosines),
        },
        "holdout": {
            "pairs": int(holdout_index.shape[0]),
            "mean_cosine": _round(mean_holdout),
            "min_cosine": _round(float(np.min(holdout_cosines))),
            "agreement_rate": _round(agreement_rate),
            "pairs_used": _pair_entries(pairs, holdout_index, holdout_cosines),
        },
        "stability": stability.to_dict(),
        "stability_summary": stability.describe(),
    }
    if include_vector:
        payload["vector"] = [_round(component) for component in direction]
    return payload


def _render(payload: dict[str, object]) -> str:
    pairs = payload["pairs"]
    params = payload["params"]
    lines = [
        "# directions discover",
        "",
        f"atlas:      {payload['atlas']}",
        f"model:      {payload['model']}",
        f"pairs:      {pairs['total']} ({pairs['train']} train, {pairs['holdout']} held-out) "
        f"from {pairs['source']}",
        f"seed:       {payload['seed']}",
        "params:     " + ", ".join(f"{key}={value}" for key, value in sorted(params.items())),
        "",
        f"generalization test: {MEASURE_HELDOUT_COSINE} >= "
        f"{MIN_HELDOUT_MEAN_COSINE} and agreement rate >= {MIN_HELDOUT_AGREEMENT_RATE} "
        "on the held-out pairs",
    ]
    for candidate in payload["candidates"]:
        verdict = "GENERALIZES" if candidate["generalizes"] else "DOES NOT GENERALIZE"
        suffix = "" if candidate["generalizes"] else " (reported as a negative result)"
        holdout = candidate["holdout"]
        train = candidate["train"]
        provenance = (
            ", ".join(f"{entry['a']}->{entry['b']}" for entry in train["pairs_used"][:5])
            or "(none)"
        )
        if len(train["pairs_used"]) > 5:
            provenance += f", ... ({len(train['pairs_used'])} pairs total)"
        lines += [
            "",
            f"## candidate {candidate['rank']} — {verdict}{suffix}",
            f"held-out:   {MEASURE_HELDOUT_COSINE} {holdout['mean_cosine']:.3f} "
            f"(min {holdout['min_cosine']:.3f}) over {holdout['pairs']} pair(s); "
            f"agreement rate {holdout['agreement_rate']:.3f}",
            f"train:      mean cosine {train['mean_cosine']:.3f} over {train['pairs']} pair(s); "
            f"energy share {train['energy_share']:.3f}",
            f"stability:  {candidate['stability_summary']}",
            f"derived from: {provenance}",
        ]
    lines += [
        "",
        f"summary: {payload['generalizing']} of {len(payload['candidates'])} candidate(s) "
        "generalize to the held-out pairs.",
    ]
    if payload["uncertain"]:
        lines += ["", "uncertain:"]
        lines += [f"- {note}" for note in payload["uncertain"]]
    lines += ["", "limitations:"]
    lines += [f"- {note}" for note in payload["limitations"]]
    lines += ["", reproduce_line(payload[REPRODUCED_BY_KEY])]
    return "\n".join(lines)


def cmd_discover(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    holdout = float(args.holdout)
    top = int(args.top)
    runs = int(args.runs)
    resamples = int(args.resamples)
    sample_pairs = int(args.sample_pairs)
    seed = int(args.seed)
    _validate(holdout, top, runs, resamples, sample_pairs)

    atlas = load_atlas(args)
    pairs, source = _collect_pairs(atlas, args, seed)
    if len(pairs) < 2:
        raise _user_error(
            f"need at least 2 item pairs to derive and then test a direction, got {len(pairs)}",
            "supply more pairs via --pairs, or omit --pairs to sample them from the atlas; "
            "a direction fitted to pairs it is then tested on is not evidence",
        )

    diffs = _difference_matrix(atlas, pairs)
    train_index, holdout_index = split_pairs(len(pairs), holdout, seed)
    directions, shares = derive_directions(diffs[train_index], top)

    params: dict[str, object] = {
        "pairs": len(pairs),
        "holdout": holdout,
        "top": top,
        "runs": runs,
        "resamples": resamples,
        "sample_pairs": sample_pairs if source == "sampled" else None,
        "pair_source": source,
    }

    candidates: list[dict[str, object]] = []
    uncertain: list[str] = []
    for rank, direction in enumerate(directions):
        stability = measure_stability(
            np.asarray(atlas.vectors, dtype=np.float64),
            _bipartition_fn(diffs, rank, holdout, top),
            params={**params, "rank": rank},
            seed=seed,
            runs=runs,
            resamples=resamples,
        )
        candidate = _candidate_payload(
            rank=rank,
            direction=direction,
            share=float(shares[rank]),
            pairs=pairs,
            train_index=train_index,
            holdout_index=holdout_index,
            train_cosines=alignment(diffs[train_index], direction),
            holdout_cosines=alignment(diffs[holdout_index], direction),
            stability=stability,
            include_vector=bool(getattr(args, "include_vectors", False)),
        )
        subject = f"candidate direction {rank}"
        if not candidate["generalizes"]:
            uncertain.append(
                f"{subject}: does not generalize — held-out {MEASURE_HELDOUT_COSINE} "
                f"{candidate['holdout']['mean_cosine']:.3f} with agreement rate "
                f"{candidate['holdout']['agreement_rate']:.3f} over "
                f"{candidate['holdout']['pairs']} held-out pair(s)"
            )
        if stability.is_uncertain:
            uncertain.append(stability.uncertain_note(subject))
        candidates.append(candidate)

    generalizing = sum(1 for candidate in candidates if candidate["generalizes"])
    payload: dict[str, object] = {
        "atlas": str(atlas.path),
        "model": atlas.manifest.model,
        "endpoint": atlas.manifest.endpoint,
        "seed": seed,
        "params": params,
        "measure": MEASURE_HELDOUT_COSINE,
        "thresholds": {
            "min_holdout_mean_cosine": MIN_HELDOUT_MEAN_COSINE,
            "min_holdout_agreement_rate": MIN_HELDOUT_AGREEMENT_RATE,
        },
        "pairs": {
            "source": source,
            "total": len(pairs),
            "train": int(train_index.shape[0]),
            "holdout": int(holdout_index.shape[0]),
        },
        "candidates": candidates,
        "generalizing": generalizing,
        "not_generalizing": len(candidates) - generalizing,
        "uncertain": uncertain,
        "limitations": [_FRAMING_LIMITATION, _PAIR_SET_LIMITATION, *atlas.manifest.limitations],
        REPRODUCED_BY_KEY: reproducing_command(),
    }
    emit_result(payload if json_mode else _render(payload), json_mode=json_mode)
    return 0


# --------------------------------------------------------------------------
# The overview sub-verb (rubric: any noun with action-verbs must expose one)
# --------------------------------------------------------------------------


def directions_sections(atlas_path: Path) -> list[dict[str, object]]:
    """Sections describing this noun. Never touches the atlas beyond ``exists()``."""
    return [
        {
            "title": "Purpose",
            "items": [
                "propose candidate semantic directions from paired items and test whether "
                "each one holds on pairs it was not derived from",
                "a direction is a vector in the space — never a dimension index, and never "
                "given a human-readable label by this tool",
            ],
        },
        {
            "title": "Sub-verbs",
            "items": [
                "directions discover — derive candidates, test generalization, measure stability",
                "directions overview — this description (also printed by a bare "
                "`embeddings directions`)",
            ],
        },
        {
            "title": "Method",
            "items": [
                "pairs come from --pairs (atlas item ids, JSONL or CSV) or are sampled from "
                "the atlas when --pairs is omitted",
                "candidates are the leading singular vectors of the training pairs' "
                "difference vectors, oriented from a toward b",
                f"generalization: {MEASURE_HELDOUT_COSINE} plus the agreement rate over the "
                "held-out pairs, against fixed thresholds reported in every payload",
                "stability: adjusted_rand_index over the item bipartition the direction "
                "induces, across seed reruns and bootstrap resamples",
            ],
        },
        {
            "title": "Reading the output",
            "items": [
                "candidates that fail the thresholds are still listed, marked "
                "'DOES NOT GENERALIZE', and recorded in uncertain[]",
                "a single run (--runs 1 --resamples 0) reports 'stability unverified' with a "
                "null score, never a zero",
                "params and seed are on every payload; the same seed over an unchanged atlas "
                "reproduces the output byte for byte",
            ],
        },
        {
            "title": "Atlas",
            "items": [
                f"resolved path: {atlas_path}",
                f"present: {'yes' if Path(atlas_path).exists() else 'no'}",
                "discover requires an atlas (exit 1 if absent); overview never does",
            ],
        },
    ]


def cmd_directions_overview(args: argparse.Namespace) -> int:
    # Descriptive: resolves where it *would* look without requiring anything to
    # be there, so `directions overview` on a machine that has never surveyed
    # anything still exits 0.
    emit_overview(
        "embeddings-cli directions",
        directions_sections(resolve_atlas_path(args)),
        json_mode=bool(getattr(args, "json", False)),
    )
    return 0


def _no_verb(args: argparse.Namespace) -> int:
    # `embeddings directions` with no sub-verb prints the noun's overview.
    return cmd_directions_overview(args)


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "directions",
        help="Discover candidate semantic directions and test whether they generalize.",
        description=(
            "Propose candidate directions from paired items, then measure whether each one "
            "holds on held-out pairs. Candidates that do not transfer are reported as "
            "negative results, never hidden."
        ),
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    add_atlas_argument(p)
    p.set_defaults(func=_no_verb, json=False)
    # `p` is a _CliArgumentParser (the top-level subparsers were built with that
    # parser_class); propagate it so `directions <sub-verb>` parse errors route
    # through the structured error contract instead of argparse's default
    # stderr/exit 2.
    noun_sub = p.add_subparsers(
        dest="directions_command",
        parser_class=type(p),
        help="What to do with directions: describe the noun, or discover candidates.",
    )

    ov = noun_sub.add_parser(
        "overview",
        help="Describe what `directions` measures and how to read its output.",
    )
    ov.add_argument(
        "target",
        nargs="?",
        help="Ignored — overview always describes this noun itself. Accepted so a stray "
        "path argument never hard-fails a descriptive verb.",
    )
    add_atlas_argument(ov)
    ov.add_argument("--json", action="store_true", help="Emit structured JSON.")
    ov.set_defaults(func=cmd_directions_overview)

    dis = noun_sub.add_parser(
        "discover",
        help="Derive candidate directions from item pairs and test them on held-out pairs.",
        description=(
            "Derive candidate directions from the training half of a pair set, then measure "
            "each candidate against the held-out half it never saw. Reports the "
            "generalization result either way, plus a stability measurement across seed "
            "reruns and bootstrap resamples."
        ),
    )
    add_atlas_argument(dis)
    dis.add_argument(
        "--pairs",
        default=None,
        metavar="<path>",
        help="File of (a, b) atlas item-id pairs, one per line: either a JSON object "
        '\'{"a": "<id>", "b": "<id>"}\' or a bare \'<id>,<id>\' row. Ids resolve against '
        "the atlas, so no embedding backend is needed. Omit to sample pairs instead.",
    )
    dis.add_argument(
        "--sample-pairs",
        type=int,
        default=DEFAULT_SAMPLE_PAIRS,
        help=f"How many random item pairs to sample when --pairs is omitted (default: "
        f"{DEFAULT_SAMPLE_PAIRS}). Sampling is seeded by --seed.",
    )
    dis.add_argument(
        "--holdout",
        type=float,
        default=DEFAULT_HOLDOUT,
        help=f"Fraction of the pair set held out from derivation and used only to test "
        f"generalization (default: {DEFAULT_HOLDOUT}). At least one pair always stays on "
        "each side of the split.",
    )
    dis.add_argument(
        "--top",
        type=int,
        default=DEFAULT_TOP,
        help=f"How many candidate directions to propose, in descending order of the "
        f"training differences' energy along them (default: {DEFAULT_TOP}).",
    )
    dis.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Base seed for the train/held-out split, pair sampling, and every stability "
        f"rerun (default: {DEFAULT_SEED}). Recorded on the payload; the same seed over an "
        "unchanged atlas reproduces the output exactly.",
    )
    dis.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Seed reruns behind each stability measurement (default: {DEFAULT_RUNS}). "
        "--runs 1 with --resamples 0 reports 'stability unverified' rather than a score.",
    )
    dis.add_argument(
        "--resamples",
        type=int,
        default=DEFAULT_RESAMPLES,
        help=f"Bootstrap resamples of the surveyed items behind each stability measurement "
        f"(default: {DEFAULT_RESAMPLES}); 0 skips the bootstrap.",
    )
    dis.add_argument(
        "--include-vectors",
        action="store_true",
        help="Include each candidate direction's full vector in the output. Off by default "
        "because it dominates the payload at realistic dimensionalities.",
    )
    dis.add_argument("--json", action="store_true", help="Emit structured JSON.")
    dis.set_defaults(func=cmd_discover)
