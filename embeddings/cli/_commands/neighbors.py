"""``embeddings neighbors`` — k nearest Atlas items to a query, by cosine distance.

The query is either free text (embedded through the gateway, then compared
against the Atlas) or an existing Atlas item id (looked up by its already-
stored vector). Those two paths have deliberately different egress
properties, and that difference is the point of this verb:

* **Free text** necessarily calls the embedding backend — there is no vector
  to compare against until the query itself is embedded. It goes through
  :func:`build_client` and the same on-disk embed cache ``survey`` uses, so a
  repeated query is served from cache rather than re-embedded.
* **An Atlas item id** (``--id``) needs nothing but the Atlas already on
  disk: its vector was stored by ``survey``, so this path never constructs an
  :class:`~embeddings.backends.client.EmbeddingClient` at all.
  ``tests/test_neighbors.py`` proves this the strong way, with
  :func:`tests._fakes.blocked_sockets` wrapped around an ``--id`` run.

Disambiguation is a flag, never a guess
-----------------------------------------
The positional argument is *always* free text unless ``--id`` is passed. A
query string that happens to spell an existing Atlas item id is still
embedded as text without ``--id`` — there is no heuristic that inspects the
string's shape and decides for the caller. An agent scripting against this
CLI needs a resolution rule it can rely on without reading the Atlas first;
"the flag decides" is that rule.

Metric
------
Distance is cosine distance (``1 - cosine similarity``, range ``[0, 2]``),
the standard choice for embedding comparison because it is invariant to
vector scale — exactly the axis embedding models are not trained to hold
fixed. It is stated in every output (top-level ``metric`` and inside
``params``) rather than left implicit, per
``docs/design-notes/semantic-measurement.md``'s "no output value may mix
metrics silently" posture: this verb has exactly one metric and never
switches it based on input shape.

The query embedding model defaults to the Atlas's own ``manifest.model``
(overridable via ``--embed-model``) rather than this CLI's built-in default,
because comparing a query embedded by one model against vectors produced by
another is comparing numbers from different, incommensurable spaces (see the
design note's §2 discussion of why raw coordinates do not transfer across
models). A resulting dimension mismatch is reported as a user error naming
``--embed-model`` rather than as a bare shape-mismatch traceback.

Boundary description
---------------------
The "boundary" of a neighborhood is reported as the sorted-distance gap
profile across the returned neighbors — the per-step increase in distance
from one neighbor to the next, plus which step holds the largest jump. This
is a measured fact about the returned distances, not a claim that the
neighborhood *is* coherent up to that point and incoherent after: with fewer
than two neighbors there is no gap to report, and with more than two the
profile is reported in full so a reader can judge it rather than being handed
a single verdict.

``seed``
--------
Recorded on every payload for consistency with the rest of the reader-verb
surface (``docs/design-notes/semantic-measurement.md`` §3: "every measurement
output records its full parameters and its seed"). The neighbor computation
itself is exact cosine distance over fixed vectors — deterministic without
any random component — so the seed does not currently change this verb's
output; it is recorded anyway so the payload shape matches every other
structural verb and stays forward-compatible with a future tie-breaking or
sampling step that would consume it.
"""

from __future__ import annotations

import argparse

import numpy as np

from embeddings.atlas.io import Atlas
from embeddings.atlas.schema import AtlasRecord
from embeddings.backends import (
    ENDPOINT_FLAG,
    ENDPOINT_HELP,
    EmbeddingCache,
    EmbeddingClient,
    get_or_embed,
)
from embeddings.backends.client import MODEL_FLAG, MODEL_HELP
from embeddings.cli._commands._atlas_arg import add_atlas_argument, load_atlas, resolve_atlas_path
from embeddings.cli._errors import EXIT_USER_ERROR, CliError
from embeddings.cli._output import emit_result
from embeddings.cli._provenance import (
    REPRODUCED_BY_KEY,
    reproduce_line,
    reproducing_command,
)

#: The one distance metric this verb ever computes or reports. Stated as a
#: module constant, never derived or switched at runtime, so "cosine" in the
#: docstring and "cosine" in every payload can never drift apart.
METRIC = "cosine"

#: Default number of neighbors returned.
DEFAULT_TOP_K = 5

#: Default seed recorded in output. See the module docstring's ``seed``
#: section for why this verb records one despite having no random step.
DEFAULT_SEED = 0


def build_client(base_url: str | None, model: str | None) -> EmbeddingClient:
    """Construct the embedding client for a free-text query — this module's only egress point.

    ``None`` for either argument defers to
    :class:`~embeddings.backends.client.EmbeddingClient`'s own environment-variable
    and built-in defaults. Callers here always pass an explicit ``model`` (the
    Atlas's own model, unless ``--embed-model`` overrides it) — see
    :func:`cmd_neighbors` — so the "defer to the client's built-in default
    model" branch is only reached when even the Atlas's model was withheld,
    which does not happen through the CLI. Kept as a single seam, matching
    ``survey.build_client``, so tests substitute a fake-transport-backed
    client without a test-only flag leaking into the CLI surface.
    """
    kwargs: dict[str, str] = {}
    if base_url:
        kwargs["base_url"] = base_url
    if model:
        kwargs["model"] = model
    return EmbeddingClient(**kwargs)


def _validate_top_k(top_k: int) -> None:
    if top_k < 1:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"--top-k must be a positive integer, got {top_k}",
            remediation=f"pass --top-k 1 or higher (default: {DEFAULT_TOP_K})",
        )


def _record_by_id(atlas: Atlas, item_id: str) -> AtlasRecord:
    for record in atlas.records:
        if record.id == item_id:
            return record
    raise CliError(
        code=EXIT_USER_ERROR,
        message=f"no atlas item with id {item_id!r}",
        remediation=(
            "check records.jsonl in the atlas directory for valid ids, or drop --id to "
            "embed the query as free text instead"
        ),
    )


def _query_vector_from_id(atlas: Atlas, item_id: str) -> tuple[np.ndarray, str | None]:
    """Look up ``item_id``'s stored vector. Touches only the already-loaded Atlas."""
    record = _record_by_id(atlas, item_id)
    return np.asarray(atlas.vectors[record.vector_index], dtype=np.float64), record.id


def _query_vector_from_text(
    text: str, atlas: Atlas, base_url: str | None, model: str | None
) -> tuple[np.ndarray, EmbeddingClient]:
    """Embed ``text`` — this verb's only path that may reach the gateway.

    ``model`` defaults to the Atlas's own model when the caller passed
    ``None`` (i.e. ``--embed-model`` was not given), so a bare
    ``embeddings neighbors <text>`` compares against the Atlas in the space
    it was actually built in.
    """
    client = build_client(base_url, model or atlas.manifest.model)
    cache = EmbeddingCache()
    (vector,) = get_or_embed(cache, client, [text])
    return np.asarray(vector, dtype=np.float64), client


def _cosine_distances(vectors: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Cosine distance (``1 - cosine similarity``) from ``query`` to every row of ``vectors``.

    A zero-magnitude row (an all-zero vector, which cosine similarity cannot
    define) is treated as maximally distant (``2.0``) rather than raising or
    silently producing ``nan`` — an all-zero embedding is already a
    degenerate input, and reporting it as "as far as possible" keeps the
    output usable instead of poisoning the sort with a NaN.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    query = np.asarray(query, dtype=np.float64)
    query_norm = float(np.linalg.norm(query))
    if query_norm == 0.0:
        raise CliError(
            code=EXIT_USER_ERROR,
            message="the query vector has zero magnitude; cosine distance is undefined for it",
            remediation="use a different query — an all-zero embedding cannot be compared",
        )
    vector_norms = np.linalg.norm(vectors, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        cosine_similarity = (vectors @ query) / (vector_norms * query_norm)
    zero_rows = vector_norms == 0.0
    cosine_similarity = np.where(zero_rows, -1.0, cosine_similarity)
    cosine_similarity = np.clip(cosine_similarity, -1.0, 1.0)
    return 1.0 - cosine_similarity


def _nearest(
    atlas: Atlas, query: np.ndarray, top_k: int, *, exclude_id: str | None
) -> list[tuple[AtlasRecord, float]]:
    candidates = [r for r in atlas.records if r.id != exclude_id]
    if not candidates:
        return []

    dimension = atlas.vectors.shape[1] if atlas.vectors.ndim == 2 else 0
    if query.shape[-1] != dimension:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=(
                f"the embedded query has dimension {query.shape[-1]}, but this atlas's "
                f"vectors have dimension {dimension}"
            ),
            remediation=(
                f"pass --embed-model to match the model this atlas was surveyed with "
                f"({atlas.manifest.model!r}), or re-survey with the model you want to query with"
            ),
        )

    indices = np.array([record.vector_index for record in candidates])
    distances = _cosine_distances(atlas.vectors[indices], query)
    order = np.argsort(distances, kind="stable")[:top_k]
    return [(candidates[i], float(distances[i])) for i in order]


def _boundary_profile(distances: list[float]) -> dict[str, object]:
    """The sorted-distance gap profile across the returned neighbors.

    See the module docstring's "Boundary description" section: this reports
    the per-step distance increase and which step holds the largest jump,
    without asserting where the neighborhood "stops being coherent" —
    that judgment is left to the reader.
    """
    if len(distances) < 2:
        return {
            "gaps": [],
            "largest_gap_index": None,
            "largest_gap": None,
            "note": "fewer than two neighbors returned; no distance gap can be computed",
        }
    gaps = [distances[i + 1] - distances[i] for i in range(len(distances) - 1)]
    largest_gap = max(gaps)
    largest_gap_index = gaps.index(largest_gap)
    return {
        "gaps": gaps,
        "largest_gap_index": largest_gap_index,
        "largest_gap": largest_gap,
        "note": (
            f"largest distance jump (gap={largest_gap:.6f}) is between neighbor "
            f"{largest_gap_index + 1} and neighbor {largest_gap_index + 2}"
        ),
    }


def _render_text(payload: dict[str, object]) -> str:
    lines = [
        f"# neighbors (metric: {payload['metric']})",
        "",
        f"query:    {payload['query']!r} (source: {payload['source']})",
        f"atlas:    {payload['atlas']}",
        f"k:        {payload['k']} (returned {payload['returned']})",
        f"seed:     {payload['seed']}",
        "",
    ]
    neighbors = payload["neighbors"]
    if isinstance(neighbors, list) and neighbors:
        for position, neighbor in enumerate(neighbors, start=1):
            lines.append(
                f"{position:>2}. {neighbor['id']}  distance={neighbor['distance']:.6f}  "
                f"text={neighbor['text']!r}"
            )
    else:
        lines.append("(no neighbors found)")
    lines.append("")
    boundary = payload["boundary"]
    if isinstance(boundary, dict):
        lines.append(f"boundary: {boundary['note']}")
    lines.append("")
    lines.append(reproduce_line(payload[REPRODUCED_BY_KEY]))
    return "\n".join(lines)


def cmd_neighbors(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    top_k = int(args.top_k)
    _validate_top_k(top_k)

    atlas = load_atlas(args)
    atlas_path = resolve_atlas_path(args)

    if bool(args.id):
        query_vector, exclude_id = _query_vector_from_id(atlas, args.query)
        model_used: str | None = None
        endpoint_used: str | None = None
    else:
        query_vector, client = _query_vector_from_text(
            args.query, atlas, getattr(args, "embed_url", None), getattr(args, "embed_model", None)
        )
        exclude_id = None
        model_used = client.model
        endpoint_used = client.base_url

    neighbors = _nearest(atlas, query_vector, top_k, exclude_id=exclude_id)
    distances = [distance for _, distance in neighbors]
    boundary = _boundary_profile(distances)

    source = "atlas-item" if args.id else "text"
    payload: dict[str, object] = {
        "query": args.query,
        "source": source,
        "atlas": str(atlas_path),
        "metric": METRIC,
        "k": top_k,
        "returned": len(neighbors),
        "neighbors": [
            {
                "id": record.id,
                "text": record.text,
                "distance": distance,
                "metadata": dict(record.metadata),
            }
            for record, distance in neighbors
        ],
        "boundary": boundary,
        "params": {
            "metric": METRIC,
            "k": top_k,
            "atlas": str(atlas_path),
            "source": source,
            "model": model_used,
            "endpoint": endpoint_used,
        },
        "seed": int(args.seed),
        REPRODUCED_BY_KEY: reproducing_command(),
    }

    emit_result(payload if json_mode else _render_text(payload), json_mode=json_mode)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "neighbors",
        help="Find the k nearest Atlas items to a text query or an existing item id.",
        description=(
            "Embed a text query (or look up an existing Atlas item id) and report its k "
            "nearest neighbors in the Atlas by cosine distance, plus a boundary description "
            "of the returned distance profile."
        ),
    )
    p.add_argument(
        "query",
        help="Free text to embed and search, or an Atlas item id when --id is given. "
        "Always treated as free text unless --id is passed -- never guessed.",
    )
    add_atlas_argument(p)
    p.add_argument(
        "--id",
        action="store_true",
        help="Treat QUERY as an existing Atlas item id rather than free text. Looks up its "
        "already-stored vector and makes no embedding-backend call at all.",
    )
    p.add_argument(
        "-k",
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help=f"Number of nearest neighbors to return (default: {DEFAULT_TOP_K}).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Seed recorded in the output for consistency with other reader verbs "
        f"(default: {DEFAULT_SEED}); this verb's cosine-distance computation has no random "
        "component and does not consume it.",
    )
    # Flag names come from the backend constants, matching survey.py, so the
    # flag, its help text, and the client's own unreachable-endpoint hint can
    # never drift apart.
    p.add_argument(ENDPOINT_FLAG, default=None, help=ENDPOINT_HELP)
    p.add_argument(
        MODEL_FLAG, default=None, help=MODEL_HELP + " Defaults to the atlas's own model."
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_neighbors)
