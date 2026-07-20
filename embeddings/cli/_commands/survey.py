"""``embeddings survey`` — embed a corpus and write the Atlas every reader verb reads.

This is the only verb in embeddings-cli that writes an Atlas, which is why its
default is to write nothing at all.

Dry run by default
------------------
Without ``--apply`` this verb *previews*: it reports how many corpus items were
loaded, which endpoint and model would embed them, where the Atlas would land,
and whether something is already there — and it performs **zero filesystem
writes and zero gateway calls** doing so. That is a hard property, not a
best-effort one, and ``tests/test_survey.py`` proves it by hashing the whole
working tree before and after a preview.

The reason is the calling pattern this CLI is built for. Agents invoke it in
loops, often with generated arguments, often without reading the help first. A
write verb that commits by default eventually overwrites an Atlas that someone
committed as study evidence, and an Atlas is an *interpretation* — re-running
the survey does not reproduce it exactly once live embeddings are involved (see
``docs/design-notes/semantic-measurement.md`` §4). Preview-then-apply makes the
destructive step something the caller had to type.

Replacing an existing Atlas
---------------------------
``--apply`` onto an existing Atlas is refused (exit 1) with a hint naming
``--force``. ``--force`` maps straight onto ``write_atlas(..., overwrite=True)``
— the old directory is removed and the new Atlas written whole. There is no
merge path anywhere in this module: a half-old, half-new Atlas would be an
interpretation nobody produced and no command reproduces.

Batching, progress, and retry
-----------------------------
Embedding runs in ``--batch-size`` chunks, each reported to **stderr** as a
progress diagnostic, so an agent watching a long survey sees liveness rather
than silence. Environment-class failures (exit code 2 — a gateway that dropped
the connection, a malformed response) are retried up to ``--retries`` times with
linear backoff; user-class failures are never retried. When retries are
exhausted the underlying :class:`CliError` propagates unchanged, so a genuinely
unreachable backend still exits 2 with the client's own endpoint-naming hint.

The float-variance limitation
-----------------------------
Per design note §4, an Atlas built from *live* (non-cache-served) embeddings
records that fact in its manifest ``limitations[]`` — GPU inference is not
bit-deterministic, so those vectors are only approximately reproducible, and
downstream clustering can amplify a float-level difference into a different
partition. :data:`LIVE_EMBEDDING_LIMITATION` is written by this module
automatically, never left to the operator: the operator most likely to omit it
is the one who did not know it applied. An Atlas served entirely from the embed
cache is byte-reproducible and carries no such entry.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

import numpy as np

from embeddings.atlas import SCHEMA_VERSION, AtlasManifest, AtlasRecord, write_atlas
from embeddings.atlas.io import DEFAULT_ATLAS_DIR
from embeddings.backends import (
    ENDPOINT_FLAG,
    ENDPOINT_HELP,
    ENV_CACHE_DIR,
    EmbeddingCache,
    EmbeddingClient,
    get_or_embed,
)
from embeddings.backends.client import MODEL_FLAG, MODEL_HELP
from embeddings.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from embeddings.cli._output import emit_diagnostic, emit_result
from embeddings.cli._provenance import (
    REPRODUCED_BY_KEY,
    reproduce_line,
    reproducing_command,
)
from embeddings.corpus import CorpusItem, load_corpus

#: The design-note §4 limitation entry written into ``limitations[]`` whenever
#: any vector in an Atlas came from a live embed rather than the cache. Kept as
#: a module constant so tests assert the exact string the manifest carries.
LIVE_EMBEDDING_LIMITATION = (
    "live-embedding float variance: some vectors in this atlas were embedded live rather "
    "than served from the embed cache. GPU inference is not bit-deterministic (kernel "
    "selection, reduction order, batch- and load-dependent code paths), so re-running this "
    "survey may produce vectors that differ at float precision, and downstream clustering "
    "can amplify such a difference into a different partition near a decision boundary. "
    "This atlas is the reproducibility boundary: everything derived from it is exact, "
    "everything upstream of it is approximate. See docs/design-notes/semantic-measurement.md."
)

#: Default number of texts sent to the backend per request.
DEFAULT_BATCH_SIZE = 64
#: Default number of *retries* (so the default total attempts per batch is 3).
DEFAULT_RETRIES = 2
#: Default RNG seed stamped into the manifest.
DEFAULT_SEED = 0
#: Linear backoff base between retry attempts, in seconds. Module-level so a
#: test can zero it out rather than sleeping through the retry path.
RETRY_BACKOFF_SECONDS = 0.25

#: Vectors are stored as float32: the Atlas's vector array is regenerable bulk
#: (git-ignored ``.npy``), and float32 halves it at no cost to any structural
#: measurement this CLI makes.
_VECTOR_DTYPE = np.float32


def build_client(base_url: str | None, model: str | None) -> EmbeddingClient:
    """Construct the embedding client for this run — the module's single egress point.

    ``None`` for either argument means "leave it to
    :class:`~embeddings.backends.client.EmbeddingClient`", which resolves the
    ``EMBEDDINGS_EMBED_URL`` / ``EMBEDDINGS_EMBED_MODEL`` environment variables
    and then the built-in defaults. Because every client this verb uses comes
    from here, tests substitute a fake-transport-backed client by replacing this
    one function, and no test seam has to leak into the flag surface.
    """
    kwargs: dict[str, str] = {}
    if base_url:
        kwargs["base_url"] = base_url
    if model:
        kwargs["model"] = model
    return EmbeddingClient(**kwargs)


def _unique_texts(items: Sequence[CorpusItem]) -> tuple[list[str], dict[str, int]]:
    """Deduplicate corpus text, preserving first-seen order.

    Identical text embeds to one vector, so it gets one row in ``vectors.npy``
    and every corpus item carrying that text points at the same
    ``vector_index`` — which ``embeddings/atlas/io.py`` explicitly allows.
    """
    ordered: list[str] = []
    index_by_text: dict[str, int] = {}
    for item in items:
        if item.text not in index_by_text:
            index_by_text[item.text] = len(ordered)
            ordered.append(item.text)
    return ordered, index_by_text


def _cache_status(cache: EmbeddingCache, model: str, texts: Sequence[str]) -> tuple[int, int]:
    """``(hits, misses)`` for ``texts`` under ``model`` — a stat per text, no reads.

    Only ``Path.exists()`` is consulted, never the entry bytes, so this is cheap
    enough to run during a preview and never decodes a vector it will not use.
    """
    hits = sum(1 for text in texts if cache.path_for(model, text).exists())
    return hits, len(texts) - hits


def _embed_batch(
    cache: EmbeddingCache,
    client: EmbeddingClient,
    texts: Sequence[str],
    *,
    retries: int,
    label: str,
) -> list[list[float]]:
    """Embed one batch, retrying environment-class failures up to ``retries`` times.

    Only :data:`~embeddings.cli._errors.EXIT_ENV_ERROR` failures are retried: a
    dropped connection or a malformed response may well succeed on a second
    attempt, whereas a user-class error is deterministic and retrying it just
    burns time. When attempts run out the last :class:`CliError` propagates
    untouched, so an unreachable backend still exits 2 carrying the client's own
    message and hint.
    """
    attempts = retries + 1
    for attempt in range(1, attempts + 1):
        try:
            return get_or_embed(cache, client, texts)
        except CliError as err:
            if err.code != EXIT_ENV_ERROR or attempt == attempts:
                raise
            emit_diagnostic(
                f"survey: {label} failed (attempt {attempt}/{attempts}): {err.message} "
                f"— retrying in {RETRY_BACKOFF_SECONDS * attempt:.2f}s"
            )
            if RETRY_BACKOFF_SECONDS:
                time.sleep(RETRY_BACKOFF_SECONDS * attempt)
        except OSError as err:  # cache write failure — environment, not usage
            raise CliError(
                code=EXIT_ENV_ERROR,
                message=f"failed to store embeddings in the cache: {err}",
                remediation=(
                    f"check the cache directory is writable, or point {ENV_CACHE_DIR} elsewhere"
                ),
            ) from err
    raise CliError(  # pragma: no cover - the loop above always returns or raises
        code=EXIT_ENV_ERROR,
        message="embedding failed without recording an error",
        remediation="re-run the survey",
    )


def _embed_all(
    cache: EmbeddingCache,
    client: EmbeddingClient,
    texts: Sequence[str],
    *,
    batch_size: int,
    retries: int,
) -> list[list[float]]:
    """Embed every text in ``batch_size`` chunks, reporting progress to stderr."""
    total = len(texts)
    batch_count = (total + batch_size - 1) // batch_size
    vectors: list[list[float]] = []
    for number, start in enumerate(range(0, total, batch_size), start=1):
        chunk = texts[start : start + batch_size]
        label = f"batch {number}/{batch_count}"
        vectors.extend(_embed_batch(cache, client, chunk, retries=retries, label=label))
        emit_diagnostic(f"survey: embedded {len(vectors)}/{total} texts ({label})")
    return vectors


def _preview_payload(
    *,
    corpus_path: Path,
    items: Sequence[CorpusItem],
    unique: Sequence[str],
    client: EmbeddingClient,
    target: Path,
    force: bool,
    cache_hits: int,
    cache_misses: int,
    batch_size: int,
    seed: int,
) -> dict[str, object]:
    exists = target.exists()
    return {
        "applied": False,
        "corpus": str(corpus_path),
        "items": len(items),
        "unique_texts": len(unique),
        "model": client.model,
        "endpoint": client.base_url,
        "atlas": str(target),
        "atlas_exists": exists,
        "would_replace": exists and force,
        "cache_hits": cache_hits,
        "cache_misses": cache_misses,
        "batch_size": batch_size,
        "seed": seed,
        REPRODUCED_BY_KEY: reproducing_command(),
    }


def _render_preview(payload: dict[str, object]) -> str:
    lines = [
        "# survey (dry run — nothing written)",
        "",
        f"corpus:        {payload['corpus']}",
        f"items:         {payload['items']} ({payload['unique_texts']} unique texts)",
        f"model:         {payload['model']}",
        f"endpoint:      {payload['endpoint']}",
        f"atlas target:  {payload['atlas']}"
        + (" (already exists)" if payload["atlas_exists"] else ""),
        f"cache:         {payload['cache_hits']} hits, {payload['cache_misses']} to embed",
        f"batch size:    {payload['batch_size']}",
        f"seed:          {payload['seed']}",
        "",
    ]
    if payload["atlas_exists"] and not payload["would_replace"]:
        lines.append("re-run with --apply --force to replace the existing atlas whole.")
    else:
        lines.append("re-run with --apply to embed the corpus and write the atlas.")
    lines.append("")
    lines.append(reproduce_line(payload[REPRODUCED_BY_KEY]))
    return "\n".join(lines)


def _render_applied(payload: dict[str, object]) -> str:
    lines = [
        "# survey (applied)",
        "",
        f"atlas:         {payload['atlas']}",
        f"items:         {payload['items']} ({payload['unique_texts']} unique texts)",
        f"model:         {payload['model']}",
        f"endpoint:      {payload['endpoint']}",
        f"vectors:       {payload['vectors']} x {payload['dimensions']}",
        f"cache:         {payload['cache_hits']} hits, {payload['embedded_live']} embedded live",
        f"seed:          {payload['seed']}",
    ]
    limitations = payload["limitations"]
    if isinstance(limitations, list) and limitations:
        lines += ["", "limitations recorded in the manifest:"]
        lines += [f"- {entry}" for entry in limitations]
    lines += ["", reproduce_line(payload[REPRODUCED_BY_KEY])]
    return "\n".join(lines)


def _validate(batch_size: int, retries: int) -> None:
    if batch_size < 1:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"--batch-size must be a positive integer, got {batch_size}",
            remediation=f"pass --batch-size 1 or higher (default: {DEFAULT_BATCH_SIZE})",
        )
    if retries < 0:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"--retries must be zero or greater, got {retries}",
            remediation=f"pass --retries 0 to disable retrying (default: {DEFAULT_RETRIES})",
        )


def cmd_survey(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    batch_size = int(args.batch_size)
    retries = int(args.retries)
    _validate(batch_size, retries)

    corpus_path = Path(args.corpus)
    items = load_corpus(str(corpus_path))
    if not items:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"corpus {corpus_path} contains no items",
            remediation="add at least one JSON object with a 'text' field, one per line",
        )

    unique, index_by_text = _unique_texts(items)
    client = build_client(getattr(args, "embed_url", None), getattr(args, "embed_model", None))
    cache = EmbeddingCache()
    target = Path(args.atlas)
    cache_hits, cache_misses = _cache_status(cache, client.model, unique)

    if not args.apply:
        payload = _preview_payload(
            corpus_path=corpus_path,
            items=items,
            unique=unique,
            client=client,
            target=target,
            force=bool(args.force),
            cache_hits=cache_hits,
            cache_misses=cache_misses,
            batch_size=batch_size,
            seed=int(args.seed),
        )
        emit_result(payload if json_mode else _render_preview(payload), json_mode=json_mode)
        return 0

    # Refuse before embedding, not after: an operator who forgot --force should
    # not pay for a full corpus embed to find out.
    if target.exists() and not args.force:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"an atlas already exists at {target}",
            remediation=(
                "re-run with --force to replace it whole (the existing atlas is deleted, "
                "never merged), or pass a different --atlas path"
            ),
        )

    vectors = _embed_all(cache, client, unique, batch_size=batch_size, retries=retries)
    array = np.asarray(vectors, dtype=_VECTOR_DTYPE)

    limitations = [LIVE_EMBEDDING_LIMITATION] if cache_misses else []
    manifest = AtlasManifest(
        schema_version=SCHEMA_VERSION,
        model=client.model,
        endpoint=client.base_url,
        params={
            "corpus": str(corpus_path),
            "corpus_items": len(items),
            "unique_texts": len(unique),
            "batch_size": batch_size,
            "retries": retries,
            "cache_hits": cache_hits,
            "embedded_live": cache_misses,
        },
        seed=int(args.seed),
        limitations=limitations,
        # survey embeds and records; it makes no structural claim (region,
        # direction, boundary) that could fail a stability check, so there is
        # nothing for it to mark uncertain. The reader verbs populate this.
        uncertain=[],
    )
    records = [
        AtlasRecord(
            id=item.id,
            text=item.text,
            vector_index=index_by_text[item.text],
            metadata=dict(item.metadata),
        )
        for item in items
    ]

    try:
        written = write_atlas(manifest, records, array, target, overwrite=bool(args.force))
    except OSError as err:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=f"failed to write the atlas to {target}: {err}",
            remediation="check the target directory is writable and has free space",
        ) from err

    payload: dict[str, object] = {
        "applied": True,
        "corpus": str(corpus_path),
        "atlas": str(written),
        "items": len(items),
        "unique_texts": len(unique),
        "model": client.model,
        "endpoint": client.base_url,
        "vectors": int(array.shape[0]),
        "dimensions": int(array.shape[1]) if array.ndim == 2 else 0,
        "cache_hits": cache_hits,
        "embedded_live": cache_misses,
        "seed": int(args.seed),
        "limitations": limitations,
        REPRODUCED_BY_KEY: reproducing_command(),
    }
    emit_diagnostic(f"survey: wrote atlas to {written}")
    emit_result(payload if json_mode else _render_applied(payload), json_mode=json_mode)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "survey",
        help="Embed a JSONL corpus and write an Atlas (previews by default; --apply writes).",
        description=(
            "Embed a corpus and write the Atlas that every reader verb reads. Previews by "
            "default and writes nothing at all without --apply."
        ),
    )
    p.add_argument(
        "corpus",
        help="Path to the JSONL corpus: one JSON object per line, each with a 'text' field "
        "(and an optional 'id'; every other key is preserved as record metadata).",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Actually embed the corpus and write the atlas. Without this flag the command "
        "only previews and performs no filesystem writes and no backend calls.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="With --apply, replace an existing atlas at the target path. The old atlas is "
        "deleted and the new one written whole — the two are never merged.",
    )
    p.add_argument(
        "--atlas",
        "-o",
        default=str(DEFAULT_ATLAS_DIR),
        help=f"Directory to write the atlas into (default: {DEFAULT_ATLAS_DIR}).",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Number of texts sent to the backend per request (default: {DEFAULT_BATCH_SIZE}). "
        "Each batch reports progress to stderr.",
    )
    p.add_argument(
        "--retries",
        type=int,
        default=DEFAULT_RETRIES,
        help=f"Retries per batch after a transient backend failure (default: {DEFAULT_RETRIES}); "
        "0 disables retrying. A persistent failure still exits 2.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"RNG seed recorded in the atlas manifest (default: {DEFAULT_SEED}). Structural "
        "claims derived from this atlas are only as reproducible as this seed makes them.",
    )
    # Flag names come from the backend constants rather than being respelled
    # here, so the flag, its help text, and the hint the client puts in an
    # unreachable-endpoint error can never drift apart. argparse derives the
    # dests (``embed_url`` / ``embed_model``) from them.
    p.add_argument(ENDPOINT_FLAG, default=None, help=ENDPOINT_HELP)
    p.add_argument(MODEL_FLAG, default=None, help=MODEL_HELP)
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_survey)
