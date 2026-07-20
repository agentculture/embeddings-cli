"""Home-dir, machine-wide embed cache keyed by ``(model, sha256(text))``.

Why a home-dir cache, not a repo-local one
-------------------------------------------
This is a bulk-embedding tool: a survey re-embeds a whole corpus every time a
clustering parameter changes unless prior results are reused. The cache is
what makes that affordable, and it is deliberately a **machine-wide home-dir
store** rather than something living inside a repo: a study for one corpus and
a study for another, in two unrelated repos, both call the same embedding
model on overlapping text, and both should get the benefit. Living outside any
repo also means the cache is never committed and never shows up in ``git
status`` -- it is operator machine state, not project state.

Cache key and layout
---------------------
Entries are keyed by ``(model, sha256(text))``. On disk that becomes::

    <root>/<model-dirname>/<digest[:2]>/<digest>.f64

* ``<root>`` defaults to ``~/.embeddings/cache`` (see :data:`ENV_CACHE_DIR`
  for the override).
* ``<model-dirname>`` is a filesystem-safe rendering of the model identifier
  (see :func:`_model_dirname` -- model ids routinely contain ``/``, e.g.
  ``"Qwen/Qwen3-Embedding-0.6B"``, which would otherwise be read as a path
  separator).
* ``<digest>`` is ``sha256(text)`` as lowercase hex (:func:`text_digest`); the
  entry is additionally sharded under its own first two hex characters so a
  large corpus does not put a million files in one directory.

Storage format and the byte-identical guarantee
--------------------------------------------------
Each entry is the vector's raw bytes: little-endian float64 (numpy dtype
``"<f8"``), no header, no length prefix. The dimension is not stored
separately -- it is recovered on read as ``len(data) // 8``, since every
entry holds exactly one flat vector. Python's ``float`` already *is* an IEEE
754 double, so encoding via ``numpy.asarray(vector, dtype="<f8").tobytes()``
copies each element's existing bit pattern rather than reformatting it
through text, and decoding via ``numpy.frombuffer(data, dtype="<f8")`` is the
exact inverse. This is what makes a cache hit byte-identical to what was
stored, including edge values (``-0.0``, ``inf``) that a text/JSON-repr
format would risk mangling. The explicit ``<`` (little-endian) byte order
means an entry written on one host reads back identically on another,
regardless of host native endianness (in practice always little-endian on
the platforms this CLI targets, but the format does not silently depend on
that).

Atomicity
---------
Every write goes through :func:`embeddings._atomic.write_atomic`, which
writes to a sibling temp file, fsyncs it, and ``os.replace``s it onto the
target. A killed run -- SIGKILL, OOM, an uncaught exception mid-write --
leaves either the previous complete entry or no entry at all; it can never
leave a truncated, unreadable ``.f64`` file for a later run to trip over.

Overriding the cache path
---------------------------
Precedence matches :mod:`embeddings.backends.client`'s
``EMBEDDINGS_EMBED_URL`` / ``EMBEDDINGS_EMBED_MODEL`` convention: an explicit
``EmbeddingCache(root=...)`` argument beats the :data:`ENV_CACHE_DIR`
environment variable, which beats the ``~/.embeddings/cache`` built-in
default. Tests must always pass ``root=tmp_path`` (or set the env var to a
tmp_path) -- never let a test resolve the real ``$HOME``.
"""

from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np

from embeddings._atomic import write_atomic
from embeddings.backends.client import EmbeddingClient

#: Environment variable that overrides the cache root, following the
#: EMBEDDINGS_* naming convention established by
#: embeddings.backends.client.ENV_BASE_URL / ENV_MODEL.
ENV_CACHE_DIR = "EMBEDDINGS_CACHE_DIR"

# Little-endian float64: fixes both dtype and endianness so an entry written
# on one host is byte-identical when read on another. See the module
# docstring's "Storage format" section.
_DTYPE = np.dtype("<f8")

# Conservative filesystem-safe allow-list for the sanitized half of a model
# directory name. Everything else (notably '/') becomes '_'.
_UNSAFE_PATH_CHARS = re.compile(r"[^A-Za-z0-9._-]")

#: File extension for a cache entry: raw little-endian float64 bytes.
_ENTRY_SUFFIX = ".f64"


def _home_cache_dir() -> Path:
    """The built-in default cache root: ``~/.embeddings/cache``.

    Resolved lazily (never baked in at import time) so tests can point
    ``$HOME`` at a tmp_path and observe the default follow it, without this
    module ever touching the real home directory itself.
    """
    return Path.home() / ".embeddings" / "cache"


def _default_root() -> Path:
    """Default-factory for :attr:`EmbeddingCache.root`: env var, else the home default."""
    override = os.environ.get(ENV_CACHE_DIR)
    return Path(override) if override else _home_cache_dir()


def _model_dirname(model: str) -> str:
    """Filesystem-safe, collision-resistant directory name for a model identifier.

    Model identifiers routinely contain ``/`` (e.g.
    ``"Qwen/Qwen3-Embedding-0.6B"``), which would otherwise be read as a path
    separator. Every character outside a conservative allow-list is replaced
    with ``_`` for readability, and a short hash of the *original* string is
    appended so two distinct model ids that happen to sanitize to the same
    text (e.g. ``"a/b"`` and ``"a_b"``) still land in different directories.
    """
    sanitized = _UNSAFE_PATH_CHARS.sub("_", model)
    digest = hashlib.sha256(model.encode("utf-8")).hexdigest()[:8]
    return f"{sanitized}-{digest}"


def text_digest(text: str) -> str:
    """``sha256(text)`` as lowercase hex -- the per-text half of the cache key."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _encode_vector(vector: Sequence[float]) -> bytes:
    return np.asarray(vector, dtype=_DTYPE).tobytes()


def _decode_vector(data: bytes) -> list[float]:
    return np.frombuffer(data, dtype=_DTYPE).tolist()


@dataclass
class EmbeddingCache:
    """A content-addressed, on-disk cache of embedding vectors.

    Keyed by ``(model, sha256(text))``; see the module docstring for the
    on-disk layout, storage format, and atomicity guarantees. ``root``
    defaults to ``~/.embeddings/cache`` (overridable via
    :data:`ENV_CACHE_DIR`); pass it explicitly (always ``tmp_path`` in tests)
    to pin a different location.
    """

    root: Path = field(default_factory=_default_root)

    def __post_init__(self) -> None:
        self.root = Path(self.root)

    def path_for(self, model: str, text: str) -> Path:
        """The on-disk path a ``(model, text)`` entry would live at, hit or not."""
        digest = text_digest(text)
        shard = digest[:2]
        return self.root / _model_dirname(model) / shard / f"{digest}{_ENTRY_SUFFIX}"

    def get(self, model: str, text: str) -> list[float] | None:
        """Return the cached vector for ``(model, text)``, or ``None`` on a miss.

        A miss and a killed-write leftover are indistinguishable from the
        caller's point of view by design: :func:`embeddings._atomic.write_atomic`
        guarantees there is never a partial entry to misread in the first
        place, so any file present at the target path is always a complete,
        valid entry.
        """
        try:
            data = self.path_for(model, text).read_bytes()
        except FileNotFoundError:
            return None
        return _decode_vector(data)

    def put(self, model: str, text: str, vector: Sequence[float]) -> None:
        """Store ``vector`` for ``(model, text)``, atomically overwriting any prior entry."""
        write_atomic(self.path_for(model, text), _encode_vector(vector))


def get_or_embed(
    cache: EmbeddingCache, client: EmbeddingClient, texts: Sequence[str]
) -> list[list[float]]:
    """Return one vector per ``texts`` entry, serving cache hits and embedding only misses.

    Cached under ``client.model`` -- the model a vector was produced by is
    exactly the model whose cache entries it is safe to reuse. Duplicate
    texts within a single call are deduplicated before hitting the client, so
    ``get_or_embed(cache, client, ["x", "x", "x"])`` sends ``"x"`` to the
    client at most once even on an empty cache. Every freshly embedded vector
    is written back into ``cache`` before this function returns, so a
    subsequent call with the same texts is served entirely from cache.

    This is the pure :meth:`EmbeddingCache.get` / :meth:`EmbeddingCache.put`
    API's counterpart for bulk callers (``survey``, in a later task); the
    pure API stays independently testable, and this helper is the
    integration point that saves a looping caller from re-deriving the
    hit/miss/dedupe bookkeeping itself.
    """
    materialized = list(texts)
    results: list[list[float]] = [[] for _ in materialized]
    pending: dict[str, list[int]] = {}

    for index, text in enumerate(materialized):
        cached = cache.get(client.model, text)
        if cached is not None:
            results[index] = cached
        else:
            pending.setdefault(text, []).append(index)

    if pending:
        miss_texts = list(pending)
        vectors = client.embed(miss_texts)
        for text, vector in zip(miss_texts, vectors):
            materialized_vector = list(vector)
            cache.put(client.model, text, materialized_vector)
            for index in pending[text]:
                results[index] = materialized_vector

    return results
