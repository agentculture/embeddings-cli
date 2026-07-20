"""Embedding backend clients.

:mod:`embeddings.backends.client` is the OpenAI-compatible ``/v1/embeddings``
client used by every embedding-touching verb; :mod:`embeddings.backends.cache`
is the home-dir cache that sits in front of it. Both are re-exported here so
callers write ``from embeddings.backends import EmbeddingClient,
EmbeddingCache`` rather than reaching into the submodules.
"""

from __future__ import annotations

from embeddings.backends.cache import ENV_CACHE_DIR, EmbeddingCache, get_or_embed, text_digest
from embeddings.backends.client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    ENDPOINT_FLAG,
    ENDPOINT_HELP,
    ENV_BASE_URL,
    ENV_MODEL,
    EmbeddingClient,
    Transport,
    urllib_transport,
)

__all__ = [
    "DEFAULT_BASE_URL",
    "DEFAULT_MODEL",
    "ENDPOINT_FLAG",
    "ENDPOINT_HELP",
    "ENV_BASE_URL",
    "ENV_CACHE_DIR",
    "ENV_MODEL",
    "EmbeddingCache",
    "EmbeddingClient",
    "Transport",
    "get_or_embed",
    "text_digest",
    "urllib_transport",
]
