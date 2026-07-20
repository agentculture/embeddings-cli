"""Embedding backend clients.

:mod:`embeddings.backends.client` is the only module here today — the
OpenAI-compatible ``/v1/embeddings`` client used by every embedding-touching
verb. Re-exported here so callers write ``from embeddings.backends import
EmbeddingClient`` rather than reaching into the submodule.
"""

from __future__ import annotations

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
    "ENV_MODEL",
    "EmbeddingClient",
    "Transport",
    "urllib_transport",
]
