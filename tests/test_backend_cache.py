"""Tests for the home-dir embed cache (embeddings/backends/cache.py).

Covers the t9 acceptance criteria:

* cache entries live under ``~/.embeddings/cache`` keyed by
  ``(model, sha256(text))``, shared machine-wide and never inside a repo;
* each entry is written atomically via ``embeddings/_atomic.py`` -- a killed
  run leaves no unreadable entry;
* a cache hit returns byte-identical vectors to what was stored, and the
  cache path is overridable by constructor argument / environment variable
  for tests.

No test in this module ever touches the real ``$HOME`` -- every test either
passes ``root=tmp_path`` explicitly, or monkeypatches ``HOME`` to a tmp_path
before letting the default resolve.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Callable

import pytest

from embeddings.backends.cache import (
    ENV_CACHE_DIR,
    EmbeddingCache,
    get_or_embed,
    text_digest,
)
from embeddings.backends.client import EmbeddingClient
from embeddings.cli._errors import EXIT_ENV_ERROR, CliError
from tests._fakes import FakeEmbeddingTransport, deterministic_vector


def _wire_transport(
    fake: FakeEmbeddingTransport,
) -> Callable[[str, bytes, dict[str, str]], bytes]:
    """Adapt a ``FakeEmbeddingTransport`` onto the ``Transport`` protocol ``EmbeddingClient`` calls.

    ``FakeEmbeddingTransport.__call__`` takes ``(texts, model=...)`` and
    returns a decoded OpenAI-shaped dict -- the shape its own docstring
    anticipates being handed directly to ``EmbeddingClient(transport=...)``.
    But ``EmbeddingClient.embed`` actually calls its transport as
    ``self.transport(endpoint, payload, headers)`` (JSON-encoded request
    bytes in, raw response bytes out) -- the wire-level ``Transport`` shape
    ``embeddings/backends/client.py`` documents. This adapter bridges the
    two so the fake's determinism and call-recording (``call_count``,
    ``last_call``) still work as documented, without editing either frozen
    module.
    """

    def _transport(url: str, data: bytes, headers: dict[str, str]) -> bytes:
        del url, headers
        body = json.loads(data)
        response = fake.embed(body["input"], model=body.get("model", "fake-embedding-model"))
        return json.dumps(response).encode("utf-8")

    return _transport


# --- cache root resolution: default / env / explicit precedence -------------


def test_default_root_resolves_under_home_when_no_env_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv(ENV_CACHE_DIR, raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    cache = EmbeddingCache()
    assert cache.root == tmp_path / ".embeddings" / "cache"


def test_env_var_overrides_default(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    override = tmp_path / "elsewhere" / "cache"
    monkeypatch.setenv(ENV_CACHE_DIR, str(override))
    cache = EmbeddingCache()
    assert cache.root == override


def test_explicit_root_beats_env_var(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_CACHE_DIR, str(tmp_path / "from-env"))
    explicit = tmp_path / "from-arg"
    cache = EmbeddingCache(root=explicit)
    assert cache.root == explicit


def test_env_var_name_follows_embeddings_convention() -> None:
    # Matches the EMBEDDINGS_EMBED_URL / EMBEDDINGS_EMBED_MODEL naming already
    # established in embeddings/backends/client.py.
    assert ENV_CACHE_DIR == "EMBEDDINGS_CACHE_DIR"


# --- text_digest --------------------------------------------------------------


def test_text_digest_is_sha256_hex() -> None:
    assert text_digest("hello") == hashlib.sha256(b"hello").hexdigest()


def test_text_digest_differs_for_different_text() -> None:
    assert text_digest("hello") != text_digest("world")


# --- path_for: keyed by (model, sha256(text)), sharded ------------------------


def test_path_for_is_sharded_by_text_hash_prefix(tmp_path: Path) -> None:
    cache = EmbeddingCache(root=tmp_path)
    digest = text_digest("some corpus text")
    path = cache.path_for("test-model", "some corpus text")
    assert path.parent.name == digest[:2]
    assert digest in path.name


def test_path_for_different_models_produce_different_paths(tmp_path: Path) -> None:
    cache = EmbeddingCache(root=tmp_path)
    path_a = cache.path_for("model-a", "same text")
    path_b = cache.path_for("model-b", "same text")
    assert path_a != path_b


def test_path_for_model_with_slashes_yields_safe_single_level_dirname(tmp_path: Path) -> None:
    cache = EmbeddingCache(root=tmp_path)
    path = cache.path_for("Qwen/Qwen3-Embedding-0.6B", "hello")
    relative_parts = path.relative_to(tmp_path).parts
    # root / model_dir / shard / filename -- exactly 3 components, and none of
    # them (particularly the sanitized model dirname) contains a literal '/'.
    assert len(relative_parts) == 3
    for part in relative_parts:
        assert "/" not in part


def test_path_for_two_different_models_sanitizing_to_the_same_text_do_not_collide(
    tmp_path: Path,
) -> None:
    """Sanitization alone must not be able to merge two distinct model ids."""
    cache = EmbeddingCache(root=tmp_path)
    # These two raw strings sanitize to the same characters (both '/' and '_'
    # are replaced/kept in ways that could coincide) -- the digest suffix must
    # still keep them apart.
    path_a = cache.path_for("a/b", "text")
    path_b = cache.path_for("a_b", "text")
    assert path_a.parent.parent != path_b.parent.parent


# --- put / get: byte-identical round trip -------------------------------------


def test_put_then_get_round_trips_exactly(tmp_path: Path) -> None:
    cache = EmbeddingCache(root=tmp_path)
    vector = deterministic_vector("hello world", dim=16)

    cache.put("test-model", "hello world", vector)
    result = cache.get("test-model", "hello world")

    assert result == vector


def test_round_trip_preserves_exact_float_bit_patterns_including_edge_values(
    tmp_path: Path,
) -> None:
    """Edge-case floats (zero, negative zero, inf, extreme magnitudes) survive exactly."""
    cache = EmbeddingCache(root=tmp_path)
    vector = [0.0, -0.0, 1.0, -1.0, 3.141592653589793, 1e308, -1e-308, float("inf"), float("-inf")]

    cache.put("test-model", "edge-case text", vector)
    result = cache.get("test-model", "edge-case text")

    assert result == vector
    # -0.0 == 0.0 under == ; assert the sign bit round-tripped too via repr.
    assert [repr(v) for v in result] == [repr(v) for v in vector]


def test_get_returns_none_for_missing_entry(tmp_path: Path) -> None:
    cache = EmbeddingCache(root=tmp_path)
    assert cache.get("test-model", "never embedded") is None


def test_get_is_keyed_by_model_not_just_text(tmp_path: Path) -> None:
    cache = EmbeddingCache(root=tmp_path)
    cache.put("model-a", "shared text", [1.0, 2.0, 3.0])
    assert cache.get("model-b", "shared text") is None
    assert cache.get("model-a", "shared text") == [1.0, 2.0, 3.0]


def test_put_overwrites_existing_entry(tmp_path: Path) -> None:
    cache = EmbeddingCache(root=tmp_path)
    cache.put("test-model", "text", [1.0, 2.0])
    cache.put("test-model", "text", [3.0, 4.0])
    assert cache.get("test-model", "text") == [3.0, 4.0]


def test_cache_entries_never_written_outside_configured_root(tmp_path: Path) -> None:
    cache = EmbeddingCache(root=tmp_path)
    cache.put("test-model", "text", [1.0, 2.0])
    written = list(tmp_path.rglob("*"))
    assert written  # something was written
    for path in written:
        assert path.is_relative_to(tmp_path)


# --- interrupted write: no unreadable entry -----------------------------------


def test_interrupted_write_leaves_no_unreadable_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash mid-write (simulated via a failing fsync) must not leave a corrupt entry.

    embeddings/_atomic.py's write_atomic already guarantees this at the
    file level (tested in tests/test_atomic.py); this test asserts the cache
    built on top of it inherits the guarantee end to end.
    """
    cache = EmbeddingCache(root=tmp_path)

    def _boom(_fd: int) -> None:
        raise OSError("simulated kill mid-write")

    monkeypatch.setattr(os, "fsync", _boom)

    with pytest.raises(OSError):
        cache.put("test-model", "text that never lands", [1.0, 2.0, 3.0])

    # The killed run must leave no entry at all -- not a truncated/corrupt one.
    path = cache.path_for("test-model", "text that never lands")
    assert not path.exists()
    assert cache.get("test-model", "text that never lands") is None
    # No stray temp file left behind in the shard directory either.
    assert list(path.parent.iterdir()) == []


def test_interrupted_write_does_not_disturb_a_previously_cached_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = EmbeddingCache(root=tmp_path)
    cache.put("test-model", "text", [1.0, 2.0, 3.0])

    def _boom(_fd: int) -> None:
        raise OSError("simulated kill mid-overwrite")

    monkeypatch.setattr(os, "fsync", _boom)
    with pytest.raises(OSError):
        cache.put("test-model", "text", [9.0, 9.0, 9.0])

    assert cache.get("test-model", "text") == [1.0, 2.0, 3.0]


# --- get_or_embed: cache hits served, only misses sent to the client ---------


def test_get_or_embed_serves_cache_hits_without_calling_transport(tmp_path: Path) -> None:
    cache = EmbeddingCache(root=tmp_path)
    transport = FakeEmbeddingTransport(dim=8)
    client = EmbeddingClient(model="test-model", transport=_wire_transport(transport))

    # Pin a cached vector that differs from what the fake transport would
    # produce, so a correct implementation is provably serving the cached
    # value rather than a fresh embed.
    cached_vector = [9.0] * 8
    cache.put("test-model", "already cached", cached_vector)

    result = get_or_embed(cache, client, ["already cached"])

    assert result == [cached_vector]
    assert transport.call_count == 0


def test_get_or_embed_sends_only_misses_to_the_client(tmp_path: Path) -> None:
    cache = EmbeddingCache(root=tmp_path)
    transport = FakeEmbeddingTransport(dim=8)
    client = EmbeddingClient(model="test-model", transport=_wire_transport(transport))

    cache.put("test-model", "cached one", [1.0] * 8)

    result = get_or_embed(cache, client, ["cached one", "new one"])

    assert result[0] == [1.0] * 8
    assert result[1] == transport.vector_for("new one")
    assert transport.call_count == 1
    assert transport.last_call is not None
    assert transport.last_call.texts == ["new one"]


def test_get_or_embed_writes_new_vectors_into_the_cache(tmp_path: Path) -> None:
    cache = EmbeddingCache(root=tmp_path)
    transport = FakeEmbeddingTransport(dim=8)
    client = EmbeddingClient(model="test-model", transport=_wire_transport(transport))

    get_or_embed(cache, client, ["brand new text"])

    # Second call must not need the transport again.
    transport.reset()
    result = get_or_embed(cache, client, ["brand new text"])
    assert transport.call_count == 0
    assert result == [transport.vector_for("brand new text")]


def test_get_or_embed_dedupes_repeated_misses_into_a_single_transport_call(
    tmp_path: Path,
) -> None:
    cache = EmbeddingCache(root=tmp_path)
    transport = FakeEmbeddingTransport(dim=8)
    client = EmbeddingClient(model="test-model", transport=_wire_transport(transport))

    result = get_or_embed(cache, client, ["dup", "dup", "dup"])

    assert transport.call_count == 1
    assert transport.last_call is not None
    assert transport.last_call.texts == ["dup"]
    expected = transport.vector_for("dup")
    assert result == [expected, expected, expected]


def test_get_or_embed_preserves_input_order_with_mixed_hits_and_misses(
    tmp_path: Path,
) -> None:
    cache = EmbeddingCache(root=tmp_path)
    transport = FakeEmbeddingTransport(dim=8)
    client = EmbeddingClient(model="test-model", transport=_wire_transport(transport))

    cache.put("test-model", "b", [2.0] * 8)
    cache.put("test-model", "d", [4.0] * 8)

    result = get_or_embed(cache, client, ["a", "b", "c", "d"])

    assert result[1] == [2.0] * 8
    assert result[3] == [4.0] * 8
    assert result[0] == transport.vector_for("a")
    assert result[2] == transport.vector_for("c")
    # Only the misses ("a", "c") were sent to the transport.
    assert transport.call_count == 1
    assert transport.last_call is not None
    assert sorted(transport.last_call.texts) == ["a", "c"]


def test_get_or_embed_empty_input_makes_no_transport_call(tmp_path: Path) -> None:
    cache = EmbeddingCache(root=tmp_path)
    transport = FakeEmbeddingTransport(dim=8)
    client = EmbeddingClient(model="test-model", transport=_wire_transport(transport))

    result = get_or_embed(cache, client, [])

    assert result == []
    assert transport.call_count == 0


# --- the miscount guard: a bad response must never poison the cache ---------
#
# ``EmbeddingClient.embed`` enforces "one vector per input" itself, so this is
# belt-and-braces against a *different* client implementation (or a future
# refactor) that doesn't. It matters because the cache is durable: a single
# misaligned write would keep returning the wrong vector for that text on every
# later run, long after the misbehaving backend was fixed.


class _MiscountingClient:
    """Stand-in client that returns the wrong number of vectors, no validation."""

    def __init__(self, vectors: list[list[float]], model: str = "test-model") -> None:
        self.vectors = vectors
        self.model = model
        self.calls = 0

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        return self.vectors


def test_get_or_embed_short_response_raises_cli_error_code_2(tmp_path: Path) -> None:
    cache = EmbeddingCache(root=tmp_path)
    client = _MiscountingClient([[1.0, 2.0]])

    with pytest.raises(CliError) as exc_info:
        get_or_embed(cache, client, ["a", "b"])  # type: ignore[arg-type]

    assert exc_info.value.code == EXIT_ENV_ERROR


def test_get_or_embed_long_response_raises_cli_error_code_2(tmp_path: Path) -> None:
    cache = EmbeddingCache(root=tmp_path)
    client = _MiscountingClient([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(CliError) as exc_info:
        get_or_embed(cache, client, ["a"])  # type: ignore[arg-type]

    assert exc_info.value.code == EXIT_ENV_ERROR


def test_get_or_embed_miscount_writes_no_cache_entry(tmp_path: Path) -> None:
    """The check happens *before* the first put — nothing lands on disk."""
    cache = EmbeddingCache(root=tmp_path)
    client = _MiscountingClient([[1.0, 2.0]])

    with pytest.raises(CliError):
        get_or_embed(cache, client, ["a", "b"])  # type: ignore[arg-type]

    assert [p for p in tmp_path.rglob("*") if p.is_file()] == []
    assert cache.get("test-model", "a") is None
    assert cache.get("test-model", "b") is None
