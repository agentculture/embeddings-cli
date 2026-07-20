"""Hermetic test doubles for embeddings-cli's embedding backend.

Why this module exists
-----------------------
CI has no ``lobes`` embedding gateway and never will — it is an actor every
test in this suite must serve, not fight. Nothing in ``tests/`` may open a
real socket to an embedding endpoint, yet a growing set of verbs (``survey``,
``neighbors``, ``directions discover``, ...) exist only to talk to one. This
module is the shared fixture that lets those tests exist anyway: a fake
transport shaped like the real OpenAI-compatible wire contract
(``POST /v1/embeddings``), plus a reserved dead port and a network guard for
the one place a test legitimately does open a socket — proving an unreachable
endpoint is handled without a traceback.

Determinism is the whole point: :func:`deterministic_vector` derives a
vector purely from the input text (via a sha256-seeded stdlib PRNG), so the
same text always yields the same vector, in this process or any other, with
or without an explicit :class:`FakeEmbeddingTransport` instance. That means
two unrelated tests can each build their own transport and still agree on
what "the vector for 'cat'" is, and :meth:`FakeEmbeddingTransport.set_override`
lets a test pin exact vectors where hash-derived placement isn't precise
enough (e.g. asserting "cat" and "kitten" land as nearest neighbors).

This module intentionally has zero runtime dependencies beyond the stdlib.
numpy is landing as embeddings-cli's first runtime dependency in a sibling
task and may or may not be present in any given worktree or install; nothing
here needs it.

Naming note: this file starts with an underscore, so pytest's
``testpaths = ["tests"]`` does not collect it as a test module (it does not
match the ``test_*.py`` pattern) — that is intentional, it is a helper
library, not a test file. Import from it directly, e.g.::

    from tests._fakes import DEAD_PORT, FakeEmbeddingTransport, blocked_sockets

Fixtures defined here (``fake_transport``, ``no_sockets``) are plain
``@pytest.fixture`` functions with no ``conftest.py`` registration; pytest
resolves fixtures by name lookup in the importing test module's namespace, so
``from tests._fakes import fake_transport`` followed by a test parameter named
``fake_transport`` works without any extra wiring.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import socket
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import pytest

# --------------------------------------------------------------------------
# Reserved dead port — for tests that must prove an *unreachable* endpoint is
# handled without a traceback. Deliberately NOT 8001 (the real lobes gateway
# port defaulted to by embeddings/backends/client.py) so a copy-paste mistake
# can never accidentally target a real, possibly-running service.
# --------------------------------------------------------------------------

# IANA dynamic/private ("ephemeral") port range is 49152-65535 (RFC 6335 §6);
# nothing in this codebase, CI, or a dev sandbox binds a *listening* server
# there, so connecting to it on loopback reliably fails fast (connection
# refused) rather than hanging.
#
# A second "reserved dead port" convention exists in tests/test_backend_client.py
# (port 1, TCPMUX) rather than this one. Deliberately, not by drift: that
# module picked port 1 because *binding* it requires root, which is a
# stronger "nothing will ever listen here" guarantee than "nothing happens to
# bind here today" — worth it for the one test in this suite that opens a
# real socket against the real urllib transport. DEAD_PORT here optimises for
# legibility (49213 reads as "nowhere real" without needing the root-binding
# footnote) for the tests that only need an endpoint that is *currently*
# unreachable, not one that is unreachable *by construction*. Both are
# correct for what they guard; unify only if a change can drop one without
# weakening either test's guarantee.
DEAD_PORT = 49213
DEAD_HOST = "127.0.0.1"


def dead_endpoint(path: str = "/v1") -> str:
    """Build a URL to :data:`DEAD_PORT` for exit-2 "unreachable backend" tests.

    Example: ``dead_endpoint()`` -> ``"http://127.0.0.1:49213/v1"``.
    """
    return f"http://{DEAD_HOST}:{DEAD_PORT}{path}"


# --------------------------------------------------------------------------
# Deterministic, hash-derived vectors
# --------------------------------------------------------------------------


def unit_normalize(vector: Sequence[float]) -> list[float]:
    """Scale ``vector`` to unit length; a zero vector is returned unchanged."""
    magnitude = math.sqrt(sum(component * component for component in vector))
    if magnitude == 0.0:
        return list(vector)
    return [component / magnitude for component in vector]


def deterministic_vector(text: str, dim: int = 8, *, normalize: bool = True) -> list[float]:
    """Derive a deterministic vector for ``text``: same text, same dim -> same vector.

    Always, in this process or any other -- the derivation is a pure function
    of ``text`` and ``dim`` (sha256(text) seeds a stdlib ``random.Random``),
    with no hidden instance state. That is what lets independently-constructed
    :class:`FakeEmbeddingTransport` instances across different test files
    agree on "the vector for this text" without coordinating.

    Set ``normalize=False`` to get the raw, non-unit-length vector.
    """
    if dim <= 0:
        raise ValueError(f"dim must be a positive integer, got {dim!r}")
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big")
    rng = random.Random(seed)
    vector = [rng.uniform(-1.0, 1.0) for _ in range(dim)]
    return unit_normalize(vector) if normalize else vector


# --------------------------------------------------------------------------
# The fake transport itself
# --------------------------------------------------------------------------


@dataclass
class FakeCall:
    """One recorded call to a :class:`FakeEmbeddingTransport` or :class:`UnreachableTransport`."""

    texts: list[str]
    model: str


class FakeEmbeddingTransport:
    """Deterministic, socket-free stand-in for an OpenAI-compatible embeddings transport.

    Shaped like the wire contract of ``POST /v1/embeddings``: call it with a
    list of texts and a model name, get back
    ``{"data": [{"embedding": [...]}, ...]}``. No network is ever touched --
    every vector comes from :func:`deterministic_vector`, optionally pinned
    per-text via :meth:`set_override` so a test can hand-place items into a
    known cluster or neighborhood.

    Designed to be the injectable transport behind an embedding client (a
    sibling task is building one as ``embeddings/backends/client.py`` in the
    same wave this was written in, targeting exactly this shape) -- wiring it
    in is expected to be "construct one, pass it in place of the real HTTP
    transport", nothing more::

        transport = FakeEmbeddingTransport(dim=16)
        client = EmbeddingClient(transport=transport, endpoint=..., model="fake")
        client.embed(["hello", "world"])

    It is also usable directly, matching the raw OpenAI-compatible shape::

        transport = FakeEmbeddingTransport()
        response = transport(["hello", "world"], model="fake-embedding-model")
        response["data"][0]["embedding"]  # -> list[float] of length `dim`
    """

    def __init__(
        self,
        dim: int = 8,
        *,
        overrides: Mapping[str, Sequence[float]] | None = None,
        normalize: bool = True,
    ) -> None:
        if dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim!r}")
        self.dim = dim
        self.normalize = normalize
        self._overrides: dict[str, list[float]] = {}
        if overrides:
            for text, vector in overrides.items():
                self.set_override(text, vector)
        self.calls: list[FakeCall] = []

    def set_override(self, text: str, vector: Sequence[float]) -> None:
        """Pin the exact vector returned for ``text``, overriding the hash-derived default.

        Use this to make specific test texts exact neighbors, exact
        opposites, or exact cluster-mates instead of relying on the
        hash-derived placement (which is deterministic but not otherwise
        controllable). ``vector`` must have length ``self.dim``.
        """
        materialized = list(vector)
        if len(materialized) != self.dim:
            raise ValueError(
                f"override for {text!r} has length {len(materialized)}, expected dim={self.dim}"
            )
        self._overrides[text] = materialized

    def vector_for(self, text: str) -> list[float]:
        """The vector this transport returns for ``text`` -- override if set, else hash-derived."""
        if text in self._overrides:
            return list(self._overrides[text])
        return deterministic_vector(text, self.dim, normalize=self.normalize)

    def embed(self, texts: Sequence[str], model: str = "fake-embedding-model") -> dict:
        """Return an OpenAI-compatible embeddings response for ``texts``. Opens no socket."""
        materialized_texts = list(texts)
        self.calls.append(FakeCall(texts=materialized_texts, model=model))
        data = [
            {"object": "embedding", "index": index, "embedding": self.vector_for(text)}
            for index, text in enumerate(materialized_texts)
        ]
        return {
            "object": "list",
            "data": data,
            "model": model,
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }

    __call__ = embed

    @property
    def call_count(self) -> int:
        return len(self.calls)

    @property
    def last_call(self) -> FakeCall | None:
        return self.calls[-1] if self.calls else None

    def reset(self) -> None:
        """Clear recorded calls. Vectors/overrides are untouched -- still deterministic."""
        self.calls.clear()


class UnreachableTransport:
    """A transport double that always raises, simulating a down embedding backend.

    For unit-testing a CLI command's error handling ("the backend is
    unreachable") without touching a socket at all. This is *not* the tool
    for the integration-level "an unreachable endpoint exits 2" test that
    must prove the real HTTP client handles a real dead connection --
    :data:`DEAD_PORT` / :func:`dead_endpoint` plus a real client is the one
    sanctioned place in this codebase for a test to open a socket, and it
    must always resolve to a reserved, nothing-listening port, never the real
    gateway port.
    """

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error or ConnectionRefusedError("simulated: embedding backend unreachable")
        self.calls: list[FakeCall] = []

    def __call__(self, texts: Sequence[str], model: str = "fake-embedding-model") -> dict:
        self.calls.append(FakeCall(texts=list(texts), model=model))
        raise self.error


# --------------------------------------------------------------------------
# Wire-level adapter -- plug a fake into the real EmbeddingClient
# --------------------------------------------------------------------------


def wire_transport(fake: FakeEmbeddingTransport | UnreachableTransport) -> Any:
    """Adapt a text-level fake to the wire-level ``Transport`` protocol.

    The fakes above speak the *semantic* contract (texts in, response dict
    out), which is what makes them convenient to assert against.
    :class:`embeddings.backends.client.EmbeddingClient` injects at the
    *transport* layer instead -- ``(url, data: bytes, headers) -> bytes`` --
    so the two do not compose directly. This bridges them: it decodes the
    request body the client actually sends, replays it through the fake, and
    re-encodes the response.

    Use this whenever a test needs a real ``EmbeddingClient`` driven by a fake
    backend. Without it, passing a fake straight to ``transport=`` raises
    ``TypeError`` at call time.
    """

    def _transport(url: str, data: bytes, headers: dict[str, str]) -> bytes:
        payload = json.loads(data.decode("utf-8"))
        texts = payload.get("input", [])
        model = payload.get("model", "fake-embedding-model")
        return json.dumps(fake(texts, model)).encode("utf-8")

    return _transport


# --------------------------------------------------------------------------
# Network guard -- assert NO real socket connection was attempted
# --------------------------------------------------------------------------


class NetworkBlockedError(AssertionError):
    """Raised in place of a real socket connection while :func:`blocked_sockets` is active."""


@contextmanager
def blocked_sockets() -> Iterator[None]:
    """Context manager: any real TCP ``connect()`` inside it raises :class:`NetworkBlockedError`.

    Wrap code under test in this to assert it never reaches past an injected
    fake transport to a real socket -- the positive proof that pairs with
    :class:`FakeEmbeddingTransport` never touching one in the first place.
    Only ``socket.socket.connect`` is patched (not socket construction), so
    unrelated stdlib/pytest machinery that merely creates socket objects
    without connecting them is unaffected.

    Usage::

        def test_client_never_touches_a_real_socket(fake_transport):
            with blocked_sockets():
                result = my_function_under_test(transport=fake_transport)
            assert result  # the fake served it; no real connection was attempted
    """
    real_connect = socket.socket.connect

    def _blocked_connect(self: socket.socket, address: object, *args: object) -> None:
        raise NetworkBlockedError(
            f"blocked_sockets(): refused a real socket connect() to {address!r} -- "
            "inject FakeEmbeddingTransport (or UnreachableTransport) instead of "
            "reaching the network in tests"
        )

    socket.socket.connect = _blocked_connect  # type: ignore[method-assign]
    try:
        yield
    finally:
        socket.socket.connect = real_connect  # type: ignore[method-assign]


@pytest.fixture
def no_sockets() -> Iterator[None]:
    """Pytest-fixture form of :func:`blocked_sockets`, for ``def test_x(no_sockets): ...``."""
    with blocked_sockets():
        yield


@pytest.fixture
def fake_transport() -> FakeEmbeddingTransport:
    """A fresh default-dim :class:`FakeEmbeddingTransport`, importable as a fixture by name."""
    return FakeEmbeddingTransport()
