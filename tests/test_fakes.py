"""Tests for the hermetic fake-transport harness (tests/_fakes.py).

These prove the two properties every later verb test will rely on:

1. determinism -- the same text always yields the same vector, independent
   of which :class:`FakeEmbeddingTransport` instance (or none at all) asked
   for it;
2. hermeticity -- the fake never opens a real socket, and :func:`blocked_sockets`
   actually catches a real connection attempt when one is made on purpose.

It also proves :data:`DEAD_PORT` is genuinely dead on this machine right now,
which is exactly what CI re-proves on every PR per the plan's acceptance
criterion for this task.
"""

from __future__ import annotations

import math
import socket

import pytest

from tests._fakes import fake_transport  # noqa: F401 -- imported for pytest fixture name resolution
from tests._fakes import no_sockets  # noqa: F401 -- imported for pytest fixture name resolution
from tests._fakes import (
    DEAD_HOST,
    DEAD_PORT,
    FakeCall,
    FakeEmbeddingTransport,
    NetworkBlockedError,
    UnreachableTransport,
    blocked_sockets,
    dead_endpoint,
    deterministic_vector,
    unit_normalize,
)

# --- deterministic_vector --------------------------------------------------


def test_deterministic_vector_is_stable_across_calls() -> None:
    first = deterministic_vector("hello world")
    second = deterministic_vector("hello world")
    assert first == second


def test_deterministic_vector_differs_for_different_text() -> None:
    assert deterministic_vector("cat") != deterministic_vector("dog")


def test_deterministic_vector_respects_dim() -> None:
    assert len(deterministic_vector("hello", dim=16)) == 16
    assert len(deterministic_vector("hello", dim=3)) == 3


def test_deterministic_vector_normalized_is_unit_length() -> None:
    vector = deterministic_vector("hello", dim=32, normalize=True)
    magnitude = math.sqrt(sum(component * component for component in vector))
    assert magnitude == pytest.approx(1.0)


def test_deterministic_vector_unnormalized_is_not_forced_unit() -> None:
    vector = deterministic_vector("hello", dim=32, normalize=False)
    magnitude = math.sqrt(sum(component * component for component in vector))
    assert magnitude != pytest.approx(1.0)


def test_deterministic_vector_rejects_non_positive_dim() -> None:
    with pytest.raises(ValueError):
        deterministic_vector("hello", dim=0)


def test_unit_normalize_zero_vector_unchanged() -> None:
    assert unit_normalize([0.0, 0.0, 0.0]) == [0.0, 0.0, 0.0]


def test_unit_normalize_scales_to_length_one() -> None:
    normalized = unit_normalize([3.0, 4.0])
    assert normalized == pytest.approx([0.6, 0.8])


# --- FakeEmbeddingTransport -------------------------------------------------


def test_transport_default_construction_uses_dim_eight() -> None:
    transport = FakeEmbeddingTransport()
    assert transport.dim == 8


def test_transport_rejects_non_positive_dim() -> None:
    with pytest.raises(ValueError):
        FakeEmbeddingTransport(dim=0)


def test_transport_is_deterministic_across_independent_instances() -> None:
    """Two unrelated transports must agree on "the vector for this text"."""
    first = FakeEmbeddingTransport(dim=12)
    second = FakeEmbeddingTransport(dim=12)
    assert first.vector_for("shared text") == second.vector_for("shared text")


def test_transport_embed_shape_is_openai_compatible() -> None:
    transport = FakeEmbeddingTransport(dim=4)
    response = transport.embed(["alpha", "beta", "gamma"], model="fake-model")
    assert response["model"] == "fake-model"
    assert len(response["data"]) == 3
    for index, item in enumerate(response["data"]):
        assert item["index"] == index
        assert len(item["embedding"]) == 4
        assert all(isinstance(component, float) for component in item["embedding"])


def test_transport_call_is_alias_for_embed() -> None:
    transport = FakeEmbeddingTransport(dim=4)
    via_call = transport(["alpha"], model="m")
    via_embed = transport.embed(["alpha"], model="m")
    assert via_call["data"][0]["embedding"] == via_embed["data"][0]["embedding"]


def test_transport_default_model_name_is_supplied() -> None:
    transport = FakeEmbeddingTransport(dim=2)
    response = transport.embed(["only text"])
    assert response["model"] == "fake-embedding-model"


def test_transport_records_calls() -> None:
    transport = FakeEmbeddingTransport(dim=4)
    assert transport.call_count == 0
    assert transport.last_call is None

    transport.embed(["one"], model="m1")
    transport.embed(["two", "three"], model="m2")

    assert transport.call_count == 2
    assert transport.calls[0] == FakeCall(texts=["one"], model="m1")
    assert transport.last_call == FakeCall(texts=["two", "three"], model="m2")


def test_transport_reset_clears_calls_but_stays_deterministic() -> None:
    transport = FakeEmbeddingTransport(dim=4)
    transport.embed(["one"], model="m1")
    transport.reset()

    assert transport.call_count == 0
    assert transport.last_call is None
    # Determinism survives a reset -- overrides/hash-derivation are untouched.
    assert transport.vector_for("one") == deterministic_vector("one", 4)


def test_transport_embed_preserves_text_order() -> None:
    transport = FakeEmbeddingTransport(dim=4)
    texts = ["first", "second", "third"]
    response = transport.embed(texts)
    expected = [transport.vector_for(text) for text in texts]
    actual = [item["embedding"] for item in response["data"]]
    assert actual == expected


# --- overrides ---------------------------------------------------------


def test_override_pins_exact_vector() -> None:
    transport = FakeEmbeddingTransport(dim=3)
    transport.set_override("cat", [1.0, 0.0, 0.0])
    transport.set_override("kitten", [0.9, 0.1, 0.0])

    assert transport.vector_for("cat") == [1.0, 0.0, 0.0]
    assert transport.vector_for("kitten") == [0.9, 0.1, 0.0]
    # Non-overridden text still falls back to the hash-derived default.
    assert transport.vector_for("car") == deterministic_vector("car", 3)


def test_override_via_constructor() -> None:
    transport = FakeEmbeddingTransport(dim=2, overrides={"north": [0.0, 1.0]})
    assert transport.vector_for("north") == [0.0, 1.0]


def test_override_wrong_length_rejected() -> None:
    transport = FakeEmbeddingTransport(dim=3)
    with pytest.raises(ValueError):
        transport.set_override("bad", [1.0, 0.0])


def test_override_used_in_embed_response() -> None:
    transport = FakeEmbeddingTransport(dim=3, overrides={"cat": [1.0, 0.0, 0.0]})
    response = transport.embed(["cat"])
    assert response["data"][0]["embedding"] == [1.0, 0.0, 0.0]


# --- UnreachableTransport ----------------------------------------------


def test_unreachable_transport_raises_by_default() -> None:
    transport = UnreachableTransport()
    with pytest.raises(ConnectionRefusedError):
        transport(["hello"])
    assert transport.calls == [FakeCall(texts=["hello"], model="fake-embedding-model")]


def test_unreachable_transport_raises_custom_error() -> None:
    custom = TimeoutError("simulated timeout")
    transport = UnreachableTransport(error=custom)
    with pytest.raises(TimeoutError):
        transport(["hello"], model="m")


# --- hermeticity: no socket opened by the fakes -------------------------


def test_fake_transport_opens_no_socket() -> None:
    """The core hermetic property: the fake never reaches the network."""
    transport = FakeEmbeddingTransport(dim=4)
    with blocked_sockets():
        response = transport.embed(["hello", "world"])
    assert len(response["data"]) == 2


def test_fake_transport_fixture_opens_no_socket(fake_transport) -> None:  # noqa: F811
    with blocked_sockets():
        response = fake_transport(["fixture text"])
    assert response["data"][0]["embedding"]


def test_no_sockets_fixture_blocks_a_real_connect(no_sockets: None) -> None:  # noqa: F811
    with pytest.raises(NetworkBlockedError):
        socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((DEAD_HOST, DEAD_PORT))


def test_blocked_sockets_guard_catches_a_real_connect_attempt() -> None:
    """Prove the guard itself works: a real connect() inside it must raise."""
    with blocked_sockets():
        with pytest.raises(NetworkBlockedError):
            socket.socket(socket.AF_INET, socket.SOCK_STREAM).connect((DEAD_HOST, DEAD_PORT))


def test_blocked_sockets_restores_real_connect_after_exit() -> None:
    """The guard must not leak into other tests -- connect() is restored on exit."""
    real_connect_before = socket.socket.connect
    with blocked_sockets():
        pass
    assert socket.socket.connect is real_connect_before


# --- DEAD_PORT / dead_endpoint -------------------------------------------


def test_dead_port_is_not_the_real_gateway_port() -> None:
    # 8002 is the real lobes gateway port (model-gear embed gear default).
    # Using it here would defeat the entire point of this constant.
    assert DEAD_PORT != 8002


def test_dead_port_is_in_iana_ephemeral_range() -> None:
    assert 49152 <= DEAD_PORT <= 65535


def test_dead_endpoint_format() -> None:
    assert dead_endpoint() == f"http://{DEAD_HOST}:{DEAD_PORT}/v1"
    assert dead_endpoint("/custom") == f"http://{DEAD_HOST}:{DEAD_PORT}/custom"


def test_dead_port_actually_refuses_a_real_connection() -> None:
    """The one sanctioned real-socket test: prove nothing listens on DEAD_PORT.

    This is what a sibling task's "unreachable endpoint" test will lean on --
    connecting to DEAD_PORT must fail fast (refused or, in a sandboxed
    network namespace, immediately unreachable) rather than hang, on any
    machine including a bare CI runner with nothing listening anywhere.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2.0)
    try:
        with pytest.raises(OSError):
            sock.connect((DEAD_HOST, DEAD_PORT))
    finally:
        sock.close()


# --------------------------------------------------------------------------
# wire_transport -- the fake must actually drive the real client
# --------------------------------------------------------------------------


def test_wire_transport_drives_the_real_embedding_client() -> None:
    """A fake backend must compose with the real client, not just look like one.

    The text-level fakes and the client's wire-level Transport protocol have
    different signatures; `wire_transport` bridges them. This asserts the
    bridge holds against the real EmbeddingClient, so a regression in either
    signature fails here rather than in every verb test downstream.
    """
    from embeddings.backends.client import EmbeddingClient
    from tests._fakes import FakeEmbeddingTransport, blocked_sockets, wire_transport

    fake = FakeEmbeddingTransport(dim=8)
    client = EmbeddingClient(transport=wire_transport(fake))

    with blocked_sockets():
        vectors = client.embed(["alpha", "beta"])

    assert len(vectors) == 2
    assert all(len(v) == 8 for v in vectors)
    # Determinism survives the round trip through the wire format.
    assert vectors[0] == pytest.approx(fake.vector_for("alpha"))
    assert vectors[0] != vectors[1]


def test_wire_transport_propagates_backend_failure() -> None:
    """An unreachable fake backend must surface as the client's exit-2 CliError."""
    from embeddings.backends.client import EmbeddingClient
    from embeddings.cli._errors import CliError
    from tests._fakes import UnreachableTransport, wire_transport

    client = EmbeddingClient(transport=wire_transport(UnreachableTransport()))

    with pytest.raises(CliError) as exc:
        client.embed(["anything"])
    assert exc.value.code == 2
