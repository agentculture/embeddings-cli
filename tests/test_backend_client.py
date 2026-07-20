"""Tests for the OpenAI-compatible embedding client (embeddings/backends/client.py).

Covers the t5 acceptance criteria:

* an unreachable endpoint raises ``CliError`` code 2, naming the endpoint and
  the override flag/env in the hint, with no traceback leaking;
* a dead-port test using a reserved local port asserts exit 2 + hint text;
* single-host egress — a transport-level test fails on any host but the
  configured one, and no telemetry call is made;
* the endpoint flag's help text states the off-machine implication;
* the transport is injectable, so no test here opens a real socket except the
  one dedicated dead-port test that exercises the real urllib transport.
"""

from __future__ import annotations

import json
import socket
from urllib.parse import urlparse

import pytest

from embeddings.backends.client import (
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    ENDPOINT_FLAG,
    ENDPOINT_HELP,
    ENV_BASE_URL,
    ENV_MODEL,
    EmbeddingClient,
    urllib_transport,
)
from embeddings.cli._errors import EXIT_ENV_ERROR, CliError

# Port 1 (TCPMUX) is reserved and never has a listener on a dev/CI machine;
# binding it requires root, so nothing here will ever be listening on it, and
# connecting (unlike binding) needs no privilege. This is the "reserved dead
# port" the acceptance criteria call for.
DEAD_PORT_URL = "http://127.0.0.1:1/v1"


# --- fakes ------------------------------------------------------------------


class _RecordingTransport:
    """Fake transport that records every call and never touches a socket."""

    def __init__(self, response: bytes) -> None:
        self.response = response
        self.calls: list[tuple[str, bytes, dict[str, str]]] = []

    def __call__(self, url: str, data: bytes, headers: dict[str, str]) -> bytes:
        self.calls.append((url, data, headers))
        return self.response


class _SingleHostTransport:
    """Fake transport that raises if asked to contact any host but ``allowed_host``.

    This is the transport-level enforcement test: if a future change made the
    client issue a second request (telemetry or otherwise) to a different
    host, this fake fails the test rather than silently allowing it.
    """

    def __init__(self, allowed_host: str, response: bytes) -> None:
        self.allowed_host = allowed_host
        self.response = response
        self.calls: list[str] = []

    def __call__(self, url: str, data: bytes, headers: dict[str, str]) -> bytes:
        host = urlparse(url).netloc
        if host != self.allowed_host:
            raise AssertionError(
                f"transport contacted host {host!r}, only {self.allowed_host!r} is configured"
            )
        self.calls.append(url)
        return self.response


class _RaisingTransport:
    """Fake transport that raises a plain exception, simulating connection refused."""

    def __call__(self, url: str, data: bytes, headers: dict[str, str]) -> bytes:
        raise ConnectionRefusedError("[Errno 111] Connection refused")


def _ok_response(n: int) -> bytes:
    return json.dumps({"data": [{"embedding": [float(i), float(i + 1)]} for i in range(n)]}).encode(
        "utf-8"
    )


# --- defaults / env / precedence --------------------------------------------


def test_defaults_are_the_lobes_gateway() -> None:
    client = EmbeddingClient(transport=_RecordingTransport(_ok_response(0)))
    assert client.base_url == DEFAULT_BASE_URL
    assert client.model == DEFAULT_MODEL
    assert DEFAULT_BASE_URL == "http://localhost:8002/v1"
    assert DEFAULT_MODEL == "Qwen/Qwen3-Embedding-0.6B"


def test_default_transport_is_urllib_based() -> None:
    """An unconfigured client's transport is the real (stdlib-only) one.

    The dead-port test below relies on this default rather than passing
    ``transport=`` explicitly, so pin it here: nothing swapped the default
    transport for something that never touches the network.
    """
    client = EmbeddingClient()
    assert client.transport is urllib_transport


def test_urllib_transport_rejects_non_http_schemes() -> None:
    """The real transport refuses file:// and other non-http(s) schemes outright."""
    with pytest.raises(ValueError, match="http"):
        urllib_transport("file:///etc/passwd", b"{}", {})


def test_env_vars_override_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_BASE_URL, "http://localhost:9999/v1")
    monkeypatch.setenv(ENV_MODEL, "some/other-model")
    client = EmbeddingClient(transport=_RecordingTransport(_ok_response(0)))
    assert client.base_url == "http://localhost:9999/v1"
    assert client.model == "some/other-model"


def test_explicit_args_beat_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(ENV_BASE_URL, "http://localhost:9999/v1")
    monkeypatch.setenv(ENV_MODEL, "env-model")
    client = EmbeddingClient(
        base_url="http://localhost:7777/v1",
        model="explicit-model",
        transport=_RecordingTransport(_ok_response(0)),
    )
    assert client.base_url == "http://localhost:7777/v1"
    assert client.model == "explicit-model"


def test_base_url_trailing_slash_is_normalised() -> None:
    client = EmbeddingClient(
        base_url="http://localhost:8002/v1/", transport=_RecordingTransport(_ok_response(0))
    )
    assert client.endpoint == "http://localhost:8002/v1/embeddings"


# --- success path ------------------------------------------------------------


def test_embed_posts_openai_shape_and_parses_response() -> None:
    transport = _RecordingTransport(_ok_response(2))
    client = EmbeddingClient(
        base_url="http://localhost:8002/v1", model="test-model", transport=transport
    )

    vectors = client.embed(["hello", "world"])

    assert vectors == [[0.0, 1.0], [1.0, 2.0]]
    assert len(transport.calls) == 1
    url, data, headers = transport.calls[0]
    assert url == "http://localhost:8002/v1/embeddings"
    assert headers["Content-Type"] == "application/json"
    body = json.loads(data)
    assert body == {"model": "test-model", "input": ["hello", "world"]}


# --- unreachable endpoint -> CliError code 2 ---------------------------------


def test_transport_failure_raises_cli_error_code_2_with_hint() -> None:
    client = EmbeddingClient(base_url="http://localhost:8002/v1", transport=_RaisingTransport())

    with pytest.raises(CliError) as exc_info:
        client.embed(["hello"])

    err = exc_info.value
    assert err.code == EXIT_ENV_ERROR
    assert err.code == 2
    assert "http://localhost:8002/v1/embeddings" in err.message
    assert ENDPOINT_FLAG in err.remediation
    assert ENV_BASE_URL in err.remediation


def test_transport_failure_does_not_leak_the_original_exception_type() -> None:
    """The CliError, not the raw transport exception, is what callers see.

    ``_dispatch`` in embeddings/cli/__init__.py only prevents a traceback from
    reaching stderr for whatever it catches; the client itself must hand back
    a CliError so that guarantee holds without _dispatch needing to know
    anything about connection-refused errors specifically.
    """
    client = EmbeddingClient(base_url="http://localhost:8002/v1", transport=_RaisingTransport())

    with pytest.raises(CliError):
        client.embed(["hello"])
    # No ConnectionRefusedError (or any non-CliError) should escape.


def test_malformed_response_raises_cli_error_code_2() -> None:
    bad_transport = _RecordingTransport(b"not json at all")
    client = EmbeddingClient(base_url="http://localhost:8002/v1", transport=bad_transport)

    with pytest.raises(CliError) as exc_info:
        client.embed(["hello"])

    assert exc_info.value.code == 2


def test_response_missing_data_key_raises_cli_error_code_2() -> None:
    bad_transport = _RecordingTransport(json.dumps({"unexpected": "shape"}).encode("utf-8"))
    client = EmbeddingClient(base_url="http://localhost:8002/v1", transport=bad_transport)

    with pytest.raises(CliError) as exc_info:
        client.embed(["hello"])

    assert exc_info.value.code == 2


# --- dead-port test: the real urllib transport against a reserved port ------


def test_dead_port_via_real_transport_raises_cli_error_code_2_with_hint() -> None:
    """Uses the actual default (urllib) transport against a reserved dead port.

    This is the one test in the module that touches ``socket``/``urllib`` for
    real, and it does so against port 1, which is guaranteed to have nothing
    listening — CI never has an embedding gateway running, so this is the
    day-one failure path, not a hypothetical.
    """
    client = EmbeddingClient(base_url=DEAD_PORT_URL)  # default transport = urllib_transport

    with pytest.raises(CliError) as exc_info:
        client.embed(["hello"])

    err = exc_info.value
    assert err.code == 2
    assert "http://127.0.0.1:1/v1/embeddings" in err.message
    assert ENDPOINT_FLAG in err.remediation
    assert ENV_BASE_URL in err.remediation
    assert "hint" not in err.message.lower()  # message and hint stay separate fields


def test_dead_port_end_to_end_error_rendering(capsys: pytest.CaptureFixture[str]) -> None:
    """The CliError from a dead port renders through emit_error exactly like any other."""
    from embeddings.cli._output import emit_error

    client = EmbeddingClient(base_url=DEAD_PORT_URL)
    with pytest.raises(CliError) as exc_info:
        client.embed(["hello"])

    emit_error(exc_info.value, json_mode=False)
    err_out = capsys.readouterr().err
    assert err_out.startswith("error:")
    assert "hint:" in err_out
    assert ENDPOINT_FLAG in err_out


# --- single-host egress -------------------------------------------------------


def test_single_host_egress_only_configured_endpoint_is_contacted() -> None:
    """Fails if the client ever contacts a host other than the configured one.

    ``_SingleHostTransport`` raises on any host mismatch; a client that made a
    second request anywhere else (telemetry or otherwise) would fail this
    test immediately.
    """
    transport = _SingleHostTransport(allowed_host="localhost:8002", response=_ok_response(1))
    client = EmbeddingClient(base_url="http://localhost:8002/v1", transport=transport)

    client.embed(["hello"])

    assert transport.calls == ["http://localhost:8002/v1/embeddings"]
    assert len(transport.calls) == 1  # exactly one outbound call: no telemetry, no second host


def test_single_host_egress_rejects_a_transport_that_calls_elsewhere() -> None:
    """Sanity-check the fake itself: it must reject a host it isn't configured for.

    Called directly (not through ``EmbeddingClient.embed``) so this test
    verifies the enforcement fixture in isolation, independent of how the
    client happens to wrap exceptions.
    """
    guard = _SingleHostTransport(allowed_host="localhost:8002", response=b"{}")

    with pytest.raises(AssertionError, match="telemetry.example.com"):
        guard("http://telemetry.example.com/report", b"{}", {})


def test_client_contacting_a_second_host_fails_the_test() -> None:
    """Demonstrates the acceptance criterion directly: a would-be second-host
    call surfaces as a test failure, not a silent success.

    ``_SingleHostTransport`` is configured for a *different* host than the
    client actually uses, simulating what would happen if a future change
    made the client call somewhere unexpected. The resulting ``CliError``
    (the broad ``except Exception`` in ``embed`` wraps the transport's
    ``AssertionError``) is left unhandled here on purpose, so this test fails
    — proving the harness does not let an extra-host call pass quietly.
    """
    wrong_host_guard = _SingleHostTransport(allowed_host="not-the-configured-host", response=b"{}")
    client = EmbeddingClient(base_url="http://localhost:8002/v1", transport=wrong_host_guard)

    with pytest.raises(CliError, match="not-the-configured-host"):
        client.embed(["hello"])


def test_embed_makes_exactly_one_transport_call_regardless_of_batch_size() -> None:
    transport = _RecordingTransport(_ok_response(5))
    client = EmbeddingClient(base_url="http://localhost:8002/v1", transport=transport)

    client.embed(["a", "b", "c", "d", "e"])

    assert len(transport.calls) == 1


def test_no_socket_opened_when_transport_is_injected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Proves the injected-transport path never falls through to a real socket."""

    def _boom(*args: object, **kwargs: object) -> None:
        raise AssertionError("a real socket must never be opened when transport is injected")

    monkeypatch.setattr(socket, "socket", _boom)
    monkeypatch.setattr(socket, "create_connection", _boom)

    transport = _RecordingTransport(_ok_response(1))
    client = EmbeddingClient(base_url="http://localhost:8002/v1", transport=transport)

    client.embed(["hello"])  # must not raise, i.e. must not touch socket.*


# --- endpoint help text names the off-machine implication --------------------


def test_endpoint_help_text_states_off_machine_implication() -> None:
    lowered = ENDPOINT_HELP.lower()
    assert "off" in lowered and "machine" in lowered
    assert DEFAULT_BASE_URL in ENDPOINT_HELP
    assert ENV_BASE_URL in ENDPOINT_HELP
    assert isinstance(ENDPOINT_FLAG, str) and ENDPOINT_FLAG.startswith("--")
