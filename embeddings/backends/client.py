"""OpenAI-compatible embedding client (single-host egress boundary).

Talks to exactly one configured OpenAI-compatible ``/v1/embeddings`` endpoint
— the local model-gear ``lobes`` gateway by default — and nowhere else.
Corpus text is the payload of every request this client makes; there is no
telemetry call and no second host anywhere in this module. Pointing the
client at a non-local endpoint is an explicit operator choice that ships
corpus content off that machine (see :data:`ENDPOINT_HELP`, which the CLI's
``--embed-url`` flag surfaces verbatim).

Environment variables
----------------------
``EMBEDDINGS_EMBED_URL``
    Overrides the default base URL (``http://localhost:8002/v1``). Naming
    follows the ``EIDETIC_EMBED_URL`` convention eidetic-cli already uses for
    the same gateway.
``EMBEDDINGS_EMBED_MODEL``
    Overrides the default model (``Qwen/Qwen3-Embedding-0.6B``).

Precedence for both: an explicit constructor argument beats the environment
variable, which beats the built-in default.

Transport injection
--------------------
:class:`EmbeddingClient` takes a ``transport`` callable — ``(url, data,
headers) -> bytes`` — defaulting to :func:`urllib_transport`, a thin
stdlib-only POST (no ``requests``/``httpx``; the runtime package has no
third-party dependencies). Tests inject a fake transport so the suite never
opens a real socket: CI runs with no embedding gateway listening anywhere, so
this is load-bearing, not incidental.

Failure contract
-----------------
Every transport failure (connection refused, DNS failure, timeout, non-2xx
response) and every malformed-response case (bad JSON, missing ``data``/
``embedding`` keys) is caught here and re-raised as
:class:`embeddings.cli._errors.CliError` with code
:data:`embeddings.cli._errors.EXIT_ENV_ERROR` (2). The message names the
endpoint that was unreachable; the remediation hint names the override
flag/env var. No exception from the underlying transport is ever left to
propagate as a bare traceback — the backend being down is the day-one
failure path here, not an edge case.
"""

from __future__ import annotations

import json
import os
import urllib.request
from dataclasses import dataclass, field
from typing import Protocol
from urllib.parse import urlparse

from embeddings.cli._errors import EXIT_ENV_ERROR, CliError

#: Built-in default endpoint: the local model-gear ``lobes`` gateway.
#:
#: The authoritative resolver for this value is ``lobes endpoint embedder``,
#: NOT documentation. An earlier default of ``:8002`` came from the build brief
#: and from eidetic-cli's README/skills, which had drifted from reality (and
#: from eidetic's own code default of ``:8101``); the gateway container
#: publishes host port 8001. When this looks wrong, ask the tool:
#:
#:     lobes endpoint embedder --json
DEFAULT_BASE_URL = "http://localhost:8001/v1"
#: Built-in default model served by the gateway above.
DEFAULT_MODEL = "Qwen/Qwen3-Embedding-0.6B"

#: Environment variable that overrides the base URL.
ENV_BASE_URL = "EMBEDDINGS_EMBED_URL"
#: Environment variable that overrides the model name.
ENV_MODEL = "EMBEDDINGS_EMBED_MODEL"
#: Environment variables consulted, in order, for the gateway bearer token.
#: The gateway rejects unauthenticated requests, so this is required in
#: practice; it is read from the environment only — never a flag, so the
#: secret cannot land in shell history, a process list, or a help string.
ENV_API_KEY = "EMBEDDINGS_API_KEY"
ENV_API_KEY_FALLBACK = "COLLEAGUE_API_KEY"

#: Flag name a CLI verb should register for the base-URL override; referenced
#: in error hints and in :data:`ENDPOINT_HELP` so the flag, the hint, and the
#: help text an operator actually sees all agree on one spelling.
ENDPOINT_FLAG = "--embed-url"
#: Flag name a CLI verb should register for the model override.
MODEL_FLAG = "--embed-model"

#: Help text for the endpoint-override flag. States the off-machine
#: implication explicitly, per the single-host-egress requirement — a
#: non-default endpoint is an explicit operator choice, not a silent upgrade.
ENDPOINT_HELP = (
    f"Override the embedding backend's base URL (default: {DEFAULT_BASE_URL}; "
    f"env: {ENV_BASE_URL}). Pointing this at a non-local endpoint ships corpus "
    "content off this machine — only do so deliberately."
)
#: Help text for the model-override flag.
MODEL_HELP = f"Override the embedding model name (default: {DEFAULT_MODEL}; env: {ENV_MODEL})."

# Socket-level timeout for the default transport. Connection-refused (the
# day-one failure here) returns immediately; this bounds the slow failure
# modes (a host that accepts the connection but never answers).
_TIMEOUT_SECONDS = 10.0


class Transport(Protocol):
    """Callable that performs one HTTP POST and returns the raw response body.

    Implementations must raise on any failure (connection, HTTP status, or
    otherwise) rather than returning a sentinel — :meth:`EmbeddingClient.embed`
    catches broadly and translates any raised exception into a
    :class:`CliError`.
    """

    def __call__(self, url: str, data: bytes, headers: dict[str, str]) -> bytes: ...  # noqa: E704


def urllib_transport(url: str, data: bytes, headers: dict[str, str]) -> bytes:
    """Default transport: one stdlib ``urllib`` POST. No third-party HTTP deps.

    This is the only place in the module that touches the network, and it is
    called from exactly one call site (:meth:`EmbeddingClient.embed`) against
    exactly one URL — the configured endpoint. There is no second request,
    telemetry or otherwise.
    """
    scheme = urlparse(url).scheme
    if scheme not in ("http", "https"):
        # Reject file:// and other custom schemes outright — the endpoint is
        # operator config (flag/env/default), never attacker-controlled, but
        # there is no reason to accept anything but plain HTTP(S) here.
        raise ValueError(f"refusing non-http(s) URL scheme {scheme!r} for {url!r}")
    request = urllib.request.Request(url, data=data, headers=headers, method="POST")
    # Scheme was restricted to http/https just above; url is the
    # operator-configured embedding endpoint, never remote/attacker input.
    with urllib.request.urlopen(request, timeout=_TIMEOUT_SECONDS) as response:  # nosec B310
        return response.read()


def _default_base_url() -> str:
    return os.environ.get(ENV_BASE_URL, DEFAULT_BASE_URL)


def _default_model() -> str:
    return os.environ.get(ENV_MODEL, DEFAULT_MODEL)


def _default_api_key() -> str | None:
    """Resolve the gateway bearer token from the environment, or ``None``.

    Environment only, by design: a token passed as a CLI flag would be visible
    in shell history and in every process listing on the box.
    """
    for name in (ENV_API_KEY, ENV_API_KEY_FALLBACK):
        value = os.environ.get(name)
        if value:
            return value
    return None


def _override_hint(base_url: str) -> str:
    return (
        f"start the embedding gateway at {base_url} (resolve the real one with "
        f"'lobes endpoint embedder'), or point elsewhere with "
        f"{ENDPOINT_FLAG} / the {ENV_BASE_URL} environment variable"
    )


def _looks_like_auth_failure(err: Exception) -> bool:
    """True when a transport failure was the gateway refusing credentials.

    Checked structurally via ``HTTPError.code`` where available, so the
    classification does not depend on the wording of an error message.
    """
    code = getattr(err, "code", None)
    return code in (401, 403)


def _auth_hint() -> str:
    return (
        f"the gateway rejected the credentials; set {ENV_API_KEY} (or "
        f"{ENV_API_KEY_FALLBACK}) in the environment to a valid gateway key"
    )


@dataclass
class EmbeddingClient:
    """Client for a single OpenAI-compatible ``/v1/embeddings`` endpoint.

    ``base_url`` and ``model`` default to the environment-variable-aware
    factories above, so an unconfigured client already honours
    ``EMBEDDINGS_EMBED_URL`` / ``EMBEDDINGS_EMBED_MODEL`` without callers
    having to read the environment themselves. Pass either explicitly (e.g.
    from a parsed CLI flag) to override both the default and the environment.
    """

    base_url: str = field(default_factory=_default_base_url)
    model: str = field(default_factory=_default_model)
    transport: Transport = urllib_transport
    #: Bearer token for the gateway, resolved from the environment. Never
    #: rendered: :meth:`__repr__` is suppressed for this field so the secret
    #: cannot leak through a traceback, a log line, or a debugger dump.
    api_key: str | None = field(default_factory=_default_api_key, repr=False)

    def __post_init__(self) -> None:
        self.base_url = self.base_url.rstrip("/")

    @property
    def endpoint(self) -> str:
        """The full ``/embeddings`` URL this client posts to — its only host."""
        return f"{self.base_url}/embeddings"

    def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of texts; returns one vector per input, in the same order.

        Raises :class:`CliError` (code 2) if the endpoint is unreachable or
        returns a response that doesn't match the OpenAI ``/v1/embeddings``
        shape. Never raises anything else.
        """
        endpoint = self.endpoint
        payload = json.dumps({"model": self.model, "input": list(texts)}).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        try:
            raw = self.transport(endpoint, payload, headers)
        except Exception as err:  # noqa: BLE001 - every transport failure becomes a CliError
            # An auth rejection is a *different* operator fix than an
            # unreachable host, so it gets its own hint rather than being
            # reported as "the gateway is down".
            if _looks_like_auth_failure(err):
                raise CliError(
                    code=EXIT_ENV_ERROR,
                    message=f"embedding backend at {endpoint} rejected the request: {err}",
                    remediation=_auth_hint(),
                ) from err
            raise CliError(
                code=EXIT_ENV_ERROR,
                message=f"embedding backend unreachable at {endpoint}: {err}",
                remediation=_override_hint(self.base_url),
            ) from err

        try:
            body = json.loads(raw)
            return [item["embedding"] for item in body["data"]]
        except (json.JSONDecodeError, KeyError, TypeError, UnicodeDecodeError) as err:
            raise CliError(
                code=EXIT_ENV_ERROR,
                message=(f"embedding backend at {endpoint} returned an unexpected response: {err}"),
                remediation=(
                    "verify the endpoint is OpenAI-compatible (POST /embeddings -> "
                    f"{{data: [{{embedding: [...]}}]}}), or override it with "
                    f"{ENDPOINT_FLAG} / {ENV_BASE_URL}"
                ),
            ) from err
