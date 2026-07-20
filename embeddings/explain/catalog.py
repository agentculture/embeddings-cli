"""Markdown catalog for ``embeddings explain <path>``.

Each entry is verbatim markdown. Keys are command-path tuples. The empty tuple,
``("embeddings",)``, and ``("embeddings-cli",)`` all resolve to the root entry
(the project is named ``embeddings-cli``; the installed command is
``embeddings``).

Keep bodies self-contained: an agent reading one entry should get enough
context without chaining reads.
"""

from __future__ import annotations

_ROOT = """\
# embeddings-cli

A tool for studying embedding spaces as **systems**, not just for querying
individual vectors. An individual embedding is hard to interpret — its
dimensions rarely correspond to clean human concepts. An embedding space,
however, has structure worth measuring: semantic regions and clusters,
neighborhoods and boundaries, recurring directions of meaning, dense and sparse
areas, gaps in what the model represents, and differences between models,
datasets, domains, and versions.

The project treats an embedding model as a **semantic landscape**. Rather than
only asking "which documents are closest to this query?", it asks "what kind of
world does this embedding model create?" — what it places together, what it
keeps apart, which distinctions it preserves, which it collapses, and where the
representation becomes weak or uncertain.

It also serves as the deterministic tool surface for a **Surveyor agent**, which
runs measured experiments against the space (sampling concepts, finding
clusters, proposing and testing region names, discovering and validating
semantic directions) and records an **Atlas** — an evidence-backed
interpretation of the space, not the space itself.

The installed command is `embeddings`.

## Verbs

- `embeddings whoami` — identity probe from `culture.yaml`.
- `embeddings learn` — structured self-teaching prompt.
- `embeddings explain <path>` — markdown docs for any noun/verb.
- `embeddings overview` — descriptive snapshot of the agent.
- `embeddings doctor` — check the agent-identity invariants.
- `embeddings cli overview` — describe the CLI surface.

## Planned verbs

The survey surface is the target design and is **not yet implemented**:
`survey`, `regions`, `neighbors`, `directions`, `gaps`, `compare`, `map`.

## Exit-code policy

- `0` success
- `1` user-input error
- `2` environment / setup error
- `3+` reserved

## See also

- `embeddings explain whoami`
- `embeddings explain doctor`
"""

_WHOAMI = """\
# embeddings whoami

Reports the agent's identity from `culture.yaml`: nick (`suffix`), backend,
served model, and the package version. Read-only.

## Usage

    embeddings whoami
    embeddings whoami --json
"""

_LEARN = """\
# embeddings learn

Prints a structured self-teaching prompt covering purpose, command map,
exit-code policy, `--json` support, and the `explain` pointer. Planned verbs are
listed separately from implemented ones, so a roadmap verb is never mistaken for
a callable one.

## Usage

    embeddings learn
    embeddings learn --json
"""

_EXPLAIN = """\
# embeddings explain <path>

Prints markdown documentation for any noun/verb path. Unlike `--help` (terse,
positional), `explain` is global and addressable by path.

## Usage

    embeddings explain embeddings-cli
    embeddings explain whoami
    embeddings explain --json <path>
"""

_OVERVIEW = """\
# embeddings overview

Read-only descriptive snapshot of the agent: identity (from `culture.yaml`), the
verb surface, and the artifacts this repo carries. Accepts an ignored `target`
so a stray path never hard-fails.

## Usage

    embeddings overview
    embeddings overview --json
"""

_DOCTOR = """\
# embeddings doctor

Checks the agent-identity invariants `steward doctor` verifies:
prompt-file-present and backend-consistency (`colleague` → `AGENTS.colleague.md`), plus a
skills-present check. Exits 1 when unhealthy.

## Usage

    embeddings doctor
    embeddings doctor --json
"""

_CLI = """\
# embeddings cli

Noun group for CLI-surface introspection. `cli overview` describes the CLI
itself (distinct from the global `overview`, which describes the agent).

## Usage

    embeddings cli overview
    embeddings cli overview --json
"""


ENTRIES: dict[tuple[str, ...], str] = {
    (): _ROOT,
    ("embeddings-cli",): _ROOT,
    ("embeddings",): _ROOT,
    ("whoami",): _WHOAMI,
    ("learn",): _LEARN,
    ("explain",): _EXPLAIN,
    ("overview",): _OVERVIEW,
    ("doctor",): _DOCTOR,
    ("cli",): _CLI,
    ("cli", "overview"): _CLI,
}
