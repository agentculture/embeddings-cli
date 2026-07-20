"""``embeddings learn`` — the learnability affordance.

Prints a structured self-teaching prompt. Must satisfy the agent-first rubric:
>=200 chars and mention purpose, command map, exit codes, --json, and explain.

The command map lists exactly what is implemented and callable today,
including the full survey surface (``survey``, ``regions``, ``neighbors``,
``directions``, ``gaps``, ``compare``, ``map``) alongside the identity and
introspection verbs. Nothing on this CLI's roadmap remains unimplemented, so
there is no separate "planned" list to keep apart from it -- but the
``planned_commands`` key stays in the ``--json`` payload (empty) for schema
stability, and ``tests/test_roadmap_lockstep.py`` asserts it against the real
parser so a future addition can never drift back into naming an
already-callable verb as merely planned.
"""

from __future__ import annotations

import argparse

from embeddings import __version__
from embeddings.cli._output import emit_result

_PURPOSE = (
    "Study embedding spaces as systems — the geometry of meaning a model "
    "produces — rather than querying individual vectors."
)

_TEXT = """\
embeddings-cli — study embedding spaces as systems, not individual vectors.

Purpose
-------
An individual embedding is hard to interpret; its dimensions rarely map to clean
human concepts. An embedding space, though, has structure worth measuring:
semantic regions, neighborhoods and boundaries, recurring directions of meaning,
dense and sparse areas, gaps in coverage, and differences between models. This
tool treats an embedding model as a semantic landscape and asks what kind of
world the model creates — not just which documents sit closest to a query.

It also serves as the deterministic tool surface for a Surveyor agent, which
runs measured experiments against the space and records an evidence-backed
Atlas: the interpretation, plus the measurements that make it inspectable.

Commands
--------
  embeddings whoami             Identity from culture.yaml.
  embeddings learn              This self-teaching prompt.
  embeddings explain <path>...  Markdown docs for any noun/verb path.
  embeddings overview           Descriptive snapshot of the agent.
  embeddings doctor             Check the agent-identity invariants.
  embeddings cli overview       Describe the CLI surface itself.
  embeddings survey <corpus>    Embed a corpus and write the atlas other verbs read.
  embeddings regions            Cluster a surveyed atlas into semantic regions.
  embeddings neighbors <text>   k nearest atlas items by cosine distance.
  embeddings directions discover  Discover semantic directions and test them.
  embeddings compare             Compare two atlases' k-NN and rank agreement.
  embeddings gaps                Find corpus-coverage and model-representation gaps.
  embeddings map                 Render a 2-D (PCA) map of a surveyed atlas.

All commands above are implemented and callable today; nothing on this CLI's
survey-surface roadmap remains planned.

Machine-readable output
-----------------------
Every command supports --json. Errors in JSON mode emit
{"code", "message", "remediation"} to stderr. Stdout and stderr never mix.

Exit-code policy
----------------
  0 success
  1 user-input error (bad flag, bad path, missing arg)
  2 environment / setup error
  3+ reserved

More detail
-----------
  embeddings explain embeddings-cli
"""


def _as_json_payload() -> dict[str, object]:
    return {
        "tool": "embeddings-cli",
        "command": "embeddings",
        "version": __version__,
        "purpose": _PURPOSE,
        "commands": [
            {"path": ["whoami"], "summary": "Identity probe from culture.yaml."},
            {"path": ["learn"], "summary": "Self-teaching prompt."},
            {"path": ["explain"], "summary": "Markdown docs by path."},
            {"path": ["overview"], "summary": "Descriptive snapshot of the agent."},
            {"path": ["doctor"], "summary": "Check the agent-identity invariants."},
            {"path": ["cli", "overview"], "summary": "Describe the CLI surface."},
            {
                "path": ["survey"],
                "summary": "Embed a corpus and write the atlas other verbs read.",
            },
            {
                "path": ["regions"],
                "summary": "Cluster a surveyed atlas into semantic regions, with stability.",
            },
            {
                "path": ["neighbors"],
                "summary": "k nearest atlas items to a text query or item id, by cosine distance.",
            },
            {
                "path": ["directions", "discover"],
                "summary": "Discover candidate semantic directions and test them held-out.",
            },
            {
                "path": ["directions", "overview"],
                "summary": "Describe the directions noun and its sub-verbs.",
            },
            {
                "path": ["compare"],
                "summary": "Compare two atlases' k-NN and rank agreement on a shared probe set.",
            },
            {
                "path": ["gaps"],
                "summary": "Find corpus-coverage gaps, plus model-representation gaps via --probe.",
            },
            {
                "path": ["map"],
                "summary": "PCA-project an atlas to 2-D, optionally writing an HTML artifact.",
            },
        ],
        # Nothing on this CLI's survey-surface roadmap remains planned -- every
        # verb once listed here (survey, regions, neighbors, directions, gaps,
        # compare, map) is now implemented above. The key stays for schema
        # stability; tests/test_roadmap_lockstep.py asserts against the real
        # parser that nothing here can ever name an already-callable verb.
        "planned_commands": [],
        "exit_codes": {
            "0": "success",
            "1": "user-input error",
            "2": "environment/setup error",
        },
        "json_support": True,
        "explain_pointer": "embeddings explain <path>",
    }


def cmd_learn(args: argparse.Namespace) -> int:
    if getattr(args, "json", False):
        emit_result(_as_json_payload(), json_mode=True)
    else:
        emit_result(_TEXT, json_mode=False)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "learn",
        help="Print a structured self-teaching prompt for agent consumers.",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_learn)
