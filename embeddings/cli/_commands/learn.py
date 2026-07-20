"""``embeddings learn`` — the learnability affordance.

Prints a structured self-teaching prompt. Must satisfy the agent-first rubric:
>=200 chars and mention purpose, command map, exit codes, --json, and explain.

The command map lists only what is implemented. The survey surface this project
is being built toward (``survey``, ``regions``, ``neighbors``, ``directions``,
``gaps``, ``compare``, ``map``) is named under "Planned" so an agent reading
this never mistakes a roadmap verb for a callable one.
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

Planned (not yet implemented — calling these will fail)
-------------------------------------------------------
  embeddings survey <corpus>    Embed a corpus and survey the resulting space.
  embeddings regions            Identify semantic regions and clusters.
  embeddings neighbors <text>   Inspect a neighborhood and its boundaries.
  embeddings directions         Discover and test semantic directions.
  embeddings gaps               Find weak, sparse, or uncertain coverage.
  embeddings compare <a> <b>    Compare two models' geometry.
  embeddings map                Render an interpretable map of the space.

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
        ],
        "planned_commands": [
            {"path": ["survey"], "summary": "Embed a corpus and survey the resulting space."},
            {"path": ["regions"], "summary": "Identify semantic regions and clusters."},
            {"path": ["neighbors"], "summary": "Inspect a neighborhood and its boundaries."},
            {"path": ["directions"], "summary": "Discover and test semantic directions."},
            {"path": ["gaps"], "summary": "Find weak, sparse, or uncertain coverage."},
            {"path": ["compare"], "summary": "Compare two models' geometry."},
            {"path": ["map"], "summary": "Render an interpretable map of the space."},
        ],
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
