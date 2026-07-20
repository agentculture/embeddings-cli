#!/usr/bin/env python3
"""Build the evidence corpus for the live end-to-end run.

Extracts prose paragraphs from this repository's own committed documents and
writes them as JSONL, one record per paragraph, with the source file and the
nearest preceding heading preserved as metadata.

The sources are chosen for topical variety rather than volume: project framing
(README), agent guidance (CLAUDE.md, AGENTS.colleague.md), a measurement
constraints note, a release changelog, a skill-provenance ledger, a spec, a
plan, and the Apache-2.0 licence text. The licence is included deliberately —
it is legal boilerplate in a corpus that is otherwise engineering prose, so a
clustering that cannot separate it is not finding much.

Usage (from the repository root):

    uv run python docs/evidence/build_corpus.py docs/evidence/corpus.jsonl

Deterministic: same repository state in, byte-identical JSONL out. No network.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

# (path, id-prefix) pairs, ordered so the output JSONL is stable across runs.
# The prefix is explicit rather than derived from the filename because the spec
# and the plan share a stem -- deriving ids from stems silently collides them,
# which is how the duplicate-id finding in the accompanying evidence document
# was discovered.
SOURCES = [
    ("README.md", "readme"),
    ("CLAUDE.md", "claude-md"),
    ("AGENTS.colleague.md", "agents-colleague"),
    ("CHANGELOG.md", "changelog"),
    ("docs/design-notes/semantic-measurement.md", "design-note"),
    ("docs/skill-sources.md", "skill-sources"),
    ("docs/specs/2026-07-20-embeddings-survey-surface.md", "spec"),
    ("docs/plans/2026-07-20-embeddings-survey-surface.md", "plan"),
    ("LICENSE", "license"),
]

MIN_CHARS = 180
MAX_CHARS = 1600


def paragraphs(text: str):
    """Yield (heading, paragraph) pairs, skipping fenced code and tables."""
    heading = ""
    buf: list[str] = []
    in_fence = False
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            in_fence = not in_fence
            if buf:
                yield heading, " ".join(buf)
                buf = []
            continue
        if in_fence:
            continue
        if stripped.startswith("#"):
            if buf:
                yield heading, " ".join(buf)
                buf = []
            heading = stripped.lstrip("#").strip()
            continue
        if not stripped:
            if buf:
                yield heading, " ".join(buf)
                buf = []
            continue
        # Skip markdown tables and horizontal rules; they embed as noise.
        if stripped.startswith("|") or set(stripped) <= set("-=*_ "):
            continue
        buf.append(stripped)
    if buf:
        yield heading, " ".join(buf)


def clean(para: str) -> str:
    """Strip list bullets and inline markdown emphasis/link syntax."""
    para = re.sub(r"^[-*+]\s+", "", para)
    para = re.sub(r"^\d+\.\s+", "", para)
    para = re.sub(r"\[([^\]]+)\]\([^)]*\)", r"\1", para)
    para = para.replace("**", "").replace("`", "")
    return re.sub(r"\s+", " ", para).strip()


def main(argv: list[str]) -> int:
    out_path = Path(argv[1] if len(argv) > 1 else "docs/evidence/corpus.jsonl")
    root = Path(__file__).resolve().parents[2]
    records = []
    for source, prefix in SOURCES:
        path = root / source
        if not path.exists():
            print(f"missing source: {source}", file=sys.stderr)
            return 2
        n = 0
        for heading, para in paragraphs(path.read_text(encoding="utf-8")):
            text = clean(para)
            if not (MIN_CHARS <= len(text) <= MAX_CHARS):
                continue
            n += 1
            records.append(
                {
                    "id": f"{prefix}-{n:03d}",
                    "text": text,
                    "source": source,
                    "heading": heading,
                }
            )
    ids = [record["id"] for record in records]
    if len(set(ids)) != len(ids):
        print("duplicate ids generated -- fix the prefix table", file=sys.stderr)
        return 2
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    print(f"wrote {len(records)} records to {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
