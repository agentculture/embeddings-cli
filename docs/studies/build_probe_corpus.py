#!/usr/bin/env python3
"""Build the probe corpora for the eidetic-embedder study.

Writes two JSONL corpora holding the *same* 120 probe texts:

* ``probe-corpus.jsonl`` -- the texts as written.
* ``probe-corpus-instruct.jsonl`` -- every text wrapped in Qwen3-Embedding's
  instruction-prefixed query format (see ``probes.INSTRUCTION``).

Record ids are identical between the two files, which is what lets the study
measure a distance between an unprefixed stored record in one atlas and an
instruction-prefixed query in the other. That cross-atlas measurement is
legitimate only because both atlases come from the same model and therefore
share a coordinate frame; ``docs/design-notes/semantic-measurement.md`` section
2 forbids it across models, and the study says so where it uses it.

Usage, from the repository root:

    uv run python docs/studies/build_probe_corpus.py

Deterministic: no network, no randomness, byte-identical output every run.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from probes import instruct, padded, texts  # noqa: E402  (path set up immediately above)


def build(out_dir: Path) -> int:
    """Write both corpora into OUT_DIR and report what was written."""
    records = []
    for record_id, text, fact_id, kind, role in texts():
        records.append(
            {
                "id": record_id,
                "text": text,
                "fact": fact_id,
                "kind": kind,
                "role": role,
            }
        )

    ids = [record["id"] for record in records]
    if len(set(ids)) != len(ids):
        print("duplicate record ids generated -- fix probes.py", file=sys.stderr)
        return 2

    plain_path = out_dir / "probe-corpus.jsonl"
    instruct_path = out_dir / "probe-corpus-instruct.jsonl"

    with plain_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    with instruct_path.open("w", encoding="utf-8") as handle:
        for record in records:
            prefixed = dict(record, text=instruct(record["text"]), prefixed=True)
            handle.write(json.dumps(prefixed, ensure_ascii=False, sort_keys=True) + "\n")

    long_path = out_dir / "probe-corpus-long.jsonl"
    with long_path.open("w", encoding="utf-8") as handle:
        for record in records:
            if record["kind"] == "control":
                continue
            longer = dict(record, text=padded(record["text"]), padded=True)
            handle.write(json.dumps(longer, ensure_ascii=False, sort_keys=True) + "\n")

    print(f"wrote {len(records)} records to {plain_path}", file=sys.stderr)
    print(f"wrote {len(records)} records to {instruct_path}", file=sys.stderr)
    print(f"wrote {len(records) - 20} records to {long_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(build(Path(__file__).resolve().parent))
