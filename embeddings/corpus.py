"""Corpus loader for JSONL files (each line is a JSON object).

Each line must contain a JSON object with:
- 'text' field (required): the text content
- 'id' field (optional): document identifier, defaults to line index
- All other keys are preserved as item metadata
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from embeddings.cli._errors import EXIT_USER_ERROR, CliError


@dataclass
class CorpusItem:
    """A single item from a corpus file.

    Attributes:
        text: The text content (required field from JSONL).
        id: Document identifier (defaults to line index as string).
        metadata: All extra keys from the JSON object beyond 'text' and 'id'.
    """

    text: str
    id: str
    metadata: dict[str, Any] = field(default_factory=dict)


def load_corpus(path: str) -> list[CorpusItem]:
    """Load a JSONL corpus file.

    Each line is parsed as JSON. The object must contain a 'text' field.
    If 'id' is missing, it defaults to the 0-based line index as a string.
    All other keys are preserved in the item's metadata dict.

    Args:
        path: Path to the JSONL file.

    Returns:
        A list of CorpusItem objects.

    Raises:
        CliError: If a line has malformed JSON or is missing the 'text' field.
                  Error code is EXIT_USER_ERROR (1) and message includes line number.
    """
    items: list[CorpusItem] = []
    seen_ids: dict[str, int] = {}  # id -> line it first appeared on
    corpus_path = Path(path)

    if not corpus_path.exists():
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"corpus file not found: {path}",
            remediation=f"check that {path} exists and is readable",
        )

    with open(corpus_path, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.rstrip("\n\r")
            if not line:  # skip empty lines
                continue

            # Parse JSON
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as e:
                raise CliError(
                    code=EXIT_USER_ERROR,
                    message=f"malformed JSON on line {line_num}: {e}",
                    remediation=f"fix the JSON syntax on line {line_num}",
                ) from e

            # Check for required 'text' field
            if "text" not in obj:
                raise CliError(
                    code=EXIT_USER_ERROR,
                    message=f"line {line_num} missing required 'text' field",
                    remediation=f"add a 'text' field to line {line_num}",
                )

            text = obj["text"]
            item_id = obj.get("id")
            if item_id is None:
                item_id = str(line_num - 1)  # 0-based index
            else:
                item_id = str(item_id)

            # Collect metadata (all keys except 'text' and 'id')
            metadata = {k: v for k, v in obj.items() if k not in ("text", "id")}

            if item_id in seen_ids:
                # Ids are the join key for `neighbors --id`, `directions
                # --pairs`, and compare's probe matching. A duplicate does not
                # fail loudly on its own -- it silently resolves to whichever
                # record was read first, so every downstream lookup is wrong in
                # a way nothing reports. Refuse at load time instead.
                raise CliError(
                    code=EXIT_USER_ERROR,
                    message=(
                        f"duplicate record id {item_id!r} on line {line_num} "
                        f"(first seen on line {seen_ids[item_id]})"
                    ),
                    remediation=(
                        "give every record a unique 'id', or omit 'id' entirely to "
                        "have line indices assigned automatically"
                    ),
                )
            seen_ids[item_id] = line_num

            items.append(CorpusItem(text=text, id=item_id, metadata=metadata))

    return items
