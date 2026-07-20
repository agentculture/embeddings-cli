"""Tests for the corpus loader (embeddings.corpus)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from embeddings.cli._errors import EXIT_USER_ERROR, CliError
from embeddings.corpus import load_corpus


def test_load_corpus_valid_minimal() -> None:
    """Load a valid corpus with only required 'text' field."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"text": "hello world"}) + "\n")
        f.write(json.dumps({"text": "goodbye world"}) + "\n")
        path = f.name

    try:
        items = load_corpus(path)
        assert len(items) == 2
        assert items[0].text == "hello world"
        assert items[0].id == "0"
        assert items[1].text == "goodbye world"
        assert items[1].id == "1"
    finally:
        Path(path).unlink()


def test_load_corpus_with_explicit_id() -> None:
    """Load corpus where id is explicitly provided."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"text": "hello", "id": "custom-1"}) + "\n")
        f.write(json.dumps({"text": "world", "id": "custom-2"}) + "\n")
        path = f.name

    try:
        items = load_corpus(path)
        assert len(items) == 2
        assert items[0].id == "custom-1"
        assert items[1].id == "custom-2"
    finally:
        Path(path).unlink()


def test_load_corpus_extra_keys_preserved() -> None:
    """Extra keys beyond 'text' and 'id' are preserved as metadata."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(
            json.dumps(
                {
                    "text": "hello",
                    "source": "test",
                    "year": 2025,
                    "tags": ["a", "b"],
                }
            )
            + "\n"
        )
        path = f.name

    try:
        items = load_corpus(path)
        assert len(items) == 1
        assert items[0].text == "hello"
        assert items[0].metadata["source"] == "test"
        assert items[0].metadata["year"] == 2025
        assert items[0].metadata["tags"] == ["a", "b"]
    finally:
        Path(path).unlink()


def test_load_corpus_mixed_extra_keys() -> None:
    """Extra keys are preserved individually per item."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"text": "hello", "source": "a"}) + "\n")
        f.write(json.dumps({"text": "world", "source": "b", "extra": 123}) + "\n")
        path = f.name

    try:
        items = load_corpus(path)
        assert items[0].metadata == {"source": "a"}
        assert items[1].metadata == {"source": "b", "extra": 123}
    finally:
        Path(path).unlink()


def test_load_corpus_missing_text_field() -> None:
    """Missing 'text' field raises CliError with code 1 naming the line number."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"text": "valid"}) + "\n")
        f.write(json.dumps({"id": "item2", "source": "data"}) + "\n")  # missing text
        path = f.name

    try:
        with pytest.raises(CliError) as exc_info:
            load_corpus(path)
        assert exc_info.value.code == EXIT_USER_ERROR
        assert "2" in exc_info.value.message  # line number
        assert "text" in exc_info.value.message.lower()
    finally:
        Path(path).unlink()


def test_load_corpus_missing_text_first_line() -> None:
    """Missing 'text' on first line raises CliError naming line 1."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"id": "item1"}) + "\n")  # missing text
        path = f.name

    try:
        with pytest.raises(CliError) as exc_info:
            load_corpus(path)
        assert exc_info.value.code == EXIT_USER_ERROR
        assert "1" in exc_info.value.message
    finally:
        Path(path).unlink()


def test_load_corpus_malformed_json() -> None:
    """Malformed JSON raises CliError with code 1 naming the line number."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"text": "valid"}) + "\n")
        f.write("{ not valid json }\n")  # malformed
        path = f.name

    try:
        with pytest.raises(CliError) as exc_info:
            load_corpus(path)
        assert exc_info.value.code == EXIT_USER_ERROR
        assert "2" in exc_info.value.message  # line number
    finally:
        Path(path).unlink()


def test_load_corpus_malformed_json_first_line() -> None:
    """Malformed JSON on first line raises CliError naming line 1."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write("not json at all\n")
        path = f.name

    try:
        with pytest.raises(CliError) as exc_info:
            load_corpus(path)
        assert exc_info.value.code == EXIT_USER_ERROR
        assert "1" in exc_info.value.message
    finally:
        Path(path).unlink()


def test_load_corpus_empty_file() -> None:
    """Empty corpus file returns empty list."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        path = f.name

    try:
        items = load_corpus(path)
        assert len(items) == 0
    finally:
        Path(path).unlink()


def test_load_corpus_id_as_non_string() -> None:
    """id field can be non-string and is converted to string."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"text": "hello", "id": 42}) + "\n")
        path = f.name

    try:
        items = load_corpus(path)
        assert items[0].id == "42"
    finally:
        Path(path).unlink()


def test_load_corpus_id_removed_from_metadata() -> None:
    """When id is provided, it is not duplicated in metadata."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps({"text": "hello", "id": "custom", "source": "test"}) + "\n")
        path = f.name

    try:
        items = load_corpus(path)
        assert items[0].id == "custom"
        assert "id" not in items[0].metadata
        assert items[0].metadata["source"] == "test"
    finally:
        Path(path).unlink()


def test_load_corpus_round_trip_preserves_all_keys() -> None:
    """Round-trip: metadata keys survive unchanged."""
    original_data = {
        "text": "content",
        "custom_field": "value",
        "nested": {"key": "value"},
        "array": [1, 2, 3],
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
        f.write(json.dumps(original_data) + "\n")
        path = f.name

    try:
        items = load_corpus(path)
        assert items[0].text == original_data["text"]
        assert items[0].metadata["custom_field"] == "value"
        assert items[0].metadata["nested"] == {"key": "value"}
        assert items[0].metadata["array"] == [1, 2, 3]
    finally:
        Path(path).unlink()


# --------------------------------------------------------------------------
# Duplicate ids (F1, found during the live end-to-end run)
#
# Ids are the join key for `neighbors --id`, `directions --pairs`, and
# compare's probe matching. A duplicate silently resolved to whichever record
# was read first, so every downstream lookup was wrong with nothing reporting
# it. Refuse at load time.
# --------------------------------------------------------------------------


def test_duplicate_ids_are_refused_naming_both_lines(tmp_path) -> None:
    corpus = tmp_path / "dupes.jsonl"
    corpus.write_text(
        '{"text": "first", "id": "shared"}\n'
        '{"text": "second", "id": "other"}\n'
        '{"text": "third", "id": "shared"}\n',
        encoding="utf-8",
    )
    with pytest.raises(CliError) as exc:
        load_corpus(str(corpus))
    assert exc.value.code == 1
    # Both the offending line and the first occurrence, so the fix is obvious.
    assert "3" in exc.value.message and "1" in exc.value.message
    assert "shared" in exc.value.message
    assert "unique" in exc.value.remediation


def test_auto_assigned_ids_never_collide(tmp_path) -> None:
    """Omitting 'id' must stay safe — line indices are unique by construction."""
    corpus = tmp_path / "auto.jsonl"
    corpus.write_text("\n".join('{"text": "item %d"}' % n for n in range(5)), encoding="utf-8")
    items = load_corpus(str(corpus))
    assert len({item.id for item in items}) == 5


def test_a_duplicate_of_an_auto_assigned_id_is_still_caught(tmp_path) -> None:
    """An explicit id colliding with an auto-assigned index must not slip through."""
    corpus = tmp_path / "mixed.jsonl"
    corpus.write_text('{"text": "auto gets id 0"}\n{"text": "explicit", "id": "0"}\n', "utf-8")
    with pytest.raises(CliError) as exc:
        load_corpus(str(corpus))
    assert exc.value.code == 1
