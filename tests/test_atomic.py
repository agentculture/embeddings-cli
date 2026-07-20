"""Tests for the atomic write-temp-then-rename helpers in ``embeddings._atomic``.

Covers the three acceptance criteria for this module: a simulated
interruption mid-write leaves no partial file, a simulated interruption
between batches of an ``atomic_dir`` write leaves no partial directory, and
concurrent writers of identical content-addressed payloads succeed without
corrupting the result.
"""

from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path

import pytest

from embeddings._atomic import atomic_dir, write_atomic

# --- write_atomic: basic behaviour -----------------------------------------


def test_write_atomic_creates_file_with_expected_content(tmp_path: Path) -> None:
    target = tmp_path / "out.bin"
    write_atomic(target, b"hello atlas")
    assert target.read_bytes() == b"hello atlas"


def test_write_atomic_creates_parent_directories(tmp_path: Path) -> None:
    target = tmp_path / "cache" / "nested" / "entry.bin"
    write_atomic(target, b"payload")
    assert target.read_bytes() == b"payload"


def test_write_atomic_overwrites_existing_target(tmp_path: Path) -> None:
    target = tmp_path / "entry.bin"
    write_atomic(target, b"first value")
    write_atomic(target, b"second value")
    assert target.read_bytes() == b"second value"


def test_write_atomic_leaves_no_temp_artifacts_behind(tmp_path: Path) -> None:
    target = tmp_path / "entry.bin"
    write_atomic(target, b"payload")
    assert list(tmp_path.iterdir()) == [target]


# --- write_atomic: simulated interruption mid-write ------------------------


def test_write_atomic_interrupted_mid_write_leaves_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A crash between the temp-file write and the rename must not create ``path``."""
    target = tmp_path / "cache" / "entry.bin"

    def _boom(_fd: int) -> None:
        raise OSError("simulated kill mid-write")

    monkeypatch.setattr(os, "fsync", _boom)

    with pytest.raises(OSError):
        write_atomic(target, b"partial payload that must never land")

    assert not target.exists()
    # the temp file created before the simulated crash must be cleaned up too
    assert target.parent.exists()
    assert list(target.parent.iterdir()) == []


def test_write_atomic_interrupted_write_preserves_previous_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An interrupted overwrite must leave the last complete value in place."""
    target = tmp_path / "entry.bin"
    write_atomic(target, b"first complete value")

    def _boom(_fd: int) -> None:
        raise OSError("simulated crash before replace")

    monkeypatch.setattr(os, "fsync", _boom)

    with pytest.raises(OSError):
        write_atomic(target, b"second value that never completes")

    assert target.read_bytes() == b"first complete value"
    assert list(tmp_path.iterdir()) == [target]


# --- atomic_dir: basic behaviour --------------------------------------------


def test_atomic_dir_materializes_on_clean_exit(tmp_path: Path) -> None:
    target = tmp_path / "atlas"
    assert not target.exists()

    with atomic_dir(target) as scratch:
        assert scratch.is_dir()
        assert scratch != target
        (scratch / "regions.json").write_bytes(b'{"regions": []}')
        (scratch / "directions.json").write_bytes(b'{"directions": []}')

    assert target.is_dir()
    assert (target / "regions.json").read_bytes() == b'{"regions": []}'
    assert (target / "directions.json").read_bytes() == b'{"directions": []}'
    # nothing else left behind alongside the materialized directory
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_dir_target_not_visible_before_clean_exit(tmp_path: Path) -> None:
    target = tmp_path / "atlas"

    with atomic_dir(target) as scratch:
        (scratch / "partial.json").write_bytes(b"{}")
        assert not target.exists()  # not yet swapped into place

    assert target.is_dir()


# --- atomic_dir: simulated interruption between batches --------------------


def _write_one_batch_then_interrupt(target: Path, batches_written: list[str]) -> None:
    """Write one "batch" into the scratch dir, then raise -- simulating a crash
    partway through a multi-batch ``atomic_dir`` write. Kept as its own
    function (rather than inline in a ``pytest.raises`` block) so the block
    itself contains exactly one call: this one."""
    with atomic_dir(target) as scratch:
        (scratch / "regions.json").write_bytes(b'{"regions": []}')
        batches_written.append("regions")
        raise RuntimeError("simulated interruption after first batch")


def test_atomic_dir_interrupted_between_batches_leaves_no_target(tmp_path: Path) -> None:
    """Raising after writing one file ("batch") of several must leave no target directory."""
    target = tmp_path / "atlas"
    batches_written: list[str] = []

    with pytest.raises(RuntimeError, match="simulated interruption"):
        _write_one_batch_then_interrupt(target, batches_written)

    assert batches_written == ["regions"]  # got partway through, as intended
    assert not target.exists()
    # the scratch directory used for the aborted write must be gone too
    assert list(tmp_path.iterdir()) == []


def _overwrite_batch_then_interrupt(target: Path) -> None:
    """Start replacing an existing ``atomic_dir`` target, then raise before it
    completes. Kept as its own function so the ``pytest.raises`` block below
    contains exactly one call: this one."""
    with atomic_dir(target) as scratch:
        (scratch / "regions.json").write_bytes(b"second atlas, never completes")
        raise RuntimeError("simulated interruption")


def test_atomic_dir_interrupted_preserves_previous_atlas(tmp_path: Path) -> None:
    """A killed re-survey must not disturb a previously-materialized Atlas."""
    target = tmp_path / "atlas"
    with atomic_dir(target) as scratch:
        (scratch / "regions.json").write_bytes(b"first atlas")

    with pytest.raises(RuntimeError):
        _overwrite_batch_then_interrupt(target)

    assert (target / "regions.json").read_bytes() == b"first atlas"


# --- concurrency: identical content-addressed payloads ----------------------


def test_write_atomic_concurrent_writers_identical_content_no_corruption(
    tmp_path: Path,
) -> None:
    """Two writers racing on the SAME content-addressed key must not corrupt the result.

    This is the shape of the embedding cache: the key is
    ``(model, sha256(text))``, so two writers racing on the same key
    necessarily agree on the bytes. ``os.replace`` makes the race benign —
    whichever writer's rename lands last leaves byte-identical content.
    """
    payload = b"vector cache entry payload " * 1000
    digest = hashlib.sha256(payload).hexdigest()
    target = tmp_path / "cache" / f"{digest}.bin"

    errors: list[BaseException] = []
    barrier = threading.Barrier(16)

    def _writer() -> None:
        try:
            barrier.wait()  # maximize overlap between writers
            write_atomic(target, payload)
        except BaseException as exc:  # noqa: BLE001 - captured for the assertion below
            errors.append(exc)

    threads = [threading.Thread(target=_writer) for _ in range(16)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert errors == []
    assert target.read_bytes() == payload
    # no stray temp files left in the cache directory once every writer is done
    leftovers = [p for p in target.parent.iterdir() if p != target]
    assert leftovers == []


def test_write_atomic_concurrent_writers_repeated_rounds_stay_clean(tmp_path: Path) -> None:
    """Repeat the race across several rounds to widen the window for corruption."""
    target = tmp_path / "shared.bin"

    for round_num in range(10):
        payload = f"round {round_num} payload".encode()
        errors: list[BaseException] = []

        def _writer() -> None:
            try:
                write_atomic(target, payload)
            except BaseException as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=_writer) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []
        assert target.read_bytes() == payload
        assert list(tmp_path.iterdir()) == [target]


# --- atomic_dir: a failing replace must not leak the scratch directory ------
#
# ``atomic_dir`` documents that it cannot swap onto a *non-empty* existing
# target (POSIX rename(2) refuses), and leaves handling that to the caller.
# But when the replace does fail, the scratch directory is a full copy of
# whatever the block just built — an entire atlas, potentially — and it must
# not be left behind on disk under a dotted name nobody ever looks at.


def test_atomic_dir_failed_replace_removes_the_scratch_directory(tmp_path: Path) -> None:
    target = tmp_path / "atlas"
    target.mkdir()
    (target / "occupant.txt").write_text("pre-existing, non-empty", encoding="utf-8")

    def _build_into_an_occupied_target() -> None:
        # The failure under test is the *replace* at the end of atomic_dir, not
        # anything in here — extracted so pytest.raises wraps exactly one call.
        with atomic_dir(target) as tmp:
            (tmp / "records.jsonl").write_text("built but never swapped", encoding="utf-8")

    with pytest.raises(OSError):
        _build_into_an_occupied_target()

    leftovers = [p for p in tmp_path.iterdir() if p.name.startswith(".atlas")]
    assert leftovers == [], f"scratch directory leaked after a failed replace: {leftovers}"
    # The pre-existing target is untouched.
    assert (target / "occupant.txt").read_text(encoding="utf-8") == "pre-existing, non-empty"
