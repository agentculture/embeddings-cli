"""Atomic write-temp-then-rename helpers.

A survey run writes an Atlas directory and populates a content-addressed
embedding cache. Agents call this CLI in loops and get killed at arbitrary
points — SIGKILL, an OOM, a crashed process, an exception mid-run. A killed
run must never leave a half-written Atlas or a corrupt cache entry that a
later run would happily read back as valid.

Both helpers follow the same pattern: build the complete result in a scratch
location that is a *sibling* of the target (same directory, therefore the same
filesystem), then swap it into place with a single ``os.replace``. POSIX
guarantees ``rename(2)`` — what ``os.replace`` uses — is atomic when both
paths share a filesystem, so there is no window in which an observer can see a
half-written result: the target either does not exist yet, holds the previous
complete result, or holds the new complete result.

* :func:`write_atomic` — atomically write a single ``bytes`` payload to a file.
* :func:`atomic_dir` — a context manager yielding a scratch directory that is
  swapped onto the target path only if the ``with`` block exits cleanly.

**What is and is not promised: crash-atomicity, not power-loss durability.**
The guarantee above is scoped to the failure this tool actually meets — a
*process* dying (SIGKILL, OOM, an exception, a killed agent loop). Against
that, ``os.replace`` is sufficient on its own: the rename is visible to every
later process the instant it returns, whether or not it has reached the
platter. Neither helper fsyncs the *parent directory* after the rename, so
what these functions do not promise is that a successful return survives a
power loss or kernel panic — such a crash can roll the directory entry back to
the previous complete value. That is deliberate. Everything written here is
regenerable: an atlas by re-running ``embeddings survey``, a cache entry by
re-embedding its text. Paying a directory fsync per write to protect against
losing the last few seconds of regenerable work would cost one synchronous
disk flush *per cached text* on a survey — thousands of them on a real corpus
— to buy a property no caller needs. Nothing here is a transaction log, and no
caller treats "returned" as "committed" to anything outside this process. If a
future caller ever does need power-loss durability, the fix is a post-rename
``os.fsync`` on a parent-directory fd (POSIX only), added there deliberately
rather than assumed here.

Two writers racing to write *identical* content to the same path — the shape
of a content-addressed cache keyed by ``(model, sha256(text))``, where both
writers necessarily agree on the bytes — is safe by construction: each writer
builds its own temp file (unique names, see below) before either replace
happens, and whichever ``os.replace`` lands last leaves byte-identical content
behind. This module makes the swap atomic; it does not adjudicate who wins
when two writers disagree on content for the same path.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import tempfile
from pathlib import Path
from typing import Iterator


def write_atomic(path: str | Path, data: bytes) -> None:
    """Atomically write ``data`` to ``path``.

    Writes to a uniquely-named temp file in ``path``'s own parent directory
    (creating that directory if needed), fsyncs the temp file so its bytes
    reach the disk before the rename, then ``os.replace``s it onto ``path``.

    If anything raises before the replace — a write error, an fsync error, an
    async exception such as ``KeyboardInterrupt`` delivered mid-call — the
    temp file is removed and the exception propagates; ``path`` is left
    exactly as it was (absent, or holding its previous complete value). Every
    temp file uses a name unique to this call (:func:`tempfile.mkstemp`), so
    concurrent callers writing the same ``path`` never collide on temp files,
    only on the final, atomic replace.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


@contextlib.contextmanager
def atomic_dir(path: str | Path) -> Iterator[Path]:
    """Context manager yielding a scratch directory swapped onto ``path`` on success.

    Usage::

        with atomic_dir(atlas_dir) as tmp:
            (tmp / "regions.json").write_bytes(regions_bytes)
            (tmp / "directions.json").write_bytes(directions_bytes)
        # atlas_dir now exists, complete. If the block above had raised
        # instead, atlas_dir would not exist at all.

    The scratch directory is created as a sibling of ``path`` (same parent
    directory, hence same filesystem, so the final swap is a single atomic
    ``os.replace``) and is moved onto ``path`` only when the ``with`` block
    exits without raising. Any exception raised inside the block — including
    one raised deliberately partway through writing several files ("between
    batches") — is left to propagate after the scratch directory is removed;
    ``path`` never comes into existence in that case, and a previously
    materialized ``path`` from an earlier successful call is left untouched.

    Note: this assumes ``path`` does not already exist, or exists as an empty
    directory — ``os.replace`` cannot atomically swap a directory onto a
    non-empty one (POSIX ``rename(2)``). Removing or renaming a pre-existing,
    non-empty ``path`` before calling is the caller's responsibility.
    """
    target = Path(path)
    parent = target.parent
    parent.mkdir(parents=True, exist_ok=True)
    tmp_dir = Path(tempfile.mkdtemp(dir=parent, prefix=f".{target.name}.", suffix=".tmp"))
    try:
        yield tmp_dir
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
    else:
        # The replace itself can fail — most commonly ENOTEMPTY, when ``path``
        # already exists non-empty (see the note above; resolving that is the
        # caller's job). The scratch directory holds a complete copy of
        # whatever the block just built — an entire atlas, potentially — so it
        # must not be left behind under a dotted name nobody ever looks at.
        try:
            os.replace(tmp_dir, target)
        except BaseException:
            shutil.rmtree(tmp_dir, ignore_errors=True)
            raise
