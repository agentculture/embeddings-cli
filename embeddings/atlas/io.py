"""Atlas read/write: the persistence layer around ``embeddings/atlas/schema.py``.

An Atlas is a directory holding three artifacts, split deliberately along the
interpretation/bulk line the module docstring in :mod:`embeddings.atlas.schema`
describes:

* ``atlas.json`` — the manifest (:class:`~embeddings.atlas.schema.AtlasManifest`),
  serialized as one indented JSON object. Small, diffable, and the thing a
  reviewer actually reads.
* ``records.jsonl`` — one JSON object per line, one line per
  :class:`~embeddings.atlas.schema.AtlasRecord`. Diffable line-by-line; a
  survey that adds one item changes one line.
* ``vectors.npy`` — every vector the survey embedded, as a single 2D
  ``float32``-or-whatever-the-caller-passed NumPy array of shape
  ``(n_vectors, dim)``. ``AtlasRecord.vector_index`` is a row index into this
  array; more than one record may share a ``vector_index`` (e.g. two corpus
  items with identical text embedded once), so the array's row count is not
  required to equal the record count.

**Why one vector file, not sharded sidecars.** The schema module leaves
sidecar batching/naming to this layer. A single array keeps the
"``vector_index`` is a row index" contract a single integer instead of a
``(shard, offset)`` pair, and it keeps read/write each a single
``numpy.save``/``numpy.load`` call. Sharding would only start to matter at a
scale this tool doesn't target: the design note's own worked example is a
~40MB array for a 10k-item corpus at 1024 dimensions, comfortably below where
a single ``.npy`` file becomes awkward to read or diff-summarize.

**Why the manifest and records are committable and the vectors are not.** The
Atlas is an evidence-backed interpretation of a surveyed embedding space, not
the space itself (see ``docs/design-notes/semantic-measurement.md``). The
manifest and records are that interpretation — a reviewer can read and diff
them. The vectors are regenerable bulk: re-running ``embeddings survey``
against the same corpus and model reproduces them (exactly, from the embed
cache; approximately, from a live re-embed — see the manifest's
``limitations``). ``.gitignore`` excludes ``*.npy`` for exactly this reason;
see the comment there for the paired explanation on the other side of that
line.

**Atomicity.** ``write_atlas`` builds the complete directory in a scratch
location and swaps it into place with :func:`embeddings._atomic.atomic_dir`,
so a reader never observes a half-written atlas: at every point in time the
target path is either absent, holds the previous complete atlas, or holds the
new complete atlas.

**Replacing an existing atlas.** ``atomic_dir`` documents that it cannot
atomically replace a *non-empty* existing target (POSIX ``rename(2)`` cannot
swap a directory onto one that already has contents). ``write_atlas`` handles
this with an explicit ``overwrite`` flag:

* ``overwrite=False`` (the default) — if an atlas already exists at ``path``,
  raise :class:`~embeddings.cli._errors.CliError` (exit 1, user error) rather
  than silently doing anything to it. This is the safe default for any direct
  caller.
* ``overwrite=True`` — remove the existing directory first (``shutil.rmtree``),
  then build and swap in the replacement as above. The old atlas is replaced
  *whole*, never merged with the new one. Note this makes the removal and the
  swap two separate steps rather than one atomic operation: there is a brief
  window in which ``path`` does not exist at all. That window trades "briefly
  absent" for "atomically correct once present again" — it never produces a
  directory that is a mix of the old and new atlas, which is the actual
  invariant this module promises. A process killed inside that window leaves
  no atlas, which a reader reports as "nothing surveyed yet" (exit 1) rather
  than silently misreporting stale or mixed data.

The CLI-facing ``--force`` flag and its exact hint text belong to the
``survey`` verb (a later task) built on top of this ``overwrite`` parameter;
this module only provides the primitive.

**The version gate.** ``read_atlas`` compares the manifest's on-disk
``schema_version`` against :data:`embeddings.atlas.schema.SCHEMA_VERSION`
*before* doing anything else — before parsing ``records.jsonl``, before
loading ``vectors.npy``. Any mismatch, whether the on-disk version is older,
newer, or simply not a version this reader has ever heard of, fails closed
with :class:`~embeddings.cli._errors.CliError` (exit 2, environment/setup): an
old CLI meeting a newer Atlas must refuse loudly, never best-effort-parse and
silently misreport a measurement.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from embeddings._atomic import atomic_dir
from embeddings.atlas.schema import SCHEMA_VERSION, AtlasManifest, AtlasRecord
from embeddings.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError

#: Default atlas directory, relative to the caller's working directory.
#: Every function here accepts ``path`` to override it.
DEFAULT_ATLAS_DIR = Path("atlas")

#: Filenames within an atlas directory. Exported so callers (tests, other
#: modules) never have to hardcode them.
MANIFEST_FILENAME = "atlas.json"
RECORDS_FILENAME = "records.jsonl"
VECTORS_FILENAME = "vectors.npy"


@dataclass(frozen=True)
class Atlas:
    """An Atlas loaded into memory: the manifest, its records, and their vectors.

    Attributes:
        manifest: the loaded :class:`AtlasManifest`.
        records: every :class:`AtlasRecord` from ``records.jsonl``, in file
            order.
        vectors: the ``(n_vectors, dim)`` array from ``vectors.npy``.
            ``records[i].vector_index`` is a row index into this array.
        path: the directory this atlas was loaded from.
    """

    manifest: AtlasManifest
    records: list[AtlasRecord]
    vectors: np.ndarray
    path: Path


def _manifest_to_json_bytes(manifest: AtlasManifest) -> bytes:
    # Field order follows the dataclass declaration (schema_version, model,
    # endpoint, params, seed, limitations, uncertain) — stable across calls
    # given the same manifest, which is what makes an unchanged manifest
    # rewrite byte-identical.
    text = json.dumps(asdict(manifest), indent=2, ensure_ascii=False) + "\n"
    return text.encode("utf-8")


def _records_to_jsonl_bytes(records: Sequence[AtlasRecord]) -> bytes:
    lines = [json.dumps(asdict(record), ensure_ascii=False) for record in records]
    if not lines:
        return b""
    return ("\n".join(lines) + "\n").encode("utf-8")


def _validate_vectors_and_indices(records: Sequence[AtlasRecord], vectors: np.ndarray) -> None:
    if vectors.ndim != 2:
        raise ValueError(
            f"vectors must be a 2D array of shape (n_vectors, dim); got shape {vectors.shape}"
        )
    if not records:
        return
    indices = [record.vector_index for record in records]
    low, high = min(indices), max(indices)
    if low < 0 or high >= vectors.shape[0]:
        raise ValueError(
            "AtlasRecord.vector_index out of bounds for vectors array of shape "
            f"{vectors.shape}: indices range [{low}, {high}], valid range is "
            f"[0, {vectors.shape[0] - 1}]"
        )


def write_atlas(
    manifest: AtlasManifest,
    records: Sequence[AtlasRecord],
    vectors: np.ndarray,
    path: str | Path = DEFAULT_ATLAS_DIR,
    *,
    overwrite: bool = False,
) -> Path:
    """Persist ``manifest``, ``records``, and ``vectors`` as an atlas at ``path``.

    Writes ``atlas.json``, ``records.jsonl``, and ``vectors.npy`` into a
    single directory, materialized atomically via
    :func:`embeddings._atomic.atomic_dir` — a failure at any point during the
    write leaves ``path`` exactly as it was before this call (absent, or
    holding whatever it held previously).

    Args:
        manifest: the atlas manifest to write verbatim (its
            ``schema_version`` is written as-is; callers writing a fresh
            atlas should pass ``schema_version=SCHEMA_VERSION`` themselves —
            this function does not stamp it for them, so that it can also be
            used to reproduce a specific historical version in tests).
        records: the per-item records; may be empty.
        vectors: a 2D array of shape ``(n_vectors, dim)``. Every
            ``record.vector_index`` must be a valid row index into it;
            multiple records may share the same index.
        path: directory to write the atlas into. Defaults to
            :data:`DEFAULT_ATLAS_DIR` (``./atlas``).
        overwrite: if ``False`` (default) and an atlas already exists at
            ``path``, raise :class:`CliError` rather than touch it. If
            ``True``, remove the existing atlas first and write the
            replacement whole — see the module docstring for the atomicity
            caveat this implies.

    Returns:
        ``path``, as a :class:`~pathlib.Path`.

    Raises:
        ValueError: ``vectors`` is not 2D, or a record's ``vector_index`` is
            out of bounds for it. Raised before anything is written.
        CliError: an atlas already exists at ``path`` and ``overwrite`` is
            ``False`` (exit 1, user error).
    """
    target = Path(path)
    vectors = np.asarray(vectors)
    _validate_vectors_and_indices(records, vectors)

    if target.exists():
        if not overwrite:
            raise CliError(
                code=EXIT_USER_ERROR,
                message=f"an atlas already exists at {target}",
                remediation=(
                    "pass overwrite=True to replace it whole (never merged); the "
                    "survey CLI verb exposes this as --force"
                ),
            )
        shutil.rmtree(target)

    with atomic_dir(target) as tmp:
        (tmp / MANIFEST_FILENAME).write_bytes(_manifest_to_json_bytes(manifest))
        (tmp / RECORDS_FILENAME).write_bytes(_records_to_jsonl_bytes(records))
        np.save(tmp / VECTORS_FILENAME, vectors)

    return target


def read_atlas(path: str | Path = DEFAULT_ATLAS_DIR) -> Atlas:
    """Load the atlas at ``path``, failing closed on a ``schema_version`` mismatch.

    The on-disk ``schema_version`` is checked against
    :data:`embeddings.atlas.schema.SCHEMA_VERSION` immediately after reading
    ``atlas.json`` and before anything else is parsed. On any mismatch —
    older, newer, or simply unrecognized — this raises :class:`CliError`
    (exit 2) without attempting to read ``records.jsonl`` or ``vectors.npy``
    at all: this reader never best-effort-parses an atlas it does not
    understand.

    Args:
        path: directory to read the atlas from. Defaults to
            :data:`DEFAULT_ATLAS_DIR` (``./atlas``).

    Returns:
        The loaded :class:`Atlas`.

    Raises:
        CliError: no atlas manifest exists at ``path`` (exit 1, user error);
            the atlas has records but no ``vectors.npy`` (exit 1 — the sidecar
            is git-ignored by design, so a fresh clone must regenerate it; see
            the comment at the check); or the manifest's ``schema_version``
            does not match this embeddings-cli's :data:`SCHEMA_VERSION`
            (exit 2, environment/setup); or ``atlas.json``/``records.jsonl``
            contain malformed JSON, or ``vectors.npy`` is present but does not
            back the records (exit 2 — a corrupt committed artifact is an
            environment problem, not a usage mistake).
    """
    target = Path(path)
    manifest_path = target / MANIFEST_FILENAME

    if not manifest_path.is_file():
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"no atlas found at {target}",
            remediation=(
                "run `embeddings survey <corpus> --apply` first, or pass the correct atlas path"
            ),
        )

    try:
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as err:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=f"atlas manifest at {manifest_path} is not valid JSON: {err}",
            remediation="restore this atlas from git or re-run the survey that produced it",
        ) from err

    on_disk_version = manifest_payload.get("schema_version")
    if on_disk_version != SCHEMA_VERSION:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=(
                f"atlas at {target} was written with schema_version={on_disk_version!r}, "
                f"but this embeddings-cli only reads schema_version={SCHEMA_VERSION!r}"
            ),
            remediation=(
                "install the embeddings-cli release that wrote "
                f"schema_version={on_disk_version!r} to read this atlas, or re-run "
                f"`embeddings survey ... --apply --force` with this embeddings-cli to "
                f"rewrite it at schema_version={SCHEMA_VERSION!r}"
            ),
        )

    # Past this point the schema_version is known-good; only now do we trust
    # the rest of the manifest shape and go on to parse records/vectors.
    manifest = AtlasManifest(**manifest_payload)

    records: list[AtlasRecord] = []
    records_path = target / RECORDS_FILENAME
    if records_path.is_file():
        for line_num, line in enumerate(records_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line:
                continue
            try:
                record_payload = json.loads(line)
            except json.JSONDecodeError as err:
                raise CliError(
                    code=EXIT_ENV_ERROR,
                    message=f"malformed record on line {line_num} of {records_path}: {err}",
                    remediation="restore this atlas from git or re-run the survey that produced it",
                ) from err
            records.append(AtlasRecord(**record_payload))

    vectors_path = target / VECTORS_FILENAME
    if vectors_path.is_file():
        vectors = np.load(vectors_path, allow_pickle=False)
    elif records:
        # Records on disk but no vectors: the atlas is incomplete, and the
        # reader must say so here rather than hand back a (0, 0) matrix that
        # turns every `atlas.vectors[record.vector_index]` lookup downstream
        # into a bare IndexError several call frames away from the cause.
        #
        # Exit 1 (user), not 2 (environment), and the choice is deliberate:
        # `*.npy` is git-ignored on purpose — the manifest and records are the
        # committable interpretation, the vectors are regenerable bulk — so
        # "records present, vectors absent" is the *expected* state of a fresh
        # clone of a repo carrying a committed atlas. Nothing is broken or
        # corrupt; the operator simply has an artifact that has not been
        # materialized on this machine yet, which is the same condition as the
        # already-exit-1 "no atlas found" case one screen up, with the same
        # fix (run the survey). Exit 2 is reserved here for on-disk state that
        # re-running cannot explain away — an unreadable manifest, a schema
        # this build cannot parse, or a vectors.npy that is present but
        # inconsistent with the records (see the bounds check below). Keeping
        # "not generated yet" at 1 also keeps an agent's retry logic uniform:
        # exit 1 plus a "run survey" hint always means run the survey.
        raise CliError(
            code=EXIT_USER_ERROR,
            message=(f"atlas at {target} has {len(records)} record(s) but no {VECTORS_FILENAME}"),
            remediation=(
                f"{VECTORS_FILENAME} is git-ignored by design (regenerable bulk, unlike the "
                "committed manifest and records), so a fresh clone never carries it: re-run "
                "`embeddings survey <corpus> --apply --force` against the same corpus and "
                "model to regenerate the vectors for this atlas"
            ),
        )
    else:
        # Zero records and no vectors is a legitimately *empty* atlas, not an
        # incomplete one — there is nothing to index into.
        vectors = np.zeros((0, 0), dtype=np.float32)

    # The vectors file exists; check it actually backs the records before any
    # caller indexes into it. A present-but-inconsistent array is corruption
    # (not the by-design absence above), so it fails at exit 2.
    try:
        _validate_vectors_and_indices(records, vectors)
    except ValueError as err:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=(
                f"atlas at {target} has a {VECTORS_FILENAME} "
                f"inconsistent with its records: {err}"
            ),
            remediation=(
                "restore this atlas from git or re-run `embeddings survey <corpus> --apply "
                "--force` to rewrite it whole"
            ),
        ) from err

    return Atlas(manifest=manifest, records=records, vectors=vectors, path=target)
