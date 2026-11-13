"""``embeddings export`` — write an Atlas with its vectors as an embeddings-tumbler data file.

`embeddings-tumbler <https://github.com/agentculture/embeddings-tumbler>`_ is a
drop-in viewer that shows an embedding space through a moving 3-D window. It
reads a two-file data set defined by its data contract (schema version 1): a
``<name>.json`` record part and a sibling ``<name>.f32`` vector block of
little-endian float32 values, row-major, exactly ``count x dimension x 4``
bytes, with no header. This verb is the producer step: it reads a surveyed
Atlas (``atlas.json`` + ``records.jsonl`` + ``vectors.npy``) and writes that
pair. It needs no gateway and opens no socket.

What is copied, and what is not
--------------------------------
* ``model``, ``seed``, ``limitations``, ``uncertain`` and ``params`` are copied
  from the Atlas manifest into ``provenance``. ``limitations`` and ``uncertain``
  are copied entry for entry, never dropped.
* **The Atlas's endpoint URL is never copied.** The viewer's rule is that the
  embedding gateway's address never reaches a page, so the contract allows
  ``provenance.endpoint`` only as a route *name* matching
  ``^[A-Za-z0-9][A-Za-z0-9 _-]{0,63}$`` (ASCII letters, digits, spaces, ``_``
  and ``-``; 1 to 64 characters; no dots, colons or slashes, so no hostname,
  URL or port). The name is taken only from ``--endpoint-name`` and checked
  against that allowlist; without the option the field is left out (it is
  optional). No name is ever derived from the URL. If the URL, its origin, or
  its ``host:port`` appears inside a copied provenance string (a limitation, a
  parameter value), that occurrence is replaced with ``<embedding endpoint>``;
  the entry itself stays. Corpus text is data, not provenance, and is not
  rewritten — ``--labels-only`` is the way to leave it out.
* Each record becomes one point, in record order: ``id``, ``label`` = the
  record's ``text``, ``metadata`` = the record's metadata. Records that share a
  ``vector_index`` each get their own row. No ``group`` is written.
* ``--labels-only`` leaves the corpus out: ``label`` = ``id``, **no metadata at
  all** (corpus lines carry their extra keys into metadata, and those can be
  text too, so the safe default is to drop it rather than guess which keys are
  harmless), and ``text_included`` is ``false``.
* ``provenance.source`` is ``atlas: <folder name>`` -- the Atlas folder's own
  name, never the path as given, so no local absolute path reaches a page.
* Both files get mode 0644 reduced by the umask (0644 under 022, 0640 under
  027, never group or other write), not ``write_atomic``'s private 0600,
  because a static web server running as another user must be able to read
  them.
* ``provenance.created`` is not written, so the same Atlas and options give a
  byte-identical pair.

Reduction
---------
Off unless requested; the file then carries an explicit ``kind: "none"`` record.

* ``--rotate`` — an exact orthogonal change of basis onto the set's principal
  directions, after subtracting the set's mean. The basis is the full set of
  eigenvectors of the ``dimension x dimension`` scatter matrix, so the block
  keeps every dimension and every pairwise distance; ``variance_dropped`` is 0.
  Coordinate 0 is the direction of greatest variance, and so on down. Each
  direction's sign is pinned (largest-magnitude loading positive) so the output
  is deterministic.
* ``--top-k K`` — the same rotation, then only the first ``K`` coordinates are
  kept. ``variance_dropped`` is the share of the set's total variance outside
  them. ``K`` must be below the source dimension. This is an approximation of
  the space, and the record says so; the viewer labels it as one.

Write gating
------------
Previews by default and writes nothing; ``--apply`` writes; an existing
``.json`` *or* ``.f32`` at the target is refused (exit 1) with a hint naming
``--force``. Each file is written through :func:`embeddings._atomic.write_atomic`
(temp file + ``os.replace``), the block first and the JSON that names it
second, after any existing JSON has been removed. Each file is atomic on its
own; the pair is not one transaction, but a failure or kill at any step leaves
either the complete new pair or a block with no JSON -- never a JSON beside a
block that does not belong to it.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from embeddings import __version__
from embeddings._atomic import write_atomic
from embeddings.atlas.io import Atlas
from embeddings.cli._commands._atlas_arg import add_atlas_argument, load_atlas
from embeddings.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from embeddings.cli._output import emit_diagnostic, emit_result
from embeddings.cli._provenance import REPRODUCED_BY_KEY, reproduce_line, reproducing_command

#: The data contract this verb writes. A different viewer schema needs a code change here.
SCHEMA_VERSION = 1
KIND = "embeddings-tumbler/data"
#: Default output, relative to the caller's working directory.
DEFAULT_OUT = "viewer-data.json"
BLOCK_SUFFIX = ".f32"
#: What a redacted endpoint occurrence inside copied provenance becomes.
ENDPOINT_PLACEHOLDER = "<embedding endpoint>"

#: The contract's "plain file name" rule for ``vectors.file``.
PLAIN_FILE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
#: The contract's allowlist for ``provenance.endpoint``; anything else is ``endpoint-is-address``.
ENDPOINT_NAME_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9 _-]{0,63}")


# --------------------------------------------------------------------------
# Argument checks (before the atlas is read)
# --------------------------------------------------------------------------


def _targets(out: str) -> tuple[Path, Path]:
    json_path = Path(out)
    block_name = json_path.stem + BLOCK_SUFFIX
    if json_path.suffix != ".json" or not PLAIN_FILE_NAME.match(json_path.name):
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"--out must name a .json file with a plain file name, got {out!r}",
            remediation=(
                "pass --out <name>.json where <name> starts with a letter or digit and holds "
                "only letters, digits, '.', '_' and '-' (the vector block is written beside "
                f"it as <name>{BLOCK_SUFFIX})"
            ),
        )
    if not PLAIN_FILE_NAME.match(block_name):  # pragma: no cover - implied by the check above
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"the vector block name {block_name!r} is not a plain file name",
            remediation="choose an --out name of letters, digits, '.', '_' and '-'",
        )
    return json_path, json_path.with_name(block_name)


def _endpoint_name(raw: str | None) -> str | None:
    if raw is None:
        return None
    if ENDPOINT_NAME_PATTERN.fullmatch(raw) is None:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=(
                f"--endpoint-name {raw!r} is not a route name the viewer accepts: it must match "
                "^[A-Za-z0-9][A-Za-z0-9 _-]{0,63}$"
            ),
            remediation=(
                "give the route a plain name of 1 to 64 ASCII letters, digits, spaces, '_' or "
                "'-', starting with a letter or digit -- no dots, colons or slashes, so no "
                "hostname, URL or port -- for example --endpoint-name 'lobes gateway'; or "
                "leave the option out to omit the endpoint"
            ),
        )
    return raw


# --------------------------------------------------------------------------
# The data set
# --------------------------------------------------------------------------


def _item_matrix(atlas: Atlas) -> np.ndarray:
    """One float64 row per record, in record order (records may share a vector)."""
    if not atlas.records:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"the atlas at {atlas.path} contains no records -- there is nothing to export",
            remediation="re-run `embeddings survey <corpus> --apply` with a non-empty corpus",
        )
    seen: set[str] = set()
    for record in atlas.records:
        if record.id in seen:
            raise CliError(
                code=EXIT_USER_ERROR,
                message=f'the atlas at {atlas.path} has record id "{record.id}" more than once; '
                "the viewer needs unique point ids",
                remediation="give each corpus item a unique id and re-run `embeddings survey`",
            )
        seen.add(record.id)
    indices = [record.vector_index for record in atlas.records]
    matrix = np.asarray(atlas.vectors[indices], dtype=np.float64)
    finite = np.isfinite(matrix).all(axis=1)
    if not finite.all():
        bad = atlas.records[int(np.argmin(finite))].id
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=f'the vector of record "{bad}" in {atlas.path} holds a non-finite value',
            remediation=(
                "re-run `embeddings survey <corpus> --apply --force` to rebuild the vectors"
            ),
        )
    return matrix


def principal_rotation(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Mean-centre ``matrix`` and rotate it onto its principal directions.

    Returns ``(coords, variances)``: ``coords`` has the same shape as
    ``matrix`` (a full orthogonal basis, so no dimension is lost), and
    ``variances[j]`` is the scatter along coordinate ``j``, in decreasing order.
    The basis is the eigenvectors of the ``D x D`` scatter matrix, which is
    complete even when there are fewer points than dimensions.
    """
    centred = matrix - matrix.mean(axis=0, keepdims=True)
    scatter = centred.T @ centred
    eigenvalues, eigenvectors = np.linalg.eigh(scatter)
    order = np.argsort(eigenvalues, kind="stable")[::-1]
    eigenvalues = np.clip(eigenvalues[order], 0.0, None)
    basis = eigenvectors[:, order]
    pivots = np.argmax(np.abs(basis), axis=0)
    signs = np.where(basis[pivots, np.arange(basis.shape[1])] < 0.0, -1.0, 1.0)
    basis = basis * signs
    return centred @ basis, eigenvalues


def _reduce(matrix: np.ndarray, *, rotate: bool, top_k: int | None) -> tuple[np.ndarray, dict]:
    source_dimension = int(matrix.shape[1])
    if top_k is not None and not 1 <= top_k < source_dimension:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=(
                f"--top-k {top_k} is outside 1..{source_dimension - 1} for a "
                f"{source_dimension}-d atlas"
            ),
            remediation=(
                f"pass --top-k <k> with 1 <= k < {source_dimension}, or use --rotate for an "
                "exact rotation that keeps all directions"
            ),
        )
    if top_k is None and not rotate:
        kind, coords, dropped = "none", matrix, 0
    else:
        coords, variances = principal_rotation(matrix)
        if top_k is None:
            # Exact by construction (an orthogonal basis); float32 rounding is not counted.
            kind, dropped = "rotation", 0
        else:
            total = float(variances.sum())
            kept = float(variances[:top_k].sum())
            dropped = min(max(1.0 - kept / total, 0.0), 1.0) if total > 0.0 else 0.0
            kind, coords = "top-k", coords[:, :top_k]
    dimension = int(coords.shape[1])
    return coords, {
        "kind": kind,
        "source_dimension": source_dimension,
        "directions_kept": dimension,
        "variance_dropped": dropped,
    }


def _redact(value: object, needles: list[str]) -> object:
    """Replace each endpoint form in every string inside ``value`` (lists, dicts, scalars)."""
    if isinstance(value, str):
        for needle in needles:
            value = value.replace(needle, ENDPOINT_PLACEHOLDER)
        return value
    if isinstance(value, list):
        return [_redact(item, needles) for item in value]
    if isinstance(value, dict):
        return {key: _redact(item, needles) for key, item in value.items()}
    return value


def _endpoint_forms(endpoint: str) -> list[str]:
    """The URL, its origin, and its host:port — longest first, empty ones dropped."""
    forms = [endpoint]
    try:
        parts = urlsplit(endpoint)
    except ValueError:
        parts = None
    if parts is not None and parts.netloc:
        forms += [f"{parts.scheme}://{parts.netloc}", parts.netloc]
    return [form for form in sorted(set(forms), key=len, reverse=True) if form]


def _atlas_name(path: Path) -> str:
    """The Atlas folder's own name: a page shows this, so no local path may reach it."""
    return Path(path).resolve().name or "atlas"


def _provenance(atlas: Atlas, endpoint_name: str | None) -> dict:
    manifest = atlas.manifest
    needles = _endpoint_forms(manifest.endpoint)
    provenance: dict[str, object] = {
        "source": f"atlas: {_atlas_name(atlas.path)}",
        "producer": f"embeddings-cli {__version__}",
        "synthetic": False,
        "seed": manifest.seed,
        "limitations": _redact(list(manifest.limitations), needles),
        "uncertain": _redact(list(manifest.uncertain), needles),
    }
    if endpoint_name is not None:
        provenance["endpoint"] = endpoint_name
    provenance["params"] = _redact(dict(manifest.params), needles)
    return provenance


def _points(atlas: Atlas, *, labels_only: bool) -> list[dict]:
    if labels_only:
        return [{"id": record.id, "label": record.id} for record in atlas.records]
    return [
        {"id": record.id, "label": record.text, "metadata": dict(record.metadata)}
        for record in atlas.records
    ]


def build(args: argparse.Namespace) -> tuple[dict, bytes, bytes, Atlas]:
    """Everything the verb would write: ``(record part, json bytes, block bytes, atlas)``."""
    _, block_path = _targets(args.out)
    endpoint_name = _endpoint_name(args.endpoint_name)
    top_k = args.top_k

    atlas = load_atlas(args)
    matrix = _item_matrix(atlas)
    coords, reduction = _reduce(matrix, rotate=bool(args.rotate), top_k=top_k)
    block = np.ascontiguousarray(coords, dtype="<f4").tobytes()
    count, dimension = int(coords.shape[0]), int(coords.shape[1])

    data = {
        "schema_version": SCHEMA_VERSION,
        "kind": KIND,
        "op": "replace",
        "model": atlas.manifest.model,
        "dimension": dimension,
        "count": count,
        "text_included": not args.labels_only,
        "provenance": _provenance(atlas, endpoint_name),
        "reduction": reduction,
        "vectors": {
            "file": block_path.name,
            "dtype": "float32",
            "byte_order": "little",
            "byte_length": count * dimension * 4,
        },
        "points": _points(atlas, labels_only=bool(args.labels_only)),
    }
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":")) + "\n"
    return data, text.encode("utf-8"), block, atlas


# --------------------------------------------------------------------------
# Writing and reporting
# --------------------------------------------------------------------------


#: The most these files are ever given: owner read/write, group and other read. Never
#: group or other write -- a served data file has no reason to be writable by anyone else.
SERVED_FILE_MODE = 0o644


def _new_file_mode() -> int:
    """:data:`SERVED_FILE_MODE` reduced by this process's umask (0644 under 022, 0600 under 077)."""
    umask = os.umask(0)
    os.umask(umask)
    return SERVED_FILE_MODE & ~umask


def _write(json_path: Path, json_bytes: bytes, block_path: Path, block: bytes) -> None:
    # write_atomic creates its temp file with mkstemp, which is always 0600. These two
    # files are meant to be served by a static web server, possibly running as another
    # user, so each is made readable (at most 0644, never group or other write, and still
    # reduced by the umask) once it is in place.
    # _atomic.py is shared with survey and map and is left as it is.
    mode = _new_file_mode()
    try:
        # Ordering, so a reader can never load a JSON whose block does not belong to it:
        # an existing JSON is removed first (only reachable under --force), then the
        # block lands, then the JSON naming it. A failure at any step leaves either the
        # full new pair or no JSON at all -- never an old JSON beside a new block, which
        # would pass every length check when count and dimension are unchanged.
        json_path.unlink(missing_ok=True)
        write_atomic(block_path, block)
        os.chmod(block_path, mode)
        write_atomic(json_path, json_bytes)
        os.chmod(json_path, mode)
    except OSError as err:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=f"failed to write the viewer data to {json_path}: {err}",
            remediation="check the target directory is writable and has free space",
        ) from err


def _render_text(payload: dict) -> str:
    files = payload["files"]
    reduction = payload["reduction"]
    heading = "# export (applied)" if payload["applied"] else "# export (dry run — nothing written)"
    reduction_line = reduction["kind"]
    if reduction["kind"] == "top-k":
        reduction_line += (
            f" ({reduction['directions_kept']} of {reduction['source_dimension']} directions, "
            f"{reduction['variance_dropped'] * 100:.2f}% of the variance dropped)"
        )
    lines = [
        heading,
        "",
        f"atlas:         {payload['atlas']}",
        f"model:         {payload['model']}",
        f"points:        {payload['points']}",
        f"dimension:     {payload['dimension']} (source {reduction['source_dimension']})",
        f"reduction:     {reduction_line}",
        f"text included: {'yes' if payload['text_included'] else 'no (labels-only)'}",
        f"endpoint name: {payload['endpoint_name'] or '(omitted)'}",
        f"data file:     {files['json']['path']} ({files['json']['bytes']} bytes)",
        f"vector block:  {files['block']['path']} ({files['block']['bytes']} bytes)",
        "",
    ]
    if payload["applied"]:
        lines.append("wrote the embeddings-tumbler data pair (schema version 1).")
    elif payload["exists"] and not payload["would_replace"]:
        lines.append("a target file already exists; re-run with --apply --force to replace it.")
    else:
        lines.append("re-run with --apply to write the files. Nothing was written.")
    lines.append("")
    lines.append(reproduce_line(payload[REPRODUCED_BY_KEY]))
    return "\n".join(lines)


def cmd_export(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    json_path, block_path = _targets(args.out)
    data, json_bytes, block, atlas = build(args)

    exists = json_path.exists() or block_path.exists()
    apply, force = bool(args.apply), bool(args.force)
    if apply and exists and not force:
        taken = [str(p) for p in (json_path, block_path) if p.exists()]
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"refusing to overwrite existing file(s): {', '.join(taken)}",
            remediation="re-run with --force to replace them, or pass a different --out <path>",
        )
    if apply:
        _write(json_path, json_bytes, block_path, block)
        emit_diagnostic(f"export: wrote {block_path} and {json_path}")

    payload = {
        "atlas": str(atlas.path),
        "model": data["model"],
        "points": data["count"],
        "dimension": data["dimension"],
        "reduction": data["reduction"],
        "text_included": data["text_included"],
        "endpoint_name": data["provenance"].get("endpoint"),
        "files": {
            "json": {"path": str(json_path), "bytes": len(json_bytes)},
            "block": {"path": str(block_path), "bytes": len(block)},
        },
        "applied": apply,
        "exists": exists,
        "would_replace": exists and force,
        REPRODUCED_BY_KEY: reproducing_command(),
    }
    emit_result(payload if json_mode else _render_text(payload), json_mode=json_mode)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "export",
        help="Write an atlas and its vectors as an embeddings-tumbler viewer data file.",
        description=(
            "Turn a surveyed atlas (with its vectors.npy) into the embeddings-tumbler data "
            "pair: <name>.json plus a little-endian float32 block <name>.f32. Optionally leave "
            "the corpus text out, rotate exactly onto principal directions, or keep only the "
            "top k of them. Previewed by default, written only with --apply."
        ),
    )
    add_atlas_argument(p)
    p.add_argument(
        "--out",
        default=DEFAULT_OUT,
        metavar="<path>",
        help=f"The data file to write, ending in .json (default: {DEFAULT_OUT}). The vector "
        f"block is written beside it with the same name and {BLOCK_SUFFIX}.",
    )
    p.add_argument(
        "--labels-only",
        action="store_true",
        help="Leave the corpus out: labels are the record ids, no metadata is written, and "
        "the file states text_included: false.",
    )
    reduction = p.add_mutually_exclusive_group()
    reduction.add_argument(
        "--rotate",
        action="store_true",
        help="Rotate the mean-centred vectors exactly onto their principal directions. Keeps "
        "every dimension and every pairwise distance.",
    )
    reduction.add_argument(
        "--top-k",
        type=int,
        default=None,
        metavar="<k>",
        help="Implies the rotation: rotate exactly as --rotate does, then keep only the first "
        "k directions; the file records the share of variance dropped. An approximation; off "
        "unless given.",
    )
    p.add_argument(
        "--endpoint-name",
        default=None,
        metavar="<name>",
        help="A name for the embedding route, written as provenance.endpoint (for example "
        "'lobes gateway'): 1-64 ASCII letters, digits, spaces, '_' or '-', no dots, colons or "
        "slashes. The atlas's endpoint address is never copied. Omitted "
        "when not given.",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Actually write the two files. Without this flag nothing is written.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="With --apply, replace an existing .json or .f32 at the target.",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_export)
