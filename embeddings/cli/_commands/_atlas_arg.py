"""Shared reader-verb atlas entry path.

Six reader verbs (``regions``, ``neighbors``, ``directions``, ``gaps``,
``compare``, ``map``) all need the same opening sequence before they can do
anything domain-specific: resolve the atlas path, fail helpfully if nothing
has been surveyed yet, load the atlas, and let a version mismatch surface
correctly. This module is that shared sequence, built once so the six verbs
depend on it rather than each reinventing (and drifting from) their own copy.

The exit-code distinction is the point
---------------------------------------
An agent driving this CLI in a loop must be able to tell two failure modes
apart without parsing prose:

* **exit 1 (user error)** -- "nothing has been surveyed yet, or the path is
  wrong." Fixed by running ``embeddings survey <corpus> --apply`` or passing
  the right ``--atlas <path>``. :func:`load_atlas` raises this itself for
  both "path does not exist" and "path exists but is not an atlas directory"
  -- distinct *messages*, so a human or log can tell which, but the same exit
  code and the same remediation hint, because the fix is identical either
  way.
* **exit 2 (environment error)** -- "something surveyed is present, but this
  CLI cannot trust it" -- e.g. a ``schema_version`` mismatch. This module
  never raises that itself; it lets :func:`embeddings.atlas.read_atlas`'s own
  fail-closed version check propagate unchanged once a manifest file is
  known to exist. Conflating the two exit codes would make the CLI
  unusable for automation, so this module never wraps or re-codes what
  ``read_atlas`` decides.

Reader verbs never need the gateway
------------------------------------
Everything here is a filesystem read (``Path.exists``, plus whatever
:func:`~embeddings.atlas.read_atlas` does, which is itself pure
JSON/JSONL/``.npy`` I/O -- see ``embeddings/atlas/io.py``). No socket is ever
opened. ``tests/test_atlas_arg.py`` proves this the strong way, with
:func:`tests._fakes.blocked_sockets` wrapped around a real
:func:`load_atlas` call: if this module (or anything it calls) ever tried to
reach a gateway, that call would raise instead of succeeding.

The shared default path
------------------------
:data:`DEFAULT_ATLAS_PATH` is exactly
:data:`embeddings.atlas.io.DEFAULT_ATLAS_DIR` (``./atlas``), re-exported here
under the name the six reader verbs import. ``survey`` (a sibling task)
writes to that same constant by default -- a bare ``embeddings survey
<corpus> --apply`` followed by a bare ``embeddings regions`` (no ``--atlas``
on either side) must resolve to the same directory. Reuse this constant
rather than a duplicate ``"atlas"`` literal on either side of that pair; a
copy that drifts is exactly the failure mode this module exists to prevent.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from embeddings.atlas.io import DEFAULT_ATLAS_DIR, MANIFEST_FILENAME, Atlas, read_atlas
from embeddings.cli._errors import EXIT_USER_ERROR, CliError

__all__ = [
    "DEFAULT_ATLAS_PATH",
    "MISSING_ATLAS_HINT",
    "add_atlas_argument",
    "resolve_atlas_path",
    "load_atlas",
]

#: Default atlas directory reader verbs resolve to when ``--atlas`` is
#: omitted. Identical to :data:`embeddings.atlas.io.DEFAULT_ATLAS_DIR` -- see
#: the module docstring for why this is a re-export, not a fresh literal.
DEFAULT_ATLAS_PATH: Path = DEFAULT_ATLAS_DIR

#: The exact remediation text for "no atlas at the resolved path", shared
#: verbatim by every reader verb so an agent scripting against this CLI sees
#: byte-identical hint text regardless of which of the six verbs it called.
MISSING_ATLAS_HINT = "run embeddings survey <corpus> --apply first, or pass --atlas <path>"


def add_atlas_argument(parser: argparse.ArgumentParser) -> None:
    """Register the shared ``--atlas <path>`` flag on a reader-verb's parser.

    Every reader verb calls this so ``--help`` shows identical wording no
    matter which verb it is attached to. The flag defaults to ``None``
    (unset); :func:`resolve_atlas_path` is what maps that to
    :data:`DEFAULT_ATLAS_PATH` -- callers should always resolve through that
    function rather than reading ``args.atlas`` directly.
    """
    parser.add_argument(
        "--atlas",
        default=None,
        metavar="<path>",
        help=(
            "Path to the atlas directory to read "
            f"(default: '{DEFAULT_ATLAS_PATH}', matching what "
            "`embeddings survey <corpus> --apply` writes)."
        ),
    )


def resolve_atlas_path(args: argparse.Namespace) -> Path:
    """Resolve the atlas path a reader verb should use from parsed ``args``.

    Reads ``args.atlas`` (populated by :func:`add_atlas_argument`); falls
    back to :data:`DEFAULT_ATLAS_PATH` when it is unset or empty. Does not
    touch the filesystem -- this is pure path arithmetic, kept separate from
    :func:`load_atlas` so a verb can report *where* it looked even when
    nothing is there.
    """
    raw = getattr(args, "atlas", None)
    return Path(raw) if raw else Path(DEFAULT_ATLAS_PATH)


def load_atlas(args: argparse.Namespace) -> Atlas:
    """Resolve, validate, and load the atlas a reader verb should operate on.

    The full shared entry sequence: resolve the path (:func:`resolve_atlas_path`),
    fail with an exit-1 :class:`CliError` and :data:`MISSING_ATLAS_HINT` if
    nothing has been surveyed there yet, otherwise hand off to
    :func:`embeddings.atlas.read_atlas` -- whose own fail-closed
    ``schema_version`` check (exit 2) is left to propagate unchanged. This
    function never opens a socket; see the module docstring.

    Two distinct "nothing to read" cases are reported with different
    messages (same code, same hint) so a human or log can tell which
    happened, even though the fix -- survey, or point ``--atlas`` somewhere
    real -- is the same either way:

    * the resolved path does not exist at all;
    * the resolved path exists but is not an atlas directory (no
      ``atlas.json`` inside it).

    Args:
        args: parsed CLI args carrying an ``atlas`` attribute, as populated
            by :func:`add_atlas_argument`.

    Returns:
        The loaded :class:`~embeddings.atlas.Atlas`.

    Raises:
        CliError: exit 1 -- nothing exists at the resolved path, or something
            exists there but is not a valid atlas directory.
        CliError: exit 2 -- propagated verbatim from
            :func:`embeddings.atlas.read_atlas` on a ``schema_version``
            mismatch or a corrupt manifest/records file.
    """
    path = resolve_atlas_path(args)

    if not path.exists():
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"no atlas found at {path} -- nothing has been surveyed yet",
            remediation=MISSING_ATLAS_HINT,
        )

    if not (path / MANIFEST_FILENAME).is_file():
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"{path} exists but is not a valid atlas directory "
            f"(no {MANIFEST_FILENAME} inside it)",
            remediation=MISSING_ATLAS_HINT,
        )

    # A manifest file is confirmed present, so read_atlas's own "no atlas
    # found" branch can never trigger here -- only its version-mismatch (and
    # malformed-JSON) fail-closed checks can still raise, and those propagate
    # unchanged: this helper does not re-wrap or re-code them.
    return read_atlas(path=path)
