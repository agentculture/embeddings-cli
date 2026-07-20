"""Typed Atlas schema (stable-contract): the shape ``embeddings/atlas/io.py`` reads and writes.

An Atlas is the evidence-backed *interpretation* of a surveyed embedding space
— not the space itself. On disk it is a directory: ``atlas.json`` (the
manifest defined here, as :class:`AtlasManifest`), ``records.jsonl`` (per-item
records, :class:`AtlasRecord` rows), and ``.npy`` vector sidecars (regenerable
bulk, git-ignored). This module defines only the shapes both sides share; a
sibling task builds the reader/writer around them.

Two manifest fields are the reason this module exists rather than being a
convention documented in a docstring: ``limitations`` and ``uncertain``. The
Atlas is a claim about what a survey found, under stated parameters, with a
stated seed. A claim that does not say what it does not know is not
evidence-backed — it is an assertion. ``AtlasManifest`` makes this a schema
constraint, not a request: both fields are required, with no default, and
``__post_init__`` further checks they are actually lists of strings. Omitting
either — or passing something that is not a list of strings — raises
``TypeError`` at construction time, not a missing paragraph discovered later.

``SCHEMA_VERSION`` is the module's other load-bearing constant: the writer
stamps every new manifest's ``schema_version`` with it, and the reader
compares a loaded manifest's ``schema_version`` against it to fail closed on
an atlas written by an unknown or newer CLI version, rather than
best-effort-parsing it (see ``embeddings/atlas/io.py``, a sibling task).
"""

from __future__ import annotations

from dataclasses import dataclass, field

# Bump when AtlasManifest's on-disk shape changes in a way an older reader
# cannot parse. The writer stamps this value into every new manifest; the
# reader compares a loaded manifest's schema_version against it and fails
# closed (exit 2) on a mismatch, rather than best-effort-parsing an atlas it
# does not understand. See embeddings/atlas/io.py (a sibling task).
SCHEMA_VERSION = 1


def _require_list_of_str(name: str, value: object) -> None:
    """Reject anything but a ``list[str]`` for a mandatory prose field.

    Guards against a field being satisfied on paper — some argument was
    passed — but not in substance: ``None``, a bare string mistaken for a
    list of one, or a list holding non-string values would all satisfy "an
    argument was supplied" while defeating the point of the field. This is
    what makes ``limitations``/``uncertain`` a schema constraint on content,
    not merely on presence.
    """
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise TypeError(f"{name} must be a list[str], got {value!r}")


@dataclass(frozen=True)
class AtlasManifest:
    """``atlas.json`` — the Atlas manifest.

    Every field below is required; none has a default. ``limitations`` and
    ``uncertain`` are exactly as mandatory as ``model`` or ``seed`` — this
    schema does not distinguish "structural" fields from "epistemic" ones.

    Attributes:
        schema_version: the schema version this particular manifest was
            written under. Compare against the module-level
            :data:`SCHEMA_VERSION` when reading; a manifest may legitimately
            carry an older (or, for a forward-compatibility check, newer)
            value than the constant currently in force.
        model: identifier of the embedding model surveyed (e.g.
            ``"Qwen/Qwen3-Embedding-0.6B"``).
        endpoint: the OpenAI-compatible endpoint the vectors were embedded
            through (e.g. ``"http://localhost:8002/v1"``).
        params: the measurement parameters in force for this survey — batch
            size, clustering/projection settings, and anything else a reader
            needs in order to reproduce the run.
        seed: the RNG seed the survey ran under. Structural claims derived
            from this Atlas are only as reproducible as this seed makes them.
        limitations: known limitations of this Atlas as an interpretation of
            the space — for example cache-vs-live float variance, or corpus
            sampling bias. Required; pass ``[]`` only as a deliberate,
            reviewable claim that none are known, never as an unfilled
            default — omitting the argument entirely raises ``TypeError``.
        uncertain: interpretations this Atlas is not confident in — regions,
            directions, or boundaries that did not survive a stability check.
            Required for the same reason as ``limitations``.
    """

    schema_version: int
    model: str
    endpoint: str
    params: dict[str, object]
    seed: int
    limitations: list[str]
    uncertain: list[str]

    def __post_init__(self) -> None:
        _require_list_of_str("limitations", self.limitations)
        _require_list_of_str("uncertain", self.uncertain)


@dataclass(frozen=True)
class AtlasRecord:
    """One line of ``records.jsonl`` — a surveyed item and a pointer to its vector.

    Mirrors the corpus contract ``embeddings/corpus.py`` enforces (a sibling
    task): every item has ``text``; ``id`` is a string (a corpus loader may
    default it to the line index, stringified); every other corpus-line key
    survives verbatim into ``metadata``. ``vector_index`` is this item's row
    within whatever ``.npy`` vector sidecar(s) the io layer assembles — the
    sidecar batching/naming scheme itself is an io-layer decision, not fixed
    here.
    """

    id: str
    text: str
    vector_index: int
    metadata: dict[str, object] = field(default_factory=dict)
