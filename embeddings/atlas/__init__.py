"""embeddings.atlas — the persisted, typed interpretation of a surveyed embedding space.

The Atlas is not the embedding space itself; it is what a survey concluded
about it, under stated parameters and a stated seed, with its known
limitations and uncertain interpretations recorded as first-class content.
:mod:`embeddings.atlas.schema` defines the shapes (:class:`AtlasManifest`,
:class:`AtlasRecord`) and the :data:`SCHEMA_VERSION` stamp/compare constant; a
sibling module builds the read/write layer around them.
"""

from __future__ import annotations

from embeddings.atlas.schema import SCHEMA_VERSION, AtlasManifest, AtlasRecord

__all__ = ["SCHEMA_VERSION", "AtlasManifest", "AtlasRecord"]
