"""Tests for the typed Atlas schema (embeddings/atlas/schema.py).

The Atlas is an evidence-backed *interpretation* of a surveyed embedding
space, not the space itself. These tests guard the one property that makes
that framing real rather than aspirational: ``limitations`` and ``uncertain``
are mandatory ``AtlasManifest`` fields enforced by the dataclass signature
itself, not a convention documented in a docstring that a caller can ignore.
"""

from __future__ import annotations

import dataclasses

import pytest

from embeddings.atlas.schema import SCHEMA_VERSION, AtlasManifest, AtlasRecord

# A complete, valid set of manifest kwargs — every test that needs "a good
# manifest, minus one field" starts here and drops a key.
_VALID_KWARGS: dict[str, object] = {
    "schema_version": SCHEMA_VERSION,
    "model": "Qwen/Qwen3-Embedding-0.6B",
    "endpoint": "http://localhost:8002/v1",
    "params": {"batch_size": 32},
    "seed": 0,
    "limitations": ["cache-vs-live float variance is unmeasured on this atlas"],
    "uncertain": ["region 'foo' did not survive a second seed"],
}


def _kwargs_without(*names: str) -> dict[str, object]:
    return {k: v for k, v in _VALID_KWARGS.items() if k not in names}


# --- the manifest constructs from a complete, valid set of fields ---------


def test_manifest_constructs_with_all_required_fields() -> None:
    manifest = AtlasManifest(**_VALID_KWARGS)
    assert manifest.schema_version == SCHEMA_VERSION
    assert manifest.model == "Qwen/Qwen3-Embedding-0.6B"
    assert manifest.endpoint == "http://localhost:8002/v1"
    assert manifest.params == {"batch_size": 32}
    assert manifest.seed == 0
    assert manifest.limitations == ["cache-vs-live float variance is unmeasured on this atlas"]
    assert manifest.uncertain == ["region 'foo' did not survive a second seed"]


def test_manifest_accepts_empty_limitations_and_uncertain_as_a_deliberate_claim() -> None:
    # Passing [] is a legitimate, reviewable claim ("none known"); it is
    # *omitting the argument entirely* that must be impossible, not stating
    # an empty list.
    manifest = AtlasManifest(
        **_kwargs_without("limitations", "uncertain"), limitations=[], uncertain=[]
    )
    assert manifest.limitations == []
    assert manifest.uncertain == []


# --- every required field is actually required -----------------------------


@pytest.mark.parametrize(
    "missing",
    ["schema_version", "model", "endpoint", "params", "seed", "limitations", "uncertain"],
)
def test_omitting_any_required_field_raises(missing: str) -> None:
    kwargs = _kwargs_without(missing)
    with pytest.raises(TypeError):
        AtlasManifest(**kwargs)


def test_omitting_limitations_raises() -> None:
    """Named explicitly per acceptance criteria, in addition to the sweep above."""
    kwargs = _kwargs_without("limitations")
    with pytest.raises(TypeError):
        AtlasManifest(**kwargs)


def test_omitting_uncertain_raises() -> None:
    """Named explicitly per acceptance criteria, in addition to the sweep above."""
    kwargs = _kwargs_without("uncertain")
    with pytest.raises(TypeError):
        AtlasManifest(**kwargs)


def test_omitting_both_limitations_and_uncertain_raises() -> None:
    kwargs = _kwargs_without("limitations", "uncertain")
    with pytest.raises(TypeError):
        AtlasManifest(**kwargs)


# --- limitations/uncertain are non-optional *schema* fields, not convention ---


def test_limitations_and_uncertain_are_schema_fields_without_defaults() -> None:
    """Assert this structurally, via dataclass field introspection.

    A "convention" would be a docstring asking the caller nicely, or a
    default of ``[]`` that quietly satisfies the constructor while the
    author forgets to think about it. Neither shows up here: the fields must
    exist on the dataclass and carry no default value and no default
    factory, so the language itself — not a code-review reminder — is what
    makes omitting them a ``TypeError``.
    """
    fields_by_name = {f.name: f for f in dataclasses.fields(AtlasManifest)}
    for name in ("limitations", "uncertain"):
        assert name in fields_by_name, f"{name} is not a field of AtlasManifest at all"
        field = fields_by_name[name]
        assert field.default is dataclasses.MISSING, f"{name} must not have a default value"
        assert (
            field.default_factory is dataclasses.MISSING
        ), f"{name} must not have a default factory"


def test_every_required_field_has_no_default() -> None:
    """All seven manifest fields are mandatory, not just the epistemic two."""
    for f in dataclasses.fields(AtlasManifest):
        assert f.default is dataclasses.MISSING, f"{f.name} must not have a default value"
        assert f.default_factory is dataclasses.MISSING, f"{f.name} must not have a default factory"


def test_manifest_field_names_match_the_required_set() -> None:
    names = {f.name for f in dataclasses.fields(AtlasManifest)}
    assert names == {
        "schema_version",
        "model",
        "endpoint",
        "params",
        "seed",
        "limitations",
        "uncertain",
    }


# --- limitations/uncertain must actually be lists of strings, not just present ---


@pytest.mark.parametrize("bad_value", [None, "a single string is not a list", 42, {"a": 1}])
def test_limitations_rejects_non_list_values(bad_value: object) -> None:
    kwargs = _kwargs_without("limitations")
    with pytest.raises(TypeError):
        AtlasManifest(**kwargs, limitations=bad_value)


@pytest.mark.parametrize("bad_value", [None, "a single string is not a list", 42, {"a": 1}])
def test_uncertain_rejects_non_list_values(bad_value: object) -> None:
    kwargs = _kwargs_without("uncertain")
    with pytest.raises(TypeError):
        AtlasManifest(**kwargs, uncertain=bad_value)


def test_limitations_rejects_list_with_non_string_items() -> None:
    kwargs = _kwargs_without("limitations")
    with pytest.raises(TypeError):
        AtlasManifest(**kwargs, limitations=[123])


def test_uncertain_rejects_list_with_non_string_items() -> None:
    kwargs = _kwargs_without("uncertain")
    with pytest.raises(TypeError):
        AtlasManifest(**kwargs, uncertain=[123])


# --- SCHEMA_VERSION is the shared stamp/compare constant --------------------


def test_schema_version_is_a_module_level_int_constant() -> None:
    assert isinstance(SCHEMA_VERSION, int)
    assert SCHEMA_VERSION >= 1


def test_schema_version_constant_is_what_a_freshly_written_manifest_stamps() -> None:
    # Mirrors how a writer (a sibling task's io layer) is expected to use the
    # constant: stamp the current SCHEMA_VERSION into every new manifest.
    manifest = AtlasManifest(**_VALID_KWARGS)
    assert manifest.schema_version == SCHEMA_VERSION


def test_manifest_schema_version_field_does_not_force_equality_with_the_constant() -> None:
    # A reader must be able to *load* a manifest written under a different
    # version to compare versions — the field itself must not force equality
    # with the current constant; that comparison is the reader's job (t10).
    newer = AtlasManifest(**{**_VALID_KWARGS, "schema_version": SCHEMA_VERSION + 1})
    assert newer.schema_version == SCHEMA_VERSION + 1
    assert newer.schema_version != SCHEMA_VERSION


# --- AtlasManifest is immutable once constructed ----------------------------


def test_manifest_is_frozen() -> None:
    manifest = AtlasManifest(**_VALID_KWARGS)
    with pytest.raises(dataclasses.FrozenInstanceError):
        manifest.limitations = []  # type: ignore[misc]


# --- AtlasRecord: the records.jsonl per-item shape --------------------------


def test_atlas_record_constructs_with_required_fields() -> None:
    record = AtlasRecord(id="0", text="hello world", vector_index=0)
    assert record.id == "0"
    assert record.text == "hello world"
    assert record.vector_index == 0
    assert record.metadata == {}


def test_atlas_record_preserves_arbitrary_metadata() -> None:
    record = AtlasRecord(id="1", text="hi", vector_index=1, metadata={"source": "doc.md"})
    assert record.metadata == {"source": "doc.md"}


def test_atlas_record_requires_id_text_and_vector_index() -> None:
    for missing in ("id", "text", "vector_index"):
        kwargs = {"id": "0", "text": "hi", "vector_index": 0}
        del kwargs[missing]
        with pytest.raises(TypeError):
            AtlasRecord(**kwargs)
