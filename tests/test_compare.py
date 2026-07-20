"""Tests for ``embeddings compare`` — relational cross-model agreement.

The acceptance criteria for t18, and where each is proven:

* **k-NN set overlap (Jaccard) and rank correlation on a shared probe set** —
  ``test_self_comparison_agrees_perfectly`` (the sanity check: a model against
  itself must score 1.0 on both) and
  ``test_two_different_models_agree_less_than_a_model_with_itself``.
* **No output value assumes a shared coordinate frame** —
  ``test_no_numeric_output_is_keyed_as_a_coordinate_cosine_or_distance`` walks
  the whole ``--json`` payload and fails on *any* number reachable under a key
  naming a coordinate, a cosine, a distance, or a raw vector, and
  ``test_models_of_different_dimensionality_compare_fine`` proves the method
  never needed a common basis in the first place.
* **The assumptions are stated in the output, and params and seed are
  recorded** — ``test_json_states_all_five_assumptions`` pins all five
  design-note §2 assumptions by id, and the chance-baseline tests below check
  the one assumption that is a *number* rather than a sentence.

Nothing here opens a socket: ``compare`` reads two already-surveyed atlases and
never needs the gateway, which ``test_compare_never_touches_the_gateway``
proves the strong way by running the real CLI entry point inside
:func:`tests._fakes.blocked_sockets`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pytest

from embeddings.atlas.io import write_atlas
from embeddings.atlas.schema import SCHEMA_VERSION, AtlasManifest, AtlasRecord
from embeddings.cli import main
from embeddings.cli._commands import compare as compare_mod
from tests._fakes import blocked_sockets, deterministic_vector

# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

TEXTS: tuple[str, ...] = tuple(f"probe document number {i}" for i in range(24))


def _write_atlas(
    path: Path,
    *,
    model: str,
    texts: Sequence[str] = TEXTS,
    dim: int = 8,
    ids: Sequence[str] | None = None,
) -> Path:
    """Write a small atlas whose geometry is a pure function of ``model`` and ``dim``.

    Salting :func:`deterministic_vector` with the model name (and varying the
    dimensionality) is how this module builds two genuinely *different* fake
    models: same model name and dim -> byte-identical vectors, different model
    name -> unrelated geometry, which is exactly the contrast the agreement
    measures are supposed to detect.
    """
    materialized = list(texts)
    record_ids = list(ids) if ids is not None else [f"doc-{i}" for i in range(len(materialized))]
    vectors = np.asarray(
        [deterministic_vector(f"{model}::{text}", dim) for text in materialized],
        dtype=np.float32,
    )
    manifest = AtlasManifest(
        schema_version=SCHEMA_VERSION,
        model=model,
        endpoint="http://localhost:8002/v1",
        params={"corpus": "fake.jsonl"},
        seed=0,
        limitations=[],
        uncertain=[],
    )
    records = [
        AtlasRecord(id=record_id, text=text, vector_index=index, metadata={})
        for index, (record_id, text) in enumerate(zip(record_ids, materialized))
    ]
    write_atlas(manifest, records, vectors, path=path, overwrite=True)
    return path


def _run(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, str, str]:
    code = main(["compare", *argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def _run_json(capsys: pytest.CaptureFixture[str], *argv: str) -> tuple[int, dict[str, Any], str]:
    code, out, err = _run(capsys, *argv, "--json")
    return code, json.loads(out), err


def _pair(tmp_path: Path, *, model_b: str = "model-b", dim_b: int = 8) -> tuple[Path, Path]:
    a = _write_atlas(tmp_path / "atlas-a", model="model-a", dim=8)
    b = _write_atlas(tmp_path / "atlas-b", model=model_b, dim=dim_b)
    return a, b


def _walk(payload: Any, path: str = "") -> list[tuple[str, Any]]:
    """Flatten a JSON payload into ``(dotted-key-path, value)`` pairs."""
    found: list[tuple[str, Any]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            found += _walk(value, f"{path}.{key}" if path else str(key))
    elif isinstance(payload, list):
        for index, value in enumerate(payload):
            found += _walk(value, f"{path}[{index}]")
    else:
        found.append((path, payload))
    return found


# --------------------------------------------------------------------------
# The sanity check: a model compared with itself
# --------------------------------------------------------------------------


def test_self_comparison_agrees_perfectly(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    atlas = _write_atlas(tmp_path / "atlas", model="model-a")

    code, payload, _ = _run_json(
        capsys, "--atlas-a", str(atlas), "--atlas-b", str(atlas), "-k", "5"
    )

    assert code == 0
    assert payload["measures"]["knn_jaccard"]["mean"] == pytest.approx(1.0)
    assert payload["measures"]["knn_jaccard"]["min"] == pytest.approx(1.0)
    assert payload["measures"]["spearman_rank_correlation"]["mean"] == pytest.approx(1.0)


def test_self_comparison_is_reported_as_a_same_model_sanity_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    atlas = _write_atlas(tmp_path / "atlas", model="model-a")

    _, payload, _ = _run_json(capsys, "--atlas-a", str(atlas), "--atlas-b", str(atlas))

    assert payload["same_model"] is True
    notes = " ".join(payload["notes"]).lower()
    assert "same model" in notes
    assert "sanity check" in notes


def test_same_model_two_atlases_still_flagged_as_same_model(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = _write_atlas(tmp_path / "a", model="model-a")
    b = _write_atlas(tmp_path / "b", model="model-a")

    _, payload, _ = _run_json(capsys, "--atlas-a", str(a), "--atlas-b", str(b), "-k", "5")

    assert payload["same_model"] is True
    assert payload["measures"]["knn_jaccard"]["mean"] == pytest.approx(1.0)


# --------------------------------------------------------------------------
# Two genuinely different models
# --------------------------------------------------------------------------


def test_two_different_models_agree_less_than_a_model_with_itself(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = _pair(tmp_path)

    _, cross, _ = _run_json(capsys, "--atlas-a", str(a), "--atlas-b", str(b), "-k", "4")
    _, same, _ = _run_json(capsys, "--atlas-a", str(a), "--atlas-b", str(a), "-k", "4")

    cross_mean = cross["measures"]["knn_jaccard"]["mean"]
    assert 0.0 <= cross_mean < same["measures"]["knn_jaccard"]["mean"]
    assert cross_mean < 0.6
    assert cross["same_model"] is False


def test_unrelated_models_score_spearman_near_its_stated_chance_value_of_zero(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Regression: the rank measure must be unbiased on its own reported baseline.

    Two hash-derived fake models are independent by construction, so their rank
    correlation must sit near the 0.0 this command reports as chance. An earlier
    implementation correlated ranks over the *union of the two top-k sets*, which
    is structurally anti-correlated -- independent models scored about -0.63 there,
    and a reader comparing that against the stated 0.0 baseline would have read a
    domain artifact as strong disagreement.
    """
    a, b = _pair(tmp_path)

    _, payload, _ = _run_json(capsys, "--atlas-a", str(a), "--atlas-b", str(b), "-k", "4")

    observed = payload["measures"]["spearman_rank_correlation"]["mean"]
    baseline = payload["chance_baseline"]["expected_spearman_rank_correlation"]
    assert baseline == 0.0
    assert abs(observed - baseline) < 0.3, (
        f"independent models scored Spearman {observed}, far from the {baseline} this "
        "command reports as its chance value -- the measure is biased by its domain"
    )


def test_rank_correlation_runs_over_the_full_ordering_not_the_top_k_union(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """k conditions the Jaccard only; the output must say so rather than imply otherwise."""
    a, b = _pair(tmp_path)

    _, payload, _ = _run_json(capsys, "--atlas-a", str(a), "--atlas-b", str(b), "-k", "4")

    domain = payload["params"]["rank_correlation_domain"].lower()
    assert "complete ordering" in domain
    assert "not the top-k union" in domain
    conditioned = next(
        entry for entry in payload["assumptions"] if entry["id"] == "conditioned-on-k"
    )
    assert "jaccard" in conditioned["status"].lower()


def test_spearman_mean_is_independent_of_k(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The full-ordering domain does not move with k -- only the Jaccard does."""
    a, b = _pair(tmp_path)

    _, small, _ = _run_json(capsys, "--atlas-a", str(a), "--atlas-b", str(b), "-k", "2")
    _, large, _ = _run_json(capsys, "--atlas-a", str(a), "--atlas-b", str(b), "-k", "12")

    assert small["measures"]["spearman_rank_correlation"]["mean"] == pytest.approx(
        large["measures"]["spearman_rank_correlation"]["mean"]
    )


def test_models_of_different_dimensionality_compare_fine(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole point of a relational measure: no common basis is required."""
    a, b = _pair(tmp_path, model_b="model-b", dim_b=32)

    code, payload, _ = _run_json(capsys, "--atlas-a", str(a), "--atlas-b", str(b), "-k", "4")

    assert code == 0
    assert payload["coordinate_frame"]["dimensions_a"] == 8
    assert payload["coordinate_frame"]["dimensions_b"] == 32
    assert payload["coordinate_frame"]["shared"] is False


def test_aggregate_agreement_is_symmetric_under_swapping_the_two_atlases(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = _pair(tmp_path)

    _, forward, _ = _run_json(capsys, "--atlas-a", str(a), "--atlas-b", str(b), "-k", "4")
    _, reverse, _ = _run_json(capsys, "--atlas-a", str(b), "--atlas-b", str(a), "-k", "4")

    assert forward["measures"]["knn_jaccard"]["mean"] == pytest.approx(
        reverse["measures"]["knn_jaccard"]["mean"]
    )
    assert forward["measures"]["spearman_rank_correlation"]["mean"] == pytest.approx(
        reverse["measures"]["spearman_rank_correlation"]["mean"]
    )


def test_repeated_runs_over_fixed_atlases_are_byte_identical(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = _pair(tmp_path)

    _, first, _ = _run(capsys, "--atlas-a", str(a), "--atlas-b", str(b), "--json")
    _, second, _ = _run(capsys, "--atlas-a", str(a), "--atlas-b", str(b), "--json")

    assert first == second


# --------------------------------------------------------------------------
# The chance baseline
# --------------------------------------------------------------------------


def test_chance_baseline_is_reported_next_to_the_observed_value(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = _pair(tmp_path)

    _, payload, _ = _run_json(capsys, "--atlas-a", str(a), "--atlas-b", str(b), "-k", "4")

    baseline = payload["chance_baseline"]
    n = payload["probe_set"]["size"]
    pool = n - 1
    expected_intersection = 4 * 4 / pool
    expected_union = 2 * 4 - expected_intersection
    assert baseline["expected_knn_jaccard"] == pytest.approx(
        expected_intersection / expected_union, abs=1e-6
    )
    assert baseline["expected_spearman_rank_correlation"] == pytest.approx(0.0)
    assert baseline["k"] == 4
    assert baseline["probe_items"] == n
    assert baseline["method"]


def test_chance_baseline_scales_with_k_over_n(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = _pair(tmp_path)

    _, small_k, _ = _run_json(capsys, "--atlas-a", str(a), "--atlas-b", str(b), "-k", "2")
    _, large_k, _ = _run_json(capsys, "--atlas-a", str(a), "--atlas-b", str(b), "-k", "12")

    assert (
        large_k["chance_baseline"]["expected_knn_jaccard"]
        > small_k["chance_baseline"]["expected_knn_jaccard"]
    )


def test_observed_agreement_is_reported_relative_to_chance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = _pair(tmp_path)

    _, payload, _ = _run_json(capsys, "--atlas-a", str(a), "--atlas-b", str(b), "-k", "4")

    observed = payload["measures"]["knn_jaccard"]["mean"]
    expected = payload["chance_baseline"]["expected_knn_jaccard"]
    assert payload["chance_baseline"]["observed_above_chance"] == pytest.approx(
        observed - expected, abs=1e-6
    )


def test_a_chance_dominated_k_is_warned_about(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Small N with large k yields high agreement that means nothing (§2)."""
    texts = [f"tiny corpus item {i}" for i in range(6)]
    a = _write_atlas(tmp_path / "a", model="model-a", texts=texts)
    b = _write_atlas(tmp_path / "b", model="model-b", texts=texts)

    _, payload, _ = _run_json(capsys, "--atlas-a", str(a), "--atlas-b", str(b), "-k", "4")

    notes = " ".join(payload["notes"]).lower()
    assert "chance" in notes


def test_chance_baseline_appears_in_text_mode_too(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = _pair(tmp_path)

    _, out, _ = _run(capsys, "--atlas-a", str(a), "--atlas-b", str(b), "-k", "4")

    assert "chance baseline" in out.lower()


# --------------------------------------------------------------------------
# No shared coordinate frame may leak into any output value
# --------------------------------------------------------------------------

#: Substrings that, on a key carrying a *number*, would mean this command had
#: emitted a value only meaningful inside one model's own basis.
_FRAME_BOUND_KEY_FRAGMENTS = (
    "cosine",
    "similarity",
    "distance",
    "coord",
    "vector",
    "embedding",
    "magnitude",
    "norm",
    "centroid",
)


def _leaf(key_path: str) -> str:
    """The final key segment of a dotted ``_walk`` path, lowercased."""
    return key_path.rsplit(".", 1)[-1].split("[", 1)[0].lower()


def test_no_numeric_output_is_keyed_as_a_coordinate_cosine_or_distance(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = _pair(tmp_path, model_b="model-b", dim_b=32)

    _, payload, _ = _run_json(capsys, "--atlas-a", str(a), "--atlas-b", str(b), "-k", "4")

    # The *leaf* key is what names the value; a parent block may legitimately be
    # called `coordinate_frame` precisely because it is the disclosure that there
    # is no shared one. `dimensions_a` is a count of axes, not a position on them.
    offenders = [
        key
        for key, value in _walk(payload)
        if isinstance(value, (int, float))
        and not isinstance(value, bool)
        and any(fragment in _leaf(key) for fragment in _FRAME_BOUND_KEY_FRAGMENTS)
    ]
    assert offenders == []


def test_output_states_that_the_two_models_share_no_coordinate_frame(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = _pair(tmp_path)

    _, payload, _ = _run_json(capsys, "--atlas-a", str(a), "--atlas-b", str(b), "-k", "4")

    statement = payload["coordinate_frame"]["statement"].lower()
    assert "no" in statement
    assert "cosine" in statement  # names the specific trap
    assert payload["coordinate_frame"]["shared"] is False


def test_output_refuses_to_rank_the_two_models(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = _pair(tmp_path)

    _, payload, out = _run_json(capsys, "--atlas-a", str(a), "--atlas-b", str(b), "-k", "4")

    not_claimed = " ".join(payload["not_claimed"]).lower()
    assert "better" in not_claimed
    assert "symmetric" in not_claimed
    assert "ground truth" in not_claimed


def test_text_mode_states_what_compare_does_not_say(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = _pair(tmp_path)

    _, out, _ = _run(capsys, "--atlas-a", str(a), "--atlas-b", str(b), "-k", "4")

    lowered = out.lower()
    assert "does not say" in lowered
    assert "better" in lowered


def test_disagreements_are_located_by_item_id_not_by_position(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = _pair(tmp_path)

    _, payload, _ = _run_json(capsys, "--atlas-a", str(a), "--atlas-b", str(b), "-k", "4")

    entry = payload["per_item"][0]
    assert entry["id"].startswith("doc-")
    assert all(isinstance(item, str) for item in entry["only_a"])
    assert all(isinstance(item, str) for item in entry["only_b"])


# --------------------------------------------------------------------------
# The five stated assumptions, params, and seed
# --------------------------------------------------------------------------


def test_json_states_all_five_assumptions(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = _pair(tmp_path)

    _, payload, _ = _run_json(capsys, "--atlas-a", str(a), "--atlas-b", str(b), "-k", "4")

    ids = [entry["id"] for entry in payload["assumptions"]]
    assert ids == [
        "common-index",
        "representative-probe-set",
        "chance-baseline",
        "conditioned-on-k",
        "model-preprocessing",
    ]
    for entry in payload["assumptions"]:
        assert entry["assumption"]
        assert entry["status"]


def test_assumptions_appear_in_text_mode_too(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = _pair(tmp_path)

    _, out, _ = _run(capsys, "--atlas-a", str(a), "--atlas-b", str(b), "-k", "4")

    lowered = out.lower()
    assert "assumptions" in lowered
    assert "conditioned-on-k" in lowered


def test_params_and_seed_are_recorded(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a, b = _pair(tmp_path)

    _, payload, _ = _run_json(
        capsys, "--atlas-a", str(a), "--atlas-b", str(b), "-k", "6", "--seed", "17"
    )

    assert payload["seed"] == 17
    assert payload["params"]["k"] == 6
    assert payload["params"]["seed"] == 17
    assert payload["params"]["atlas_a"] == str(a)
    assert payload["params"]["atlas_b"] == str(b)
    assert payload["params"]["probe_items"] == payload["probe_set"]["size"]
    assert payload["k"] == 6


def test_k_is_stamped_on_every_output(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a, b = _pair(tmp_path)

    _, payload, _ = _run_json(capsys, "--atlas-a", str(a), "--atlas-b", str(b), "-k", "3")
    _, out, _ = _run(capsys, "--atlas-a", str(a), "--atlas-b", str(b), "-k", "3")

    assert payload["k"] == 3
    assert payload["chance_baseline"]["k"] == 3
    assert "k=3" in out or "k:" in out


def test_json_payload_has_the_documented_shape(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = _pair(tmp_path)

    _, payload, _ = _run_json(capsys, "--atlas-a", str(a), "--atlas-b", str(b), "-k", "4")

    assert set(payload) >= {
        "command",
        "atlas_a",
        "atlas_b",
        "same_model",
        "probe_set",
        "k",
        "params",
        "seed",
        "measures",
        "chance_baseline",
        "per_item",
        "assumptions",
        "not_claimed",
        "coordinate_frame",
        "notes",
    }
    assert payload["command"] == "compare"
    assert payload["atlas_a"]["model"] == "model-a"
    assert payload["atlas_b"]["model"] == "model-b"
    jaccard = payload["measures"]["knn_jaccard"]
    assert set(jaccard) >= {"measure", "mean", "min", "max", "spread", "stdev", "comparisons"}
    assert jaccard["comparisons"] == payload["probe_set"]["size"]


def test_per_item_covers_every_probe_item(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = _pair(tmp_path)

    _, payload, _ = _run_json(capsys, "--atlas-a", str(a), "--atlas-b", str(b), "-k", "4")

    assert len(payload["per_item"]) == payload["probe_set"]["size"] == len(TEXTS)
    entry = payload["per_item"][0]
    assert set(entry) >= {"id", "jaccard", "shared_neighbors", "union_size", "spearman"}


# --------------------------------------------------------------------------
# The shared probe set is a requirement, not a convenience
# --------------------------------------------------------------------------


def test_no_shared_items_exits_1_with_a_probe_set_hint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = _write_atlas(tmp_path / "a", model="model-a", ids=[f"a-{i}" for i in range(len(TEXTS))])
    b = _write_atlas(tmp_path / "b", model="model-b", ids=[f"b-{i}" for i in range(len(TEXTS))])

    code, out, err = _run(capsys, "--atlas-a", str(a), "--atlas-b", str(b))

    assert code == 1
    assert out == ""
    assert "error:" in err
    assert "hint:" in err
    assert "probe set" in err.lower()


def test_ids_present_in_both_but_carrying_different_text_are_excluded(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Assumption 1 is enforced, not merely asserted: same id + different text
    is not the same probe item, so it cannot enter the common index."""
    a = _write_atlas(tmp_path / "a", model="model-a", texts=TEXTS)
    b = _write_atlas(
        tmp_path / "b", model="model-b", texts=("completely different text", *TEXTS[1:])
    )

    _, payload, _ = _run_json(capsys, "--atlas-a", str(a), "--atlas-b", str(b), "-k", "4")

    assert payload["probe_set"]["size"] == len(TEXTS) - 1
    assert payload["probe_set"]["excluded_id_text_mismatch"] == 1
    assert "doc-0" not in [entry["id"] for entry in payload["per_item"]]


def test_a_one_item_probe_set_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a = _write_atlas(tmp_path / "a", model="model-a", texts=("only one",))
    b = _write_atlas(tmp_path / "b", model="model-b", texts=("only one",))

    code, _, err = _run(capsys, "--atlas-a", str(a), "--atlas-b", str(b))

    assert code == 1
    assert "hint:" in err


# --------------------------------------------------------------------------
# Flag validation and missing atlases
# --------------------------------------------------------------------------


def test_k_below_one_exits_1(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    a, b = _pair(tmp_path)

    code, _, err = _run(capsys, "--atlas-a", str(a), "--atlas-b", str(b), "-k", "0")

    assert code == 1
    assert "--top-k" in err
    assert "hint:" in err


def test_k_larger_than_the_probe_set_exits_1_naming_the_maximum(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = _pair(tmp_path)

    code, _, err = _run(capsys, "--atlas-a", str(a), "--atlas-b", str(b), "-k", "999")

    assert code == 1
    assert str(len(TEXTS) - 1) in err


def test_a_missing_atlas_a_exits_1_with_the_shared_hint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    from embeddings.cli._commands._atlas_arg import MISSING_ATLAS_HINT

    b = _write_atlas(tmp_path / "b", model="model-b")

    code, _, err = _run(capsys, "--atlas-a", str(tmp_path / "nope"), "--atlas-b", str(b))

    assert code == 1
    assert MISSING_ATLAS_HINT in err
    assert "--atlas-a" in err


def test_a_missing_atlas_b_names_the_b_flag(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a = _write_atlas(tmp_path / "a", model="model-a")

    code, _, err = _run(capsys, "--atlas-a", str(a), "--atlas-b", str(tmp_path / "nope"))

    assert code == 1
    assert "--atlas-b" in err


def test_errors_render_as_json_when_json_is_requested(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = _pair(tmp_path)

    code, out, err = _run(capsys, "--atlas-a", str(a), "--atlas-b", str(b), "-k", "0", "--json")

    assert code == 1
    assert out == ""
    assert json.loads(err)["code"] == 1


def test_every_flag_has_help_text() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers()
    compare_mod.register(sub)
    help_text = sub.choices["compare"].format_help()

    for flag in ("--atlas-a", "--atlas-b", "--top-k", "--seed", "--json"):
        assert flag in help_text
    # No flag may be registered without help text.
    for action in sub.choices["compare"]._actions:
        assert action.help, f"{action.dest} has no help text"


# --------------------------------------------------------------------------
# compare needs no gateway
# --------------------------------------------------------------------------


def test_compare_never_touches_the_gateway(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    a, b = _pair(tmp_path)

    with blocked_sockets():
        code = main(["compare", "--atlas-a", str(a), "--atlas-b", str(b), "-k", "4", "--json"])

    out = capsys.readouterr().out
    assert code == 0
    assert json.loads(out)["command"] == "compare"


# --------------------------------------------------------------------------
# The measurement primitives, unit level
# --------------------------------------------------------------------------


def test_spearman_of_identical_orderings_is_one() -> None:
    values = np.array([3.0, 1.0, 2.0, 0.0])
    assert compare_mod.spearman(values, values) == pytest.approx(1.0)


def test_spearman_of_reversed_orderings_is_minus_one() -> None:
    values = np.array([0.0, 1.0, 2.0, 3.0])
    assert compare_mod.spearman(values, values[::-1]) == pytest.approx(-1.0)


def test_spearman_needs_at_least_two_points() -> None:
    assert compare_mod.spearman(np.array([1.0]), np.array([1.0])) is None


def test_spearman_is_invariant_under_monotone_rescaling() -> None:
    """The property that makes rank correlation survive two different models."""
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([10.0, 400.0, 9000.0, 90000.0])
    assert compare_mod.spearman(a, b) == pytest.approx(1.0)


def test_jaccard_of_disjoint_sets_is_zero() -> None:
    assert compare_mod.jaccard({1, 2}, {3, 4}) == pytest.approx(0.0)


def test_jaccard_of_identical_sets_is_one() -> None:
    assert compare_mod.jaccard({1, 2}, {1, 2}) == pytest.approx(1.0)


def test_jaccard_is_symmetric() -> None:
    assert compare_mod.jaccard({1, 2, 3}, {3, 4}) == compare_mod.jaccard({3, 4}, {1, 2, 3})


def test_chance_baseline_formula_matches_k_over_n() -> None:
    baseline = compare_mod.chance_baseline(probe_items=101, k=10)
    # pool = 100, E|A n B| = 100/100 = 1.0, E|A u B| = 20 - 1 = 19
    assert baseline["expected_knn_jaccard"] == pytest.approx(1.0 / 19.0, abs=1e-6)


def test_chance_baseline_is_one_when_k_exhausts_the_pool() -> None:
    """With k == N-1 both models must pick every other item; agreement is forced."""
    baseline = compare_mod.chance_baseline(probe_items=5, k=4)
    assert baseline["expected_knn_jaccard"] == pytest.approx(1.0)
