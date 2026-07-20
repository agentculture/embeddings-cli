"""Tests for the ``embeddings directions`` noun group.

Two properties get disproportionate attention here, because they are the ones
whose failure would make the verb *sound* authoritative while saying nothing:

* **A candidate that does not transfer is reported as a negative result.** A
  direction fitted to a handful of pairs always looks like something; the claim
  only means anything if it holds on pairs it was not fitted on. So the
  generalizing and non-generalizing cases are both asserted, in both text and
  ``--json`` mode, including that the non-generalizing one is still *present* in
  the output rather than filtered out.
* **Stability is measured, never assumed.** ``--runs 1 --resamples 0`` must
  surface the literal ``stability unverified`` label with ``score: null`` — not
  a zero a caller could compare against a threshold.

Nothing here opens a socket. ``discover`` works entirely from Atlas item ids and
the Atlas's own vectors, so :func:`tests._fakes.blocked_sockets` wraps a real
run to prove the reader path never reaches for the embedding gateway.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from embeddings.atlas.io import write_atlas
from embeddings.atlas.schema import SCHEMA_VERSION, AtlasManifest, AtlasRecord
from embeddings.cli import main
from embeddings.cli._commands._atlas_arg import MISSING_ATLAS_HINT
from embeddings.measure import VERDICT_UNVERIFIED
from tests._fakes import blocked_sockets, deterministic_vector

DIM = 8
PAIR_COUNT = 12

# --------------------------------------------------------------------------
# Atlas fixtures
# --------------------------------------------------------------------------


def _planted_direction() -> np.ndarray:
    """A fixed unit vector in the space — the "concept" the planted pairs encode.

    It is a *vector*, not a dimension index: nothing in this suite (or in the
    verb) may name or interpret an individual coordinate.
    """
    raw = np.asarray(deterministic_vector("the planted semantic direction", DIM), dtype=np.float64)
    return raw / np.linalg.norm(raw)


def _noise(text: str, scale: float) -> np.ndarray:
    return scale * np.asarray(deterministic_vector(text, DIM), dtype=np.float64)


def _write_atlas(target: Path, *, planted: bool) -> None:
    """Write an Atlas whose ``a<i>``/``b<i>`` items either do or do not share a direction.

    With ``planted=True`` every ``b<i>`` sits at ``a<i> + 0.6 * u + small noise``,
    so the pair differences share a real direction that must survive a held-out
    split. With ``planted=False`` the ``b<i>`` items are unrelated points, so any
    direction fitted to the training differences is fitting noise and must be
    reported as not generalizing.
    """
    unit = _planted_direction()
    rows: list[np.ndarray] = []
    records: list[AtlasRecord] = []
    for index in range(PAIR_COUNT):
        anchor = np.asarray(deterministic_vector(f"anchor-{index}", DIM), dtype=np.float64)
        if planted:
            other = anchor + 0.6 * unit + _noise(f"jitter-{index}", 0.05)
        else:
            other = np.asarray(deterministic_vector(f"unrelated-{index}", DIM), dtype=np.float64)
        records.append(AtlasRecord(id=f"a{index}", text=f"anchor {index}", vector_index=len(rows)))
        rows.append(anchor)
        records.append(AtlasRecord(id=f"b{index}", text=f"other {index}", vector_index=len(rows)))
        rows.append(other)

    manifest = AtlasManifest(
        schema_version=SCHEMA_VERSION,
        model="fake-embedding-model",
        endpoint="http://localhost:8002/v1",
        params={"batch_size": 32},
        seed=0,
        limitations=[],
        uncertain=[],
    )
    write_atlas(manifest, records, np.asarray(rows, dtype=np.float32), path=target)


def _write_pairs(path: Path, count: int = PAIR_COUNT) -> Path:
    lines = [json.dumps({"a": f"a{index}", "b": f"b{index}"}) for index in range(count)]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


@pytest.fixture
def planted_atlas(tmp_path: Path) -> Path:
    target = tmp_path / "atlas"
    _write_atlas(target, planted=True)
    return target


@pytest.fixture
def unrelated_atlas(tmp_path: Path) -> Path:
    target = tmp_path / "atlas-unrelated"
    _write_atlas(target, planted=False)
    return target


def _discover(atlas: Path, pairs: Path, *extra: str, json_mode: bool = True) -> list[str]:
    argv = ["directions", "discover", "--atlas", str(atlas), "--pairs", str(pairs), *extra]
    if json_mode:
        argv.append("--json")
    return argv


def _run_json(capsys: pytest.CaptureFixture[str], argv: list[str]) -> dict:
    rc = main(argv)
    assert rc == 0, capsys.readouterr().err
    return json.loads(capsys.readouterr().out)


# --------------------------------------------------------------------------
# The noun group's shape: bare noun, overview, and the nested error contract
# --------------------------------------------------------------------------


def test_bare_directions_prints_the_noun_overview_and_exits_0(
    capsys: pytest.CaptureFixture[str],
) -> None:
    rc = main(["directions"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "# embeddings-cli directions" in out
    assert "discover" in out


def test_directions_overview_json_returns_the_sections_payload(
    capsys: pytest.CaptureFixture[str],
) -> None:
    payload = _run_json(capsys, ["directions", "overview", "--json"])
    assert payload["subject"] == "embeddings-cli directions"
    assert isinstance(payload["sections"], list) and payload["sections"]
    for section in payload["sections"]:
        assert section["title"]
        assert isinstance(section["items"], list) and section["items"]


def test_bare_directions_json_matches_overview_json(capsys: pytest.CaptureFixture[str]) -> None:
    bare = _run_json(capsys, ["directions", "--json"])
    explicit = _run_json(capsys, ["directions", "overview", "--json"])
    assert bare == explicit


def test_directions_overview_survives_a_missing_atlas(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Descriptive verbs must not hard-fail on a missing target (rubric)."""
    monkeypatch.chdir(tmp_path)
    assert main(["directions", "overview"]) == 0
    assert main(["directions"]) == 0
    assert main(["directions", "overview", "--atlas", str(tmp_path / "nope")]) == 0
    capsys.readouterr()  # drain the text-mode runs above
    payload = _run_json(
        capsys, ["directions", "overview", "--atlas", str(tmp_path / "nope"), "--json"]
    )
    assert payload["sections"]


def test_directions_overview_ignores_a_stray_positional_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert main(["directions", "overview", "/no/such/path"]) == 0


def test_parse_error_inside_a_sub_verb_keeps_the_structured_error_contract(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """``parser_class=type(p)`` on the noun's subparsers is what makes this pass."""
    with pytest.raises(SystemExit) as exc:
        main(["directions", "overview", "--not-a-flag"])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert captured.err.startswith("error:")
    assert "hint:" in captured.err
    assert captured.out == ""


def test_unknown_sub_verb_keeps_the_structured_error_contract(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["directions", "bogus-verb"])
    assert exc.value.code == 1
    assert "hint:" in capsys.readouterr().err


def test_discover_parse_error_in_json_mode_emits_json_to_stderr(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(["directions", "discover", "--not-a-flag", "--json"])
    assert exc.value.code == 1
    captured = capsys.readouterr()
    assert json.loads(captured.err)["code"] == 1
    assert captured.out == ""


@pytest.mark.parametrize(
    "argv",
    [
        ["directions", "--json"],
        ["directions", "overview", "--json"],
        ["directions", "discover", "--json"],
    ],
)
def test_every_sub_verb_declares_its_own_json_flag(argv: list[str]) -> None:
    """argparse does not inherit the parent's ``--json``; each sub-verb declares it."""
    from embeddings.cli import _build_parser

    args = _build_parser().parse_args(argv)
    assert args.json is True


def test_every_flag_carries_help_text() -> None:
    """The rubric wants a self-describing surface: no undocumented flags."""
    import argparse

    from embeddings.cli import _build_parser

    def subparsers_of(parser: argparse.ArgumentParser) -> dict[str, argparse.ArgumentParser]:
        for action in parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                return dict(action.choices)
        return {}

    noun = subparsers_of(_build_parser())["directions"]
    for name, verb_parser in [("directions", noun), *subparsers_of(noun).items()]:
        for action in verb_parser._actions:
            assert action.help, f"{name}: '{action.dest}' has no help text"


# --------------------------------------------------------------------------
# discover: the missing-atlas path is inherited from _atlas_arg
# --------------------------------------------------------------------------


def test_discover_without_an_atlas_exits_1_with_the_shared_hint(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pairs = _write_pairs(tmp_path / "pairs.jsonl")
    rc = main(["directions", "discover", "--atlas", str(tmp_path / "nope"), "--pairs", str(pairs)])
    assert rc == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert MISSING_ATLAS_HINT in err


# --------------------------------------------------------------------------
# discover: generalization is measured on held-out pairs
# --------------------------------------------------------------------------


def test_discover_reports_a_generalizing_direction(
    planted_atlas: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pairs = _write_pairs(tmp_path / "pairs.jsonl")
    payload = _run_json(capsys, _discover(planted_atlas, pairs, "--top", "1"))

    assert payload["seed"] == 0
    assert payload["params"]
    assert payload["candidates"], "discover must propose at least one candidate direction"
    candidate = payload["candidates"][0]
    assert candidate["generalizes"] is True
    assert candidate["holdout"]["pairs"] >= 1
    assert candidate["holdout"]["mean_cosine"] > 0.5
    assert candidate["holdout"]["agreement_rate"] == 1.0
    assert payload["generalizing"] == 1


def test_discover_reports_a_non_generalizing_candidate_as_a_negative_result(
    unrelated_atlas: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A direction fitted to unrelated pairs must be *reported*, marked, not hidden."""
    pairs = _write_pairs(tmp_path / "pairs.jsonl")
    payload = _run_json(capsys, _discover(unrelated_atlas, pairs, "--top", "1"))

    assert payload["candidates"], "a non-generalizing candidate is still reported"
    candidate = payload["candidates"][0]
    assert candidate["generalizes"] is False
    assert candidate["holdout"]["mean_cosine"] < candidate["train"]["mean_cosine"]
    assert payload["not_generalizing"] == 1
    assert any("generaliz" in note for note in payload["uncertain"])


def test_text_mode_marks_the_generalization_verdict_plainly(
    planted_atlas: Path, unrelated_atlas: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pairs = _write_pairs(tmp_path / "pairs.jsonl")

    assert main(_discover(planted_atlas, pairs, "--top", "1", json_mode=False)) == 0
    good = capsys.readouterr().out
    assert "GENERALIZES" in good
    assert "DOES NOT GENERALIZE" not in good

    assert main(_discover(unrelated_atlas, pairs, "--top", "1", json_mode=False)) == 0
    bad = capsys.readouterr().out
    assert "DOES NOT GENERALIZE" in bad


def test_candidate_vectors_are_opt_in(
    planted_atlas: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pairs = _write_pairs(tmp_path / "pairs.jsonl")
    without = _run_json(capsys, _discover(planted_atlas, pairs, "--top", "1"))
    assert "vector" not in without["candidates"][0]

    with_vectors = _run_json(
        capsys, _discover(planted_atlas, pairs, "--top", "1", "--include-vectors")
    )
    vector = with_vectors["candidates"][0]["vector"]
    assert len(vector) == DIM
    assert np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)


def test_a_direction_is_a_vector_not_a_dimension_index(
    planted_atlas: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The framing rule: no output field names or ranks an individual dimension."""
    pairs = _write_pairs(tmp_path / "pairs.jsonl")
    payload = _run_json(capsys, _discover(planted_atlas, pairs, "--top", "1", "--include-vectors"))
    candidate = payload["candidates"][0]
    assert isinstance(candidate["vector"], list)
    # The candidate records carry no per-dimension vocabulary at all. (The
    # limitations prose does say "dimension" — to disclaim it — so this checks
    # the findings themselves, not the whole document.)
    findings = json.dumps(payload["candidates"])
    for banned in ("dimension", "dim_index", "axis_index", "component_index"):
        assert banned not in findings
    assert any("not a coordinate" in note for note in payload["limitations"])


def test_a_pair_set_that_does_not_encode_a_direction_shows_the_overfit_signature(
    unrelated_atlas: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """High train alignment with near-zero held-out alignment is the case this verb exists for."""
    pairs = _write_pairs(tmp_path / "pairs.jsonl")
    candidate = _run_json(capsys, _discover(unrelated_atlas, pairs, "--top", "1"))["candidates"][0]
    assert candidate["train"]["mean_cosine"] > 0.15
    assert candidate["holdout"]["mean_cosine"] < 0.15
    assert candidate["generalizes"] is False


def test_holdout_pairs_are_disjoint_from_the_pairs_the_direction_was_derived_from(
    planted_atlas: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pairs = _write_pairs(tmp_path / "pairs.jsonl")
    payload = _run_json(capsys, _discover(planted_atlas, pairs, "--top", "1"))
    candidate = payload["candidates"][0]

    train = {(entry["a"], entry["b"]) for entry in candidate["train"]["pairs_used"]}
    holdout = {(entry["a"], entry["b"]) for entry in candidate["holdout"]["pairs_used"]}
    assert train and holdout
    assert not (train & holdout)
    assert len(train) + len(holdout) == PAIR_COUNT
    assert payload["pairs"]["train"] == len(train)
    assert payload["pairs"]["holdout"] == len(holdout)


def test_holdout_fraction_flag_changes_the_split(
    planted_atlas: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pairs = _write_pairs(tmp_path / "pairs.jsonl")
    default_split = _run_json(capsys, _discover(planted_atlas, pairs))["pairs"]
    wider = _run_json(capsys, _discover(planted_atlas, pairs, "--holdout", "0.5"))["pairs"]
    assert wider["holdout"] > default_split["holdout"]


# --------------------------------------------------------------------------
# discover: stability, params, and seed (design note §3)
# --------------------------------------------------------------------------


def test_every_candidate_carries_params_seed_and_a_stability_measure(
    planted_atlas: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pairs = _write_pairs(tmp_path / "pairs.jsonl")
    payload = _run_json(capsys, _discover(planted_atlas, pairs, "--top", "2"))

    assert len(payload["candidates"]) == 2
    for candidate in payload["candidates"]:
        stability = candidate["stability"]
        assert stability["params"]
        assert stability["seed"] == 0
        assert stability["verdict"]
        assert "score" in stability


def test_single_run_with_no_resampling_is_labeled_stability_unverified(
    planted_atlas: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pairs = _write_pairs(tmp_path / "pairs.jsonl")
    payload = _run_json(
        capsys,
        _discover(planted_atlas, pairs, "--top", "1", "--runs", "1", "--resamples", "0"),
    )
    stability = payload["candidates"][0]["stability"]
    assert stability["verdict"] == VERDICT_UNVERIFIED
    assert stability["score"] is None
    assert payload["uncertain"]


def test_stability_unverified_is_visible_in_text_mode_too(
    planted_atlas: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pairs = _write_pairs(tmp_path / "pairs.jsonl")
    argv = _discover(
        planted_atlas, pairs, "--top", "1", "--runs", "1", "--resamples", "0", json_mode=False
    )
    assert main(argv) == 0
    assert VERDICT_UNVERIFIED in capsys.readouterr().out


def test_a_planted_direction_is_stable_across_seed_reruns(
    planted_atlas: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pairs = _write_pairs(tmp_path / "pairs.jsonl")
    payload = _run_json(capsys, _discover(planted_atlas, pairs, "--top", "1", "--runs", "4"))
    stability = payload["candidates"][0]["stability"]
    assert stability["verified"] is True
    assert stability["score"] is not None and stability["score"] > 0.5


# --------------------------------------------------------------------------
# discover: determinism and the no-gateway guarantee
# --------------------------------------------------------------------------


def test_discover_is_byte_identical_for_the_same_seed_over_a_fixed_atlas(
    planted_atlas: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pairs = _write_pairs(tmp_path / "pairs.jsonl")
    argv = _discover(planted_atlas, pairs, "--top", "2")

    assert main(argv) == 0
    first = capsys.readouterr().out
    assert main(argv) == 0
    second = capsys.readouterr().out
    assert first == second


def test_a_different_seed_is_recorded_in_the_payload(
    planted_atlas: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pairs = _write_pairs(tmp_path / "pairs.jsonl")
    payload = _run_json(capsys, _discover(planted_atlas, pairs, "--seed", "7"))
    assert payload["seed"] == 7
    assert payload["candidates"][0]["stability"]["seed"] == 7


def test_discover_never_opens_a_socket(
    planted_atlas: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Reader verbs work from Atlas item ids; no gateway is ever needed."""
    pairs = _write_pairs(tmp_path / "pairs.jsonl")
    with blocked_sockets():
        rc = main(_discover(planted_atlas, pairs, "--top", "1"))
    assert rc == 0
    assert json.loads(capsys.readouterr().out)["candidates"]


def test_unsupervised_mode_samples_pairs_when_no_pairs_file_is_given(
    planted_atlas: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = [
        "directions",
        "discover",
        "--atlas",
        str(planted_atlas),
        "--sample-pairs",
        "10",
        "--json",
    ]
    with blocked_sockets():
        payload = _run_json(capsys, argv)
    assert payload["pairs"]["source"] == "sampled"
    assert payload["pairs"]["total"] == 10
    assert payload["candidates"]


# --------------------------------------------------------------------------
# discover: pairs-file parsing
# --------------------------------------------------------------------------


def test_pairs_may_be_a_two_column_csv_with_comments_and_blank_lines(
    planted_atlas: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pairs = tmp_path / "pairs.csv"
    lines = ["# a comment", ""] + [f"a{index}, b{index}" for index in range(PAIR_COUNT)]
    pairs.write_text("\n".join(lines) + "\n", encoding="utf-8")

    payload = _run_json(capsys, _discover(planted_atlas, pairs, "--top", "1"))
    assert payload["pairs"]["total"] == PAIR_COUNT
    assert payload["candidates"][0]["generalizes"] is True


def test_the_two_pair_spellings_agree(
    planted_atlas: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    jsonl = _write_pairs(tmp_path / "pairs.jsonl")
    csv = tmp_path / "pairs.csv"
    csv.write_text(
        "\n".join(f"a{index},b{index}" for index in range(PAIR_COUNT)) + "\n", encoding="utf-8"
    )

    from_jsonl = _run_json(capsys, _discover(planted_atlas, jsonl, "--top", "1"))["candidates"][0]
    from_csv = _run_json(capsys, _discover(planted_atlas, csv, "--top", "1"))["candidates"][0]
    # The pair *source* differs (it is recorded as provenance); the measurement
    # derived from the identical pair set must not.
    assert from_jsonl["train"] == from_csv["train"]
    assert from_jsonl["holdout"] == from_csv["holdout"]
    assert from_jsonl["generalizes"] == from_csv["generalizes"]


def test_a_malformed_csv_row_exits_1(
    planted_atlas: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pairs = tmp_path / "pairs.csv"
    pairs.write_text("a0,b0\na1,b1,a2\n", encoding="utf-8")
    rc = main(["directions", "discover", "--atlas", str(planted_atlas), "--pairs", str(pairs)])
    assert rc == 1
    assert "hint:" in capsys.readouterr().err


def test_a_json_pair_line_without_a_and_b_keys_exits_1(
    planted_atlas: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text(json.dumps({"left": "a0", "right": "b0"}) + "\n", encoding="utf-8")
    rc = main(["directions", "discover", "--atlas", str(planted_atlas), "--pairs", str(pairs)])
    assert rc == 1
    assert "hint:" in capsys.readouterr().err


# --------------------------------------------------------------------------
# discover: user-error paths
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flag,value",
    [
        ("--holdout", "1.5"),
        ("--holdout", "0"),
        ("--top", "0"),
        ("--runs", "0"),
        ("--resamples", "-1"),
        ("--sample-pairs", "1"),
    ],
)
def test_out_of_range_flag_values_exit_1_with_a_hint(
    planted_atlas: Path,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    flag: str,
    value: str,
) -> None:
    pairs = _write_pairs(tmp_path / "pairs.jsonl")
    rc = main(_discover(planted_atlas, pairs, flag, value, json_mode=False))
    assert rc == 1
    err = capsys.readouterr().err
    assert err.startswith("error:")
    assert flag in err
    assert "hint:" in err


def test_sampling_from_a_single_item_atlas_exits_1(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "tiny"
    manifest = AtlasManifest(
        schema_version=SCHEMA_VERSION,
        model="fake-embedding-model",
        endpoint="http://localhost:8002/v1",
        params={},
        seed=0,
        limitations=[],
        uncertain=[],
    )
    records = [AtlasRecord(id="only", text="only item", vector_index=0)]
    write_atlas(manifest, records, np.zeros((1, DIM), dtype=np.float32), path=target)

    rc = main(["directions", "discover", "--atlas", str(target)])
    assert rc == 1
    assert "hint:" in capsys.readouterr().err


def test_a_pair_naming_an_unknown_item_id_exits_1(
    planted_atlas: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pairs = tmp_path / "pairs.jsonl"
    lines = [
        json.dumps({"a": "a0", "b": "b0"}),
        json.dumps({"a": "a1", "b": "not-in-the-atlas"}),
    ]
    pairs.write_text("\n".join(lines) + "\n", encoding="utf-8")
    rc = main(["directions", "discover", "--atlas", str(planted_atlas), "--pairs", str(pairs)])
    assert rc == 1
    err = capsys.readouterr().err
    assert "not-in-the-atlas" in err
    assert "hint:" in err


def test_too_few_pairs_to_split_exits_1(
    planted_atlas: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pairs = _write_pairs(tmp_path / "pairs.jsonl", count=1)
    rc = main(["directions", "discover", "--atlas", str(planted_atlas), "--pairs", str(pairs)])
    assert rc == 1
    assert "hint:" in capsys.readouterr().err


def test_a_missing_pairs_file_exits_1(
    planted_atlas: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    rc = main(
        [
            "directions",
            "discover",
            "--atlas",
            str(planted_atlas),
            "--pairs",
            str(tmp_path / "nope.jsonl"),
        ]
    )
    assert rc == 1
    assert "hint:" in capsys.readouterr().err


def test_a_malformed_pairs_line_exits_1(
    planted_atlas: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pairs = tmp_path / "pairs.jsonl"
    pairs.write_text("{not json at all\n", encoding="utf-8")
    rc = main(["directions", "discover", "--atlas", str(planted_atlas), "--pairs", str(pairs)])
    assert rc == 1
    assert "hint:" in capsys.readouterr().err


def test_results_and_diagnostics_are_never_mixed(
    planted_atlas: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pairs = _write_pairs(tmp_path / "pairs.jsonl")
    assert main(_discover(planted_atlas, pairs)) == 0
    captured = capsys.readouterr()
    json.loads(captured.out)  # stdout is pure JSON, nothing else
    assert "error:" not in captured.err
