"""Tests for ``embeddings gaps`` — the dual-labeled, probe-based gap verb.

The error this verb exists to not make
--------------------------------------
A sparse area of a surveyed corpus is a gap in **the corpus**. It is not a gap
in what the model can represent — the model may represent that area perfectly
well and simply never have been sent anything from it. "The model has a blind
spot here" is the claim a reader acts on, and it travels in the direction of a
more interesting result, so it is the claim this suite guards hardest.

The load-bearing assertions here, in order of how much damage their absence
would do:

* ``test_no_probe_step_emits_zero_model_representation_findings`` and its
  ``--no-probe`` twin — the acceptance criterion. With probing off, the
  ``model_representation_findings`` list is present and **empty**. A model claim
  can only ever come from inputs embedded at probe time.
* ``test_the_two_kinds_are_separate_top_level_keys`` — the structural guard.
  The kinds are separate JSON keys, never one list with a ``kind`` field, so an
  aggregating agent cannot flatten them and lose the distinction. This test also
  asserts no finding carries a ``kind``-like discriminator, because adding one
  would re-open exactly the merge this design forecloses.
* ``test_gateway_down_still_emits_corpus_findings_and_marks_probe_unavailable``
  — the common case on a machine with no gateway. The corpus half is still
  valid, so the verb must not fail; but the probe step is marked unavailable
  **and** zero model findings are emitted. Silently omitting the distinction
  would be the same error by a quieter route.
* ``test_disclaimer_states_corpus_sparsity_says_nothing_about_the_model`` — the
  prose. A reader who parses neither key names nor exit codes still gets told.

No test here opens a socket. Every probing test wires a
:class:`FakeEmbeddingTransport` (or :class:`UnreachableTransport`) through the
``build_client`` seam, matching ``survey`` and ``neighbors``; the corpus-only
tests run inside ``blocked_sockets()`` to prove the default path never reaches
a gateway at all.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from embeddings.atlas.io import write_atlas
from embeddings.atlas.schema import SCHEMA_VERSION, AtlasManifest, AtlasRecord
from embeddings.backends import EmbeddingClient
from embeddings.cli import main
from embeddings.cli._commands import gaps as gaps_mod
from tests._fakes import (
    FakeEmbeddingTransport,
    UnreachableTransport,
    blocked_sockets,
    wire_transport,
)

# --------------------------------------------------------------------------
# Fixture atlas: twelve hand-placed 8-dim vectors in three groups whose
# isolation is worked out by hand, so "which region is sparse" is a fact about
# the fixture rather than about whatever k-means happened to do.
#
#   A (5 items, tight)  around [1, 0, 0]  -- nearest-neighbour distance ~0.005
#   B (5 items, tight)  around [0, 1, 0]  -- nearest-neighbour distance ~0.005
#   C (2 items, spread) around [0, 0, 1]  -- nearest-neighbour distance ~0.08
#
# C is the sparse region: fewest items and by far the largest nearest-neighbour
# distance. A and B sit ~1.0 cosine distance from each other and from C, so the
# k=3 partition is unambiguous and the midpoints between centroids are genuinely
# empty (the inter-region voids this verb reports).
# --------------------------------------------------------------------------

_DIM = 8


def _vec(x: float, y: float, z: float) -> list[float]:
    return [x, y, z, 0.0, 0.0, 0.0, 0.0, 0.0]


_GROUP_A = [
    _vec(1.0, 0.0, 0.0),
    _vec(0.99, 0.1, 0.0),
    _vec(0.99, -0.1, 0.0),
    _vec(0.98, 0.05, 0.05),
    _vec(0.98, -0.05, -0.05),
]
_GROUP_B = [
    _vec(0.0, 1.0, 0.0),
    _vec(0.1, 0.99, 0.0),
    _vec(-0.1, 0.99, 0.0),
    _vec(0.05, 0.98, 0.05),
    _vec(-0.05, 0.98, -0.05),
]
# Only two items, and they sit ~0.08 cosine distance apart -- an order of
# magnitude more isolated than anything in A or B.
_GROUP_C = [
    _vec(0.2, 0.0, 0.98),
    _vec(-0.2, 0.0, 0.98),
]

_ALL_VECTORS = _GROUP_A + _GROUP_B + _GROUP_C
_IDS = [f"a{i}" for i in range(5)] + [f"b{i}" for i in range(5)] + [f"c{i}" for i in range(2)]

# Probe vectors pinned near C's direction and tightly grouped: they land in the
# sparse region and their spread is far below the corpus-wide baseline.
_COHERENT_PROBES = {
    "probe one": _vec(0.05, 0.0, 0.999),
    "probe two": _vec(-0.05, 0.0, 0.999),
    "probe three": _vec(0.0, 0.05, 0.999),
}
# Probe vectors that still land nearest C's centroid but are spread wide apart
# from each other -- the "scattered" shape.
_SCATTERED_PROBES = {
    "probe one": _vec(0.6, 0.0, 0.8),
    "probe two": _vec(-0.6, 0.0, 0.8),
    "probe three": _vec(0.0, 0.6, 0.8),
}


def _manifest(**overrides: object) -> AtlasManifest:
    kwargs: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "model": "fake-embedding-model",
        "endpoint": "http://fake.invalid/v1",
        "params": {},
        "seed": 0,
        "limitations": [],
        "uncertain": [],
    }
    kwargs.update(overrides)
    return AtlasManifest(**kwargs)  # type: ignore[arg-type]


def _write_fixture_atlas(target: Path) -> None:
    records = [
        AtlasRecord(id=item_id, text=f"text for {item_id}", vector_index=index, metadata={})
        for index, item_id in enumerate(_IDS)
    ]
    vectors = np.array(_ALL_VECTORS, dtype=np.float32)
    write_atlas(_manifest(), records, vectors, path=target)


def _wire_fake(monkeypatch: pytest.MonkeyPatch, fake: Any) -> None:
    """Replace gaps' client factory with one backed by ``fake``. Opens no socket."""

    def _build(base_url: str | None, model_name: str | None) -> EmbeddingClient:
        return EmbeddingClient(
            base_url=base_url or "http://fake.invalid/v1",
            model=model_name or "fake-embedding-model",
            transport=wire_transport(fake),
        )

    monkeypatch.setattr(gaps_mod, "build_client", _build)


def _pinned_transport(overrides: dict[str, list[float]]) -> FakeEmbeddingTransport:
    transport = FakeEmbeddingTransport(dim=_DIM)
    for text, vector in overrides.items():
        transport.set_override(text, vector)
    return transport


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("EMBEDDINGS_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.delenv("EMBEDDINGS_EMBED_URL", raising=False)
    monkeypatch.delenv("EMBEDDINGS_EMBED_MODEL", raising=False)
    monkeypatch.chdir(tmp_path)
    _write_fixture_atlas(tmp_path / "atlas")
    return tmp_path


def _probes_file(target: Path, texts: list[str]) -> Path:
    path = target / "probes.txt"
    body = ["# a comment line, skipped", ""] + texts
    path.write_text("\n".join(body) + "\n", encoding="utf-8")
    return path


def _run(argv: list[str]) -> int:
    return main(argv)


def _run_json(capsys: pytest.CaptureFixture[str], argv: list[str]) -> dict:
    code = _run([*argv, "--json"])
    assert code == 0, f"expected exit 0, got {code}: {capsys.readouterr().err}"
    return json.loads(capsys.readouterr().out)


# --------------------------------------------------------------------------
# The acceptance criterion: no probe step, no model-representation findings
# --------------------------------------------------------------------------


def test_no_probe_step_emits_zero_model_representation_findings(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The headline rule. Probing is off by default, so a bare run makes no model claim."""
    with blocked_sockets():
        payload = _run_json(capsys, ["gaps", "-k", "3"])

    assert payload["model_representation_findings"] == []
    # The corpus half must actually have found something, or the assertion above
    # would pass vacuously on an empty analysis.
    assert payload["corpus_coverage_findings"], "fixture should yield corpus-coverage findings"
    assert payload["probe_step"]["requested"] is False
    assert payload["probe_step"]["attempted"] is False


def test_explicit_no_probe_emits_zero_model_representation_findings(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with blocked_sockets():
        payload = _run_json(capsys, ["gaps", "-k", "3", "--no-probe"])

    assert payload["model_representation_findings"] == []
    assert payload["probe_step"]["requested"] is False


def test_corpus_only_run_never_opens_a_socket(sandbox: Path) -> None:
    """Without --probe this is an ordinary reader verb: atlas in, findings out."""
    with blocked_sockets():
        assert _run(["gaps", "-k", "3"]) == 0


# --------------------------------------------------------------------------
# The structural guard: separate keys, never one list
# --------------------------------------------------------------------------


def test_the_two_kinds_are_separate_top_level_keys(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Separate keys are what stop a downstream aggregator flattening the distinction."""
    payload = _run_json(capsys, ["gaps", "-k", "3"])

    assert isinstance(payload["corpus_coverage_findings"], list)
    assert isinstance(payload["model_representation_findings"], list)
    # No merged list may exist under any of the names an aggregator would reach
    # for first.
    for merged in ("findings", "all_findings", "gaps"):
        assert merged not in payload, f"{merged!r} would invite the two kinds to be merged"


def test_no_finding_carries_a_kind_discriminator(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A ``kind`` field would make flattening a one-liner. The key *is* the kind."""
    payload = _run_json(capsys, ["gaps", "-k", "3"])

    for finding in payload["corpus_coverage_findings"]:
        assert "kind" not in finding
        # Each finding still says in prose which half it belongs to.
        assert finding["evidence_class"] == gaps_mod.EVIDENCE_CORPUS


def test_counts_are_reported_separately_and_never_summed(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = _run_json(capsys, ["gaps", "-k", "3"])
    counts = payload["counts"]

    assert counts["corpus_coverage"] == len(payload["corpus_coverage_findings"])
    assert counts["model_representation"] == len(payload["model_representation_findings"])
    assert "total" not in counts, "a combined total is the merge this design forbids"


# --------------------------------------------------------------------------
# The prose disclaimer
# --------------------------------------------------------------------------


def test_disclaimer_states_corpus_sparsity_says_nothing_about_the_model(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = _run_json(capsys, ["gaps", "-k", "3"])
    disclaimer = payload["disclaimer"].lower()

    assert "corpus sparsity alone says nothing about model capability" in disclaimer
    assert "probe" in disclaimer


def test_text_mode_renders_the_disclaimer_and_both_headings(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(["gaps", "-k", "3"]) == 0
    out = capsys.readouterr().out

    assert "corpus sparsity alone says nothing about model capability" in out.lower()
    assert "corpus coverage" in out.lower()
    assert "model representation" in out.lower()
    # The model half must say it is empty rather than simply be absent -- an
    # absent section reads as "nothing to worry about" instead of "not measured".
    assert "no probe step ran" in out.lower()


# --------------------------------------------------------------------------
# The probe path
# --------------------------------------------------------------------------


def test_probing_with_coherent_probes_reports_a_corpus_artifact(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Probes that land in the sparse region and cohere are evidence *against* a model gap."""
    _wire_fake(monkeypatch, _pinned_transport(_COHERENT_PROBES))
    probes = _probes_file(sandbox, list(_COHERENT_PROBES))

    payload = _run_json(capsys, ["gaps", "-k", "3", "--probe", "--probes", str(probes)])

    assert payload["probe_step"]["available"] is True
    assert payload["probe_step"]["probe_count"] == 3
    findings = payload["model_representation_findings"]
    assert findings, "probes landed in the sparse region; a finding is owed"
    finding = findings[0]
    assert finding["verdict"] == gaps_mod.PROBE_VERDICT_COHERENT
    assert finding["evidence_class"] == gaps_mod.EVIDENCE_MODEL
    # The finding must point *away* from a model gap, not toward one.
    assert "corpus" in finding["shows"].lower()


def test_probing_with_scattered_probes_reports_representation_evidence(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Probes that land in the region but scatter are evidence about the representation.

    ``--coherence-factor 0.2`` is passed explicitly rather than relying on the
    default: these probes spread ~0.48 against a corpus baseline of ~0.65, which
    sits close enough to the default 0.75 threshold that the test would be
    asserting the threshold rather than the mechanism. Tightening it makes the
    classification unambiguous and demonstrates that the threshold is a stated
    parameter, not a discovered truth.
    """
    _wire_fake(monkeypatch, _pinned_transport(_SCATTERED_PROBES))
    probes = _probes_file(sandbox, list(_SCATTERED_PROBES))

    payload = _run_json(
        capsys,
        [
            "gaps",
            "-k",
            "3",
            "--probe",
            "--probes",
            str(probes),
            "--coherence-factor",
            "0.2",
        ],
    )

    findings = payload["model_representation_findings"]
    assert findings
    finding = findings[0]
    assert finding["verdict"] == gaps_mod.PROBE_VERDICT_SCATTERED
    # Evidence, never proof -- and the output must say so itself.
    assert "evidence" in finding["shows"].lower()
    assert "not proof" in finding["does_not_show"].lower()


def test_every_probe_records_its_provenance(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A model claim is only as good as its probes, so the probes must be auditable."""
    _wire_fake(monkeypatch, _pinned_transport(_COHERENT_PROBES))
    probes = _probes_file(sandbox, list(_COHERENT_PROBES))

    payload = _run_json(capsys, ["gaps", "-k", "3", "--probe", "--probes", str(probes)])

    finding = payload["model_representation_findings"][0]
    assert finding["probes"], "the finding must carry the probes behind it"
    for probe in finding["probes"]:
        assert probe["text"] in _COHERENT_PROBES
        provenance = probe["provenance"]
        assert provenance["source"] == gaps_mod.PROVENANCE_FILE
        assert provenance["path"] == str(probes)
        assert isinstance(provenance["line"], int)


def test_a_model_finding_carries_its_distinguishing_measurement(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """ "We probed" is not a measurement. The numbers that discriminate must be present."""
    _wire_fake(monkeypatch, _pinned_transport(_COHERENT_PROBES))
    probes = _probes_file(sandbox, list(_COHERENT_PROBES))

    payload = _run_json(capsys, ["gaps", "-k", "3", "--probe", "--probes", str(probes)])
    measurement = payload["model_representation_findings"][0]["measurement"]

    for key in (
        "probes_landed",
        "probe_spread",
        "region_spread",
        "corpus_baseline_spread",
        "coherence_threshold",
        "mean_distance_to_nearest_corpus_item",
    ):
        assert key in measurement, f"missing distinguishing measurement {key!r}"
    assert measurement["probe_spread"] < measurement["coherence_threshold"]


def test_probes_landing_elsewhere_are_inconclusive_not_a_model_finding(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """A probe that misses the sparse region says nothing about the region."""
    elsewhere = {text: _vec(1.0, 0.0, 0.0) for text in _COHERENT_PROBES}
    _wire_fake(monkeypatch, _pinned_transport(elsewhere))
    probes = _probes_file(sandbox, list(elsewhere))

    payload = _run_json(capsys, ["gaps", "-k", "3", "--probe", "--probes", str(probes)])

    assert payload["model_representation_findings"] == []
    assert payload["inconclusive_probes"], "the miss must be visible, not silently dropped"
    assert payload["inconclusive_probes"][0]["verdict"] == gaps_mod.PROBE_VERDICT_ELSEWHERE


def test_holdout_probes_are_re_embedded_and_excluded_from_the_corpus_half(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Held-out items must leave the reference set, or a probe is its own nearest neighbour."""
    _wire_fake(monkeypatch, FakeEmbeddingTransport(dim=_DIM))

    payload = _run_json(capsys, ["gaps", "-k", "3", "--probe", "--holdout", "2", "--seed", "0"])

    assert payload["probe_step"]["available"] is True
    assert payload["probe_step"]["probe_count"] == 2
    held_out = payload["params"]["holdout_ids"]
    assert len(held_out) == 2
    assert payload["items_analyzed"] == len(_IDS) - 2
    # Every held-out id is gone from the corpus half's membership lists.
    for finding in payload["corpus_coverage_findings"]:
        assert not set(finding.get("members", [])) & set(held_out)


# --------------------------------------------------------------------------
# Gateway down: degrade, do not fail, do not silently drop the distinction
# --------------------------------------------------------------------------


def test_gateway_down_still_emits_corpus_findings_and_marks_probe_unavailable(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _wire_fake(monkeypatch, UnreachableTransport())
    probes = _probes_file(sandbox, list(_COHERENT_PROBES))

    code = _run(["gaps", "-k", "3", "--probe", "--probes", str(probes), "--json"])
    captured = capsys.readouterr()

    assert code == 0, "the corpus half is still valid; the whole verb must not fail"
    payload = json.loads(captured.out)
    assert payload["corpus_coverage_findings"]
    assert payload["model_representation_findings"] == []
    probe_step = payload["probe_step"]
    assert probe_step["requested"] is True
    assert probe_step["attempted"] is True
    assert probe_step["available"] is False
    assert probe_step["reason"], "an unavailable probe step must say why"
    # The degradation is announced, not silent.
    assert "probe" in captured.err.lower()


def test_gateway_down_keeps_results_on_stdout_and_diagnostics_on_stderr(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _wire_fake(monkeypatch, UnreachableTransport())
    probes = _probes_file(sandbox, list(_COHERENT_PROBES))

    _run(["gaps", "-k", "3", "--probe", "--probes", str(probes), "--json"])
    captured = capsys.readouterr()

    json.loads(captured.out)  # stdout is pure JSON, undisturbed by the diagnostic
    assert captured.err.strip(), "the unavailable probe step is reported on stderr"


def test_gateway_down_text_mode_says_the_probe_step_was_unavailable(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _wire_fake(monkeypatch, UnreachableTransport())
    probes = _probes_file(sandbox, list(_COHERENT_PROBES))

    assert _run(["gaps", "-k", "3", "--probe", "--probes", str(probes)]) == 0
    assert "probe step unavailable" in capsys.readouterr().out.lower()


def test_probes_embedded_by_a_wrong_dimension_model_degrade_like_a_down_gateway(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Probes from a different model are incomparable, so they must yield no finding.

    Comparing a 4-dim probe against an 8-dim atlas is not a hard error to be
    fixed mid-run -- it is a probe step that cannot produce evidence. It takes
    the same route as an unreachable gateway: corpus half intact, probe step
    unavailable with a reason, zero model findings, exit 0.
    """
    _wire_fake(monkeypatch, FakeEmbeddingTransport(dim=4))
    probes = _probes_file(sandbox, list(_COHERENT_PROBES))

    code = _run(["gaps", "-k", "3", "--probe", "--probes", str(probes), "--json"])
    captured = capsys.readouterr()

    assert code == 0
    payload = json.loads(captured.out)
    assert payload["model_representation_findings"] == []
    assert payload["corpus_coverage_findings"]
    assert payload["probe_step"]["available"] is False
    assert "dimension" in payload["probe_step"]["reason"]
    assert "--embed-model" in captured.err


def test_a_single_landed_probe_is_too_few_to_measure_spread(
    sandbox: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """One point has no pairwise spread, so it cannot support a verdict either way."""
    mixed = {
        "probe in the sparse area": _vec(0.05, 0.0, 0.999),
        "probe somewhere else": _vec(1.0, 0.0, 0.0),
    }
    _wire_fake(monkeypatch, _pinned_transport(mixed))
    probes = _probes_file(sandbox, list(mixed))

    payload = _run_json(capsys, ["gaps", "-k", "3", "--probe", "--probes", str(probes)])

    assert payload["model_representation_findings"] == []
    assert payload["inconclusive_probes"][0]["verdict"] == gaps_mod.PROBE_VERDICT_TOO_FEW


# --------------------------------------------------------------------------
# Flag contract
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("flag", "value"),
    [
        ("--clusters", "1"),
        ("--runs", "0"),
        ("--resamples", "-1"),
        ("--sparsity-quantile", "0"),
        ("--sparsity-quantile", "1"),
        ("--void-factor", "0"),
        ("--coherence-factor", "-0.5"),
    ],
)
def test_out_of_range_parameters_are_user_errors_with_hints(
    sandbox: Path, capsys: pytest.CaptureFixture[str], flag: str, value: str
) -> None:
    """Every threshold is a parameter, so every one of them validates its range."""
    with blocked_sockets():
        code = _run(["gaps", flag, value])

    assert code == 1
    err = capsys.readouterr().err
    assert err.startswith("error: ")
    assert flag in err
    assert "hint: " in err


def test_probe_source_without_the_probe_flag_is_a_user_error(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Network egress is never implicit -- --probes alone does not enable probing."""
    probes = _probes_file(sandbox, list(_COHERENT_PROBES))

    with blocked_sockets():
        code = _run(["gaps", "-k", "3", "--probes", str(probes)])

    assert code == 1
    err = capsys.readouterr().err
    assert "--probe" in err
    assert err.startswith("error: ")
    assert "hint: " in err


def test_probe_flag_without_a_source_is_a_user_error(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with blocked_sockets():
        code = _run(["gaps", "-k", "3", "--probe"])

    assert code == 1
    err = capsys.readouterr().err
    assert "--probes" in err and "--holdout" in err


def test_missing_probes_file_is_a_user_error(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with blocked_sockets():
        code = _run(["gaps", "-k", "3", "--probe", "--probes", str(sandbox / "nope.txt")])

    assert code == 1
    assert "hint: " in capsys.readouterr().err


def test_empty_probes_file_is_a_user_error(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = sandbox / "probes.txt"
    path.write_text("# only a comment\n\n", encoding="utf-8")

    with blocked_sockets():
        code = _run(["gaps", "-k", "3", "--probe", "--probes", str(path)])

    assert code == 1
    assert "no probe texts" in capsys.readouterr().err.lower()


def test_missing_atlas_exits_one_with_the_shared_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert _run(["gaps"]) == 1
    assert "survey" in capsys.readouterr().err


# --------------------------------------------------------------------------
# Parameters, seed, stability, determinism (design note SS3 and SS4)
# --------------------------------------------------------------------------


def test_params_and_seed_are_stamped_on_the_payload(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = _run_json(capsys, ["gaps", "-k", "3", "--seed", "7"])

    assert payload["seed"] == 7
    params = payload["params"]
    for key in ("clusters", "sparsity_quantile", "void_factor", "coherence_factor", "runs"):
        assert key in params


def test_sparsity_findings_carry_a_stability_measure(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A sparse region that only exists at one seed is a finding about the seed."""
    payload = _run_json(capsys, ["gaps", "-k", "3", "--runs", "1", "--resamples", "0"])

    assert payload["stability"]["verdict"] == "stability unverified"
    sparse = [
        f
        for f in payload["corpus_coverage_findings"]
        if f["type"] == gaps_mod.FINDING_SPARSE_REGION
    ]
    assert sparse
    assert all(finding["asserted"] is False for finding in sparse)


def test_the_corpus_half_is_byte_identical_across_runs(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Design note SS4: reader verbs over a fixed atlas are byte-reproducible."""
    _run(["gaps", "-k", "3", "--json"])
    first = capsys.readouterr().out
    _run(["gaps", "-k", "3", "--json"])
    second = capsys.readouterr().out

    assert first == second


def test_the_sparse_region_found_is_the_isolated_one(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Guards the fixture's premise: C, not A or B, is what sparsity detection finds."""
    payload = _run_json(capsys, ["gaps", "-k", "3"])

    sparse = [
        f
        for f in payload["corpus_coverage_findings"]
        if f["type"] == gaps_mod.FINDING_SPARSE_REGION
    ]
    assert len(sparse) == 1
    assert set(sparse[0]["members"]) == {"c0", "c1"}


def test_sparse_regions_and_voids_share_one_region_numbering(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two numbering schemes in one payload would make "region 2" ambiguous within it.

    Voids name reported region ids, not k-means' internal cluster labels, which
    are arbitrary per run. With k=3 the reported ids are exactly 0..2 and every
    void must be a pair drawn from them.
    """
    payload = _run_json(capsys, ["gaps", "-k", "3"])
    findings = payload["corpus_coverage_findings"]

    valid = set(range(3))
    sparse_ids = {f["region_id"] for f in findings if f["type"] == gaps_mod.FINDING_SPARSE_REGION}
    assert sparse_ids <= valid
    for void in (f for f in findings if f["type"] == gaps_mod.FINDING_VOID):
        left, right = void["between_regions"]
        assert {left, right} <= valid
        assert left < right
    # The sparse region is the most isolated, so it sorts first under the shared
    # numbering -- and voids touching it must use that same id.
    assert sparse_ids == {0}


def test_inter_region_voids_are_reported_as_corpus_coverage(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    payload = _run_json(capsys, ["gaps", "-k", "3"])

    voids = [f for f in payload["corpus_coverage_findings"] if f["type"] == gaps_mod.FINDING_VOID]
    assert voids
    for void in voids:
        assert void["evidence_class"] == gaps_mod.EVIDENCE_CORPUS
        assert void["measurement"]["nearest_item_distance"] > 0
