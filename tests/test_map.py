"""Tests for ``embeddings map`` — the 2-D projection and its self-contained artifact.

Three properties get disproportionate attention here, because a projection is the
most persuasive and most misleading thing this CLI produces.

* **The artifact carries its own provenance and caveat.** A screenshot of a
  scatter plot travels without the command that made it, so the algorithm, its
  parameters, the seed, the explained variance, and the parameter-dependence
  caveat of ``docs/design-notes/semantic-measurement.md`` §3 must all be *inside
  the file*. ``test_artifact_embeds_the_full_provenance_block`` and
  ``test_artifact_states_the_parameter_dependence_caveat`` assert exactly that.
* **The artifact is genuinely one file.** No ``src=``/``href=`` attribute, no
  ``@import``, no ``url(``, no ``fetch``/``XHR``/``WebSocket``. The only URL
  allowed anywhere in the document is the embedding endpoint, which is
  provenance *text*, not a request — the test strips exactly that string and
  then asserts no scheme survives.
* **Corpus text cannot inject markup.** Item text is untrusted input. It reaches
  the page only through a JSON block with ``<``/``>``/``&`` escaped to unicode
  sequences, read by inline JS into ``textContent`` — never ``innerHTML``, never
  server-side interpolation into an attribute.

``map`` is a reader verb, so it needs no gateway at all;
:func:`tests._fakes.blocked_sockets` proves it directly. And writing a file makes
it a write verb, so the survey-verb gating applies: no ``--apply``, no bytes —
asserted with the same whole-tree hash snapshot ``tests/test_survey.py`` uses.
"""

from __future__ import annotations

import hashlib
import html as html_mod
import json
import re
from pathlib import Path

import numpy as np
import pytest

from embeddings.atlas.io import write_atlas
from embeddings.atlas.schema import SCHEMA_VERSION, AtlasManifest, AtlasRecord
from embeddings.cli import main
from embeddings.cli._commands import map as map_mod
from embeddings.cli._commands._atlas_arg import MISSING_ATLAS_HINT
from embeddings.measure import VERDICT_UNVERIFIED
from tests._fakes import blocked_sockets, deterministic_vector

DIM = 8
MODEL = "fake-embedding-model"
ENDPOINT = "http://localhost:8002/v1"

# --------------------------------------------------------------------------
# Fixtures
# --------------------------------------------------------------------------


def _tree_snapshot(root: Path) -> dict[str, str]:
    """Content-addressed snapshot of every path under ``root`` (see test_survey.py)."""
    snapshot: dict[str, str] = {}
    for entry in sorted(root.rglob("*")):
        key = str(entry.relative_to(root))
        if entry.is_dir():
            snapshot[key + "/"] = "<dir>"
        else:
            snapshot[key] = hashlib.sha256(entry.read_bytes()).hexdigest()
    return snapshot


def _write_atlas(target: Path, texts: list[str], vectors: np.ndarray) -> Path:
    manifest = AtlasManifest(
        schema_version=SCHEMA_VERSION,
        model=MODEL,
        endpoint=ENDPOINT,
        params={"batch_size": 32},
        seed=0,
        limitations=[],
        uncertain=[],
    )
    records = [
        AtlasRecord(id=f"doc-{index}", text=text, vector_index=index, metadata={})
        for index, text in enumerate(texts)
    ]
    return write_atlas(manifest, records, vectors.astype(np.float32), path=target)


def blob_atlas(target: Path, *, blobs: int = 3, per_blob: int = 6, spread: float = 0.02) -> Path:
    """``blobs`` tight, well-separated groups — genuine structure, reproducible from text."""
    texts: list[str] = []
    rows: list[list[float]] = []
    for blob in range(blobs):
        centre = np.zeros(DIM)
        centre[blob % DIM] = 1.0
        for member in range(per_blob):
            text = f"blob-{blob}-item-{member}"
            jitter = np.asarray(deterministic_vector(text, DIM)) * spread
            texts.append(text)
            rows.append(list(centre + jitter))
    return _write_atlas(target, texts, np.asarray(rows))


@pytest.fixture
def sandbox(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def atlas_dir(sandbox: Path) -> Path:
    return blob_atlas(sandbox / "atlas")


def _run(argv: list[str]) -> int:
    return main(argv)


def _json_out(capsys: pytest.CaptureFixture[str]) -> dict:
    return json.loads(capsys.readouterr().out)


def _artifact(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------
# Registration and flag surface
# --------------------------------------------------------------------------


def test_map_is_registered_on_the_root_parser(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        _run(["--help"])
    assert "map" in capsys.readouterr().out


def test_map_declares_its_own_json_flag(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        _run(["map", "--help"])
    assert "--json" in capsys.readouterr().out


def test_every_map_flag_has_help_text(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        _run(["map", "--help"])
    text = capsys.readouterr().out
    for flag in ("--atlas", "--json", "--html", "--apply", "--force", "--seed", "--clusters"):
        assert flag in text
    # No flag may be listed without a description following it on its own block.
    assert "Emit structured JSON." in text


def test_missing_atlas_exits_1_with_the_shared_hint(
    sandbox: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(["map", "--atlas", str(sandbox / "nope")]) == 1
    assert MISSING_ATLAS_HINT in capsys.readouterr().err


def test_map_needs_no_gateway(atlas_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with blocked_sockets():
        assert _run(["map", "--atlas", str(atlas_dir), "--json"]) == 0
    assert _json_out(capsys)["algorithm"] == map_mod.ALGORITHM


# --------------------------------------------------------------------------
# The summary (no --html)
# --------------------------------------------------------------------------


def test_json_summary_shape(atlas_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["map", "--atlas", str(atlas_dir), "--json"]) == 0
    payload = _json_out(capsys)
    for key in (
        "atlas",
        "model",
        "endpoint",
        "items",
        "dimensions",
        "algorithm",
        "params",
        "seed",
        "explained_variance",
        "points",
        "stability",
        "caveat",
        "uncertain",
        "html",
    ):
        assert key in payload, f"missing --json key {key!r}"

    assert payload["algorithm"] == "pca"
    assert payload["items"] == len(payload["points"])
    assert payload["html"] is None
    assert payload["params"]["components"] == 2
    point = payload["points"][0]
    assert set(point) == {"id", "text", "x", "y", "cluster"}


def test_explained_variance_is_reported_per_axis_and_bounded(
    atlas_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(["map", "--atlas", str(atlas_dir), "--json"]) == 0
    variance = _json_out(capsys)["explained_variance"]
    assert set(variance) == {"pc1", "pc2", "total"}
    assert 0.0 <= variance["pc1"] <= 1.0
    assert 0.0 <= variance["pc2"] <= 1.0
    # PCA orders components by variance, and the two together cannot exceed the whole.
    assert variance["pc1"] >= variance["pc2"]
    assert variance["total"] == pytest.approx(variance["pc1"] + variance["pc2"])
    assert variance["total"] <= 1.0 + 1e-9
    # Three axis-aligned blobs in 8-D put nearly all their variance in a 2-D plane.
    assert variance["total"] > 0.9


def test_text_summary_names_the_algorithm_and_carries_the_caveat(
    atlas_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(["map", "--atlas", str(atlas_dir)]) == 0
    out = capsys.readouterr().out
    assert "pca" in out
    assert "explained variance" in out.lower()
    assert "not faithful" in out


def test_projection_is_deterministic(atlas_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert _run(["map", "--atlas", str(atlas_dir), "--json"]) == 0
    first = capsys.readouterr().out
    assert _run(["map", "--atlas", str(atlas_dir), "--json"]) == 0
    assert capsys.readouterr().out == first


def test_the_seed_does_not_move_the_projection_but_is_still_recorded(
    atlas_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """PCA is seed-free; §3 still requires the seed on every structural payload."""
    assert _run(["map", "--atlas", str(atlas_dir), "--json", "--seed", "0", "-k", "0"]) == 0
    zero = _json_out(capsys)
    assert _run(["map", "--atlas", str(atlas_dir), "--json", "--seed", "99", "-k", "0"]) == 0
    ninety_nine = _json_out(capsys)

    assert zero["points"] == ninety_nine["points"]
    assert zero["seed"] == 0
    assert ninety_nine["seed"] == 99


def test_pca_sign_convention_is_pinned(atlas_dir: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """An unfixed SVD sign would flip the picture between numpy builds."""
    assert _run(["map", "--atlas", str(atlas_dir), "--json"]) == 0
    assert "sign_convention" in _json_out(capsys)["params"]


def test_cluster_colouring_carries_a_stability_verdict(
    atlas_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["map", "--atlas", str(atlas_dir), "--json", "-k", "3"]
    argv += ["--runs", "3", "--resamples", "4"]
    assert _run(argv) == 0
    payload = _json_out(capsys)
    assert payload["stability"] is not None
    assert payload["stability"]["seed"] == payload["seed"]
    assert payload["stability"]["params"]["clusters"] == 3
    assert {point["cluster"] for point in payload["points"]} == {0, 1, 2}


def test_single_run_colouring_is_labelled_stability_unverified(
    atlas_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    argv = ["map", "--atlas", str(atlas_dir), "--json", "-k", "3"]
    argv += ["--runs", "1", "--resamples", "0"]
    assert _run(argv) == 0
    payload = _json_out(capsys)
    assert payload["stability"]["verdict"] == VERDICT_UNVERIFIED
    assert payload["uncertain"], "an unverified colouring belongs in uncertain[]"


def test_clusters_zero_disables_colouring(
    atlas_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(["map", "--atlas", str(atlas_dir), "--json", "-k", "0"]) == 0
    payload = _json_out(capsys)
    assert payload["stability"] is None
    assert all(point["cluster"] is None for point in payload["points"])
    assert payload["params"]["clusters"] == 0


def test_negative_clusters_is_a_user_error(
    atlas_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(["map", "--atlas", str(atlas_dir), "-k", "-2"]) == 1
    assert "hint:" in capsys.readouterr().err


def test_empty_atlas_exits_1(sandbox: Path, capsys: pytest.CaptureFixture[str]) -> None:
    target = sandbox / "empty"
    _write_atlas(target, [], np.zeros((0, DIM)))
    assert _run(["map", "--atlas", str(target)]) == 1
    assert "hint:" in capsys.readouterr().err


# --------------------------------------------------------------------------
# --html is a write verb: dry run by default
# --------------------------------------------------------------------------


def test_html_dry_run_performs_zero_filesystem_writes(
    sandbox: Path, atlas_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = _tree_snapshot(sandbox)
    assert _run(["map", "--atlas", str(atlas_dir), "--html", str(sandbox / "m.html")]) == 0
    capsys.readouterr()
    assert _tree_snapshot(sandbox) == before, "a preview must write nothing at all"


def test_html_dry_run_previews_path_items_and_algorithm(
    sandbox: Path, atlas_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = sandbox / "m.html"
    assert _run(["map", "--atlas", str(atlas_dir), "--html", str(target), "--json"]) == 0
    payload = _json_out(capsys)
    assert payload["html"] == {
        "applied": False,
        "path": str(target),
        "exists": False,
        "would_replace": False,
    }
    assert payload["items"] == 18
    assert payload["algorithm"] == "pca"


def test_html_dry_run_text_mode_says_it_wrote_nothing(
    sandbox: Path, atlas_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = sandbox / "m.html"
    assert _run(["map", "--atlas", str(atlas_dir), "--html", str(target)]) == 0
    out = capsys.readouterr().out
    assert "--apply" in out
    assert str(target) in out
    assert not target.exists()


def test_apply_writes_the_artifact(
    sandbox: Path, atlas_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = sandbox / "m.html"
    argv = ["map", "--atlas", str(atlas_dir), "--html", str(target), "--apply", "--json"]
    assert _run(argv) == 0
    payload = _json_out(capsys)
    assert payload["html"]["applied"] is True
    assert target.is_file()
    assert _artifact(target).startswith("<!DOCTYPE html>")


def test_apply_writes_atomically_and_leaves_no_scratch_behind(
    sandbox: Path, atlas_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[Path] = []
    real = map_mod.write_atomic

    def _recording(path, data):  # type: ignore[no-untyped-def]
        calls.append(Path(path))
        real(path, data)

    target = sandbox / "out" / "m.html"
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(map_mod, "write_atomic", _recording)
        argv = ["map", "--atlas", str(atlas_dir), "--html", str(target), "--apply"]
        assert _run(argv) == 0
    capsys.readouterr()
    assert calls == [target], "the artifact must go through write_atomic, not open()"
    assert [entry.name for entry in target.parent.iterdir()] == ["m.html"]


def test_apply_onto_an_existing_artifact_exits_1_and_names_force(
    sandbox: Path, atlas_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = sandbox / "m.html"
    target.write_text("previous artifact", encoding="utf-8")
    before = _tree_snapshot(sandbox)

    argv = ["map", "--atlas", str(atlas_dir), "--html", str(target), "--apply"]
    assert _run(argv) == 1
    err = capsys.readouterr().err
    assert "--force" in err
    assert _tree_snapshot(sandbox) == before, "a refused apply must leave the file untouched"


def test_force_replaces_the_artifact(
    sandbox: Path, atlas_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = sandbox / "m.html"
    target.write_text("previous artifact", encoding="utf-8")
    argv = ["map", "--atlas", str(atlas_dir), "--html", str(target), "--apply", "--force"]
    assert _run(argv) == 0
    capsys.readouterr()
    assert "previous artifact" not in _artifact(target)


def test_force_without_apply_still_writes_nothing(
    sandbox: Path, atlas_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    before = _tree_snapshot(sandbox)
    argv = ["map", "--atlas", str(atlas_dir), "--html", str(sandbox / "m.html"), "--force"]
    assert _run(argv) == 0
    capsys.readouterr()
    assert _tree_snapshot(sandbox) == before


def test_apply_without_html_is_a_user_error(
    atlas_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(["map", "--atlas", str(atlas_dir), "--apply"]) == 1
    err = capsys.readouterr().err
    assert "--html" in err


def test_diagnostics_go_to_stderr_and_stdout_stays_pure_json(
    sandbox: Path, atlas_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = sandbox / "m.html"
    argv = ["map", "--atlas", str(atlas_dir), "--html", str(target), "--apply", "--json"]
    assert _run(argv) == 0
    captured = capsys.readouterr()
    json.loads(captured.out)
    assert str(target) in captured.err


# --------------------------------------------------------------------------
# The artifact itself
# --------------------------------------------------------------------------


@pytest.fixture
def artifact(sandbox: Path, atlas_dir: Path, capsys: pytest.CaptureFixture[str]) -> str:
    target = sandbox / "m.html"
    argv = ["map", "--atlas", str(atlas_dir), "--html", str(target), "--apply", "-k", "3"]
    assert _run(argv) == 0
    capsys.readouterr()
    return _artifact(target)


def test_artifact_makes_no_external_request(artifact: str) -> None:
    """One file: nothing may point outward, and nothing may fetch."""
    for marker in (
        "src=",
        "href=",
        "<link",
        "<iframe",
        "<img",
        "@import",
        "url(",
        "fetch(",
        "XMLHttpRequest",
        "WebSocket",
        "EventSource",
        "importScripts",
        "navigator.sendBeacon",
    ):
        assert marker not in artifact, f"external-request marker {marker!r} in the artifact"


def test_the_only_url_in_the_artifact_is_the_provenance_endpoint(artifact: str) -> None:
    """The endpoint is recorded *text*, not a request; nothing else may carry a scheme."""
    assert ENDPOINT in artifact or html_mod.escape(ENDPOINT, quote=True) in artifact
    stripped = artifact.replace(html_mod.escape(ENDPOINT, quote=True), "").replace(ENDPOINT, "")
    assert "http://" not in stripped
    assert "https://" not in stripped


def test_artifact_ships_both_themes(artifact: str) -> None:
    assert "prefers-color-scheme: dark" in artifact
    assert ':root[data-theme="dark"]' in artifact
    assert ':root[data-theme="light"]' in artifact


def test_artifact_embeds_the_full_provenance_block(
    sandbox: Path, atlas_dir: Path, artifact: str
) -> None:
    """Everything a reader needs to judge the picture, inside the picture's own file."""
    assert html_mod.escape(str(atlas_dir), quote=True) in artifact
    assert MODEL in artifact
    assert "pca" in artifact
    assert "explained variance" in artifact.lower()
    assert "seed" in artifact.lower()
    assert "18 items" in artifact or ">18<" in artifact
    assert map_mod.SIGN_CONVENTION in artifact


def test_artifact_states_the_parameter_dependence_caveat(artifact: str) -> None:
    """§3: a projection separated from its parameters asserts more than the measurement."""
    assert html_mod.escape(map_mod.PROJECTION_CAVEAT, quote=True) in artifact
    lowered = map_mod.PROJECTION_CAVEAT.lower()
    assert "not faithful" in lowered
    assert "lossy" in lowered
    assert "parameters" in lowered
    assert "seed" in lowered


def test_artifact_reports_the_explained_variance_numerically(artifact: str) -> None:
    assert re.search(r"PC1[^<]*</dt>\s*<dd[^>]*>\s*\d+\.\d+%", artifact)
    assert re.search(r"PC2[^<]*</dt>\s*<dd[^>]*>\s*\d+\.\d+%", artifact)


def test_artifact_renders_every_item_as_a_mark(artifact: str) -> None:
    assert artifact.count("<circle") == 18


def test_artifact_render_is_byte_deterministic(sandbox: Path, atlas_dir: Path) -> None:
    first = map_mod.render_html(_payload(atlas_dir))
    second = map_mod.render_html(_payload(atlas_dir))
    assert first == second


def _payload(atlas_dir: Path) -> dict:
    import argparse

    args = argparse.Namespace(
        atlas=str(atlas_dir),
        json=False,
        html=None,
        apply=False,
        force=False,
        seed=0,
        clusters=3,
        runs=3,
        resamples=4,
        max_iterations=100,
        tolerance=1e-6,
    )
    return map_mod.build_payload(args)


def test_cluster_colouring_is_legible_in_the_artifact(artifact: str) -> None:
    assert "cluster" in artifact.lower()
    # A stability verdict travels with the colouring, never a bare partition.
    assert "stability" in artifact.lower()


def test_artifact_degrades_without_javascript(artifact: str) -> None:
    """The plot and the provenance are server-rendered; JS only adds the tooltip."""
    body = artifact.split("<script", 1)[0]
    assert "<circle" in body
    assert "explained variance" in body.lower()
    assert html_mod.escape(map_mod.PROJECTION_CAVEAT, quote=True) in body


# --------------------------------------------------------------------------
# Untrusted corpus text
# --------------------------------------------------------------------------

HOSTILE = "<script>alert(\"pwned\")</script> & 'quoted' <img onerror=x>"


@pytest.fixture
def hostile_artifact(sandbox: Path, capsys: pytest.CaptureFixture[str]) -> str:
    target = sandbox / "hostile" / "m.html"
    atlas = sandbox / "hostile-atlas"
    texts = [HOSTILE] + [f"benign-{index}" for index in range(5)]
    rows = [deterministic_vector(text, DIM) for text in texts]
    _write_atlas(atlas, texts, np.asarray(rows))
    argv = ["map", "--atlas", str(atlas), "--html", str(target), "--apply", "-k", "2"]
    assert _run(argv) == 0
    capsys.readouterr()
    return _artifact(target)


def test_hostile_item_text_cannot_inject_markup(hostile_artifact: str) -> None:
    assert "<script>alert" not in hostile_artifact
    assert "<img onerror" not in hostile_artifact
    # Exactly the script tags this renderer emits itself: the data block and the
    # behaviour block. Any third one would be injected.
    assert hostile_artifact.count("<script") == 2


def test_hostile_item_text_survives_as_data(hostile_artifact: str) -> None:
    """Escaped, not dropped — the reader must still be able to read the item."""
    assert "\\u003cscript\\u003ealert" in hostile_artifact
    assert "pwned" in hostile_artifact


def test_item_text_never_reaches_innerhtml(hostile_artifact: str) -> None:
    assert "innerHTML" not in hostile_artifact
    assert "textContent" in hostile_artifact


def test_json_data_block_escapes_angle_brackets_and_ampersands(hostile_artifact: str) -> None:
    block = hostile_artifact.split('type="application/json">', 1)[1].split("</script>", 1)[0]
    assert "<" not in block, "an unescaped '<' in the data block can close the script element"
    assert ">" not in block
    assert "&" not in block
    # …and it is still valid JSON once the browser hands it to JSON.parse.
    parsed = json.loads(block)
    assert any(point["text"] == HOSTILE for point in parsed["points"])
