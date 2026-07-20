"""``embeddings map`` — project an Atlas to 2-D, and ship the caveat with the picture.

Why PCA, and why not UMAP or t-SNE
-----------------------------------
The projection is **principal component analysis, computed from a thin SVD of
the mean-centred item matrix, on numpy alone**. Three properties decided it, and
each of them is a constraint this CLI is under rather than a preference:

* **It is deterministic.** No initialization, no seed, no optimizer. The same
  Atlas projects to the same coordinates on every run — which is exactly the
  reader-verb guarantee ``docs/design-notes/semantic-measurement.md`` §4 makes
  for everything downstream of the Atlas boundary. (The SVD's sign is arbitrary
  per component, so it is pinned here — see :data:`SIGN_CONVENTION`.)
* **Its axes mean something statable.** Each component carries a share of the
  total variance, and that share is reported, so a reader can see *how much* of
  the structure the 2-D view actually captures. A view holding 12% of the
  variance and a view holding 95% look equally convincing on screen; only the
  number distinguishes them.
* **It does not manufacture apparent clusters.** §3 names UMAP and t-SNE
  specifically: they reliably produce well-separated blobs, from noise as
  readily as from signal, and the specific structure shifts with
  ``n_neighbors``, ``min_dist``, perplexity — and with the seed alone. PCA is a
  linear rotation; it can hide structure, but it does not invent it. Adding
  ``umap-learn`` or ``scikit-learn`` would also widen a dependency surface this
  project keeps one library wide.

What PCA gives up, stated plainly
----------------------------------
It is linear, so genuinely curved or manifold structure is flattened rather than
unrolled, and two regions can land on top of each other because the plane that
best preserves *variance* is not the plane that best separates *meaning*. Low
explained variance means the picture is mostly a shadow. This is a real loss and
is why the honest framing is "PCA shows you a faithful shadow" rather than "PCA
shows you the space".

The artifact carries its own provenance
----------------------------------------
``--html`` writes a single self-contained file. It embeds the atlas path, model,
endpoint, algorithm, every parameter, the seed, the item count, the per-axis
explained variance, and :data:`PROJECTION_CAVEAT` in plain prose — because the
one thing certain about a scatter plot is that it will be screenshotted and
shared without the command that made it. Provenance stapled to the command line
is provenance that does not travel; provenance inside the file does.

The file references nothing outside itself: no stylesheet, no font, no image, no
script, no ``fetch``. The plot, the provenance, and the caveat are all rendered
server-side into the document, so it is fully readable with JavaScript disabled;
the small inline script only adds hover/click inspection of item text. Corpus
text is untrusted input and reaches the page **only** through a JSON block with
``<``, ``>`` and ``&`` escaped to unicode sequences, read into ``textContent`` —
never interpolated into markup, never through ``innerHTML``.

``--html`` makes this a write verb
-----------------------------------
So it follows ``survey``'s gating exactly: previews by default and writes zero
bytes, ``--apply`` commits, an existing file is refused (exit 1) with a hint
naming ``--force``, and the commit goes through
:func:`embeddings._atomic.write_atomic` so a killed run never leaves a
half-written page that still opens in a browser.
"""

from __future__ import annotations

import argparse
import html as html_escaping
import json
from pathlib import Path
from string import Template
from typing import Sequence

import numpy as np

from embeddings._atomic import write_atomic
from embeddings.atlas.io import Atlas
from embeddings.cli._commands._atlas_arg import add_atlas_argument, load_atlas
from embeddings.cli._commands.regions import (
    DEFAULT_MAX_ITERATIONS,
    DEFAULT_TOLERANCE,
    cluster_labels,
)
from embeddings.cli._errors import EXIT_ENV_ERROR, EXIT_USER_ERROR, CliError
from embeddings.cli._output import emit_diagnostic, emit_result
from embeddings.measure import StabilityResult, measure_stability
from embeddings.measure.stability import DEFAULT_RESAMPLES, DEFAULT_RUNS

#: The projection this verb implements, stamped on every payload and artifact.
ALGORITHM = "pca"
#: Number of principal components kept. Two, because the output is a page.
COMPONENTS = 2

#: How the arbitrary per-component SVD sign is pinned. Without this, an
#: otherwise-deterministic projection could come out mirrored on a different
#: numpy/LAPACK build, and two runs a reader believes identical would disagree.
SIGN_CONVENTION = "largest-magnitude loading forced positive"

#: Axes share one scale, so within-plane distances stay comparable; stretching
#: each axis to fill the box would exaggerate the weaker component.
ASPECT = "uniform scale on both axes"

#: Default number of colour groups. ``0`` (or ``1``) turns colouring off.
DEFAULT_CLUSTERS = 5
#: Default base seed. PCA ignores it; the colour clustering does not.
DEFAULT_SEED = 0

#: The prose that must travel with the picture. Stated as a module constant so
#: the artifact, the text summary, and the ``--json`` payload cannot drift into
#: three differently-hedged versions of the same warning.
PROJECTION_CAVEAT = (
    "This is a lossy 2-D projection of a high-dimensional embedding space. Only the share "
    "of variance reported above survives the projection; the rest was discarded to fit the "
    "page. Distances you can see here are not faithful to distances in the space -- "
    "especially the distance between apparent clusters, which a projection is free to "
    "rearrange. A gap on this page is not evidence of a gap in the model. The layout is "
    "conditioned on the algorithm, its parameters, the item set, and (for the colouring) the "
    "seed: change any of them and the picture changes, so a structure visible only at these "
    "settings is a finding about the settings, not about the model. Read this picture only "
    "together with the provenance above; separated from it, a screenshot asserts more than "
    "the measurement supports. See docs/design-notes/semantic-measurement.md section 3."
)

#: Categorical colours, one deliberately-stepped set per surface. Cluster
#: identity is the only thing they encode; nothing else on the page wears them.
_LIGHT_COLOURS = (
    "#b65b38",
    "#4b3ba6",
    "#0e8f76",
    "#2a78d6",
    "#a06a00",
    "#c0417a",
    "#5c6b1f",
    "#7a4bbf",
)
_DARK_COLOURS = (
    "#cb6e44",
    "#877ae0",
    "#1fa083",
    "#3987e5",
    "#c98500",
    "#d55181",
    "#93a53a",
    "#a97ae8",
)

_PLOT_WIDTH = 720
_PLOT_HEIGHT = 460
_PLOT_PADDING = 26
_POINT_RADIUS = 5.5


# --------------------------------------------------------------------------
# The projection
# --------------------------------------------------------------------------


def project(data: np.ndarray) -> tuple[np.ndarray, list[float]]:
    """PCA to :data:`COMPONENTS` dimensions; returns ``(coords, explained_ratios)``.

    Mean-centres the item matrix, takes a thin SVD, and scores each item on the
    leading components. ``explained_ratios`` is each kept component's share of
    the total variance, in ``[0, 1]`` — the number that says how much of the
    structure the picture actually holds.

    Degenerate inputs are handled rather than raised on: a single item, or a set
    of identical vectors, has zero total variance, and both ratios are then
    ``0.0`` with every item at the origin. Fewer than :data:`COMPONENTS`
    available components (a 1-D embedding, or two items) pads the missing axis
    with zeros rather than fabricating one.
    """
    array = np.asarray(data, dtype=np.float64)
    centred = array - array.mean(axis=0, keepdims=True)
    left, singular, right = np.linalg.svd(centred, full_matrices=False)

    # Pin the sign: SVD determines each component only up to a sign, so without
    # this the same Atlas could project mirrored on a different LAPACK build.
    for component in range(right.shape[0]):
        pivot = int(np.argmax(np.abs(right[component])))
        if right[component, pivot] < 0.0:
            right[component] *= -1.0
            left[:, component] *= -1.0

    variances = singular**2
    total = float(variances.sum())
    coords = np.zeros((array.shape[0], COMPONENTS), dtype=np.float64)
    explained = [0.0] * COMPONENTS
    for axis in range(min(COMPONENTS, singular.shape[0])):
        coords[:, axis] = left[:, axis] * singular[axis]
        explained[axis] = float(variances[axis] / total) if total > 0.0 else 0.0
    return coords, explained


def _item_matrix(atlas: Atlas) -> np.ndarray:
    """One row per Atlas record, in record order, as float64.

    Records may share a ``vector_index`` (identical text embeds once), so this
    is a gather rather than a slice.
    """
    if not atlas.records:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"the atlas at {atlas.path} contains no records — there is nothing to map",
            remediation="re-run `embeddings survey <corpus> --apply` with a non-empty corpus",
        )
    if atlas.vectors.ndim != 2 or atlas.vectors.shape[1] == 0:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=f"the atlas at {atlas.path} has no vectors to project",
            remediation=(
                "vectors.npy is regenerable bulk and is git-ignored; re-run the survey "
                "that produced this atlas to rebuild it"
            ),
        )
    indices = [record.vector_index for record in atlas.records]
    if min(indices) < 0 or max(indices) >= atlas.vectors.shape[0]:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=(
                f"the atlas at {atlas.path} has a record pointing outside its vector array "
                f"(indices {min(indices)}..{max(indices)}, {atlas.vectors.shape[0]} vectors)"
            ),
            remediation="re-run the survey that produced this atlas to rebuild it",
        )
    return np.asarray(atlas.vectors[indices], dtype=np.float64)


# --------------------------------------------------------------------------
# Payload assembly — one fold, rendered three ways
# --------------------------------------------------------------------------


def _validate(*, clusters: int, runs: int, resamples: int) -> None:
    if clusters < 0:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"--clusters must be zero or more, got {clusters}",
            remediation=(
                f"pass --clusters 0 to draw the map uncoloured, or 2 or more to colour it "
                f"by k-means groups (default: {DEFAULT_CLUSTERS})"
            ),
        )
    if runs < 1:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"--runs must be at least 1, got {runs}",
            remediation=(
                "pass --runs 2 or higher to obtain seed-rerun evidence for the colouring; "
                "--runs 1 is reported as 'stability unverified'"
            ),
        )
    if resamples < 0:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"--resamples must be zero or more, got {resamples}",
            remediation="pass --resamples 0 to skip bootstrap resampling",
        )


def _colour(
    data: np.ndarray,
    *,
    clusters: int,
    seed: int,
    runs: int,
    resamples: int,
    max_iterations: int,
    tolerance: float,
) -> tuple[list[int | None], StabilityResult | None]:
    """k-means colour groups plus the stability of that partition, or ``(None…, None)``.

    Colouring a projection *is* a structural claim, so it never ships bare: the
    same clustering is re-run across seeds and bootstrap resamples through
    :func:`embeddings.measure.measure_stability`, and the verdict rides along on
    the payload and into the artifact. §3's rule applies to a picture exactly as
    it applies to a table.
    """
    if clusters < 2:
        return [None] * data.shape[0], None

    def _cluster(matrix: np.ndarray, run_seed: int) -> np.ndarray:
        return cluster_labels(
            matrix,
            run_seed,
            clusters=clusters,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )

    params: dict[str, object] = {
        "algorithm": "kmeans",
        "clusters": clusters,
        "max_iterations": max_iterations,
        "tolerance": tolerance,
    }
    labels = _cluster(data, seed)
    stability = measure_stability(
        data, _cluster, params=params, seed=seed, runs=runs, resamples=resamples
    )
    return [int(label) for label in labels], stability


def build_payload(args: argparse.Namespace) -> dict:
    """The single fold behind ``--json``, the text summary, and the HTML artifact.

    Kept as one function returning one dict so the three renderings cannot
    disagree about what was measured — the artifact is a view of this payload,
    not a second, parallel computation.
    """
    clusters = int(args.clusters)
    runs = int(args.runs)
    resamples = int(args.resamples)
    seed = int(args.seed)
    max_iterations = int(args.max_iterations)
    tolerance = float(args.tolerance)
    _validate(clusters=clusters, runs=runs, resamples=resamples)

    atlas = load_atlas(args)
    data = _item_matrix(atlas)
    effective = min(clusters, data.shape[0]) if clusters >= 2 else 0

    coords, explained = project(data)
    labels, stability = _colour(
        data,
        clusters=effective,
        seed=seed,
        runs=runs,
        resamples=resamples,
        max_iterations=max_iterations,
        tolerance=tolerance,
    )

    uncertain: list[str] = []
    if stability is not None and stability.is_uncertain:
        uncertain.append(stability.uncertain_note(f"the {effective}-group colouring on this map"))
    uncertain.append(
        f"the 2-D layout itself: {ALGORITHM} keeps "
        f"{(explained[0] + explained[1]) * 100:.1f}% of the total variance, so distances "
        "between the marks — above all between apparent clusters — are not faithful to "
        "distances in the space"
    )

    params: dict[str, object] = {
        "algorithm": ALGORITHM,
        "components": COMPONENTS,
        "sign_convention": SIGN_CONVENTION,
        "aspect": ASPECT,
        "clusters": effective,
        "runs": runs,
        "resamples": resamples,
        "max_iterations": max_iterations,
        "tolerance": tolerance,
    }
    return {
        "atlas": str(atlas.path),
        "model": atlas.manifest.model,
        "endpoint": atlas.manifest.endpoint,
        "items": int(data.shape[0]),
        "dimensions": int(data.shape[1]),
        "algorithm": ALGORITHM,
        "params": params,
        "seed": seed,
        "explained_variance": {
            "pc1": round(explained[0], 6),
            "pc2": round(explained[1], 6),
            "total": round(explained[0] + explained[1], 6),
        },
        "points": [
            {
                "id": record.id,
                "text": record.text,
                "x": round(float(coords[index, 0]), 6),
                "y": round(float(coords[index, 1]), 6),
                "cluster": labels[index],
            }
            for index, record in enumerate(atlas.records)
        ],
        "stability": None if stability is None else stability.to_dict(),
        "caveat": PROJECTION_CAVEAT,
        "uncertain": uncertain,
        "html": None,
    }


# --------------------------------------------------------------------------
# Rendering
# --------------------------------------------------------------------------


def _escape(value: object) -> str:
    """HTML-escape anything on its way into the document, quotes included."""
    return html_escaping.escape(str(value), quote=True)


def _json_block(payload: object) -> str:
    """Serialize ``payload`` for a ``<script type="application/json">`` block.

    ``<``, ``>`` and ``&`` become unicode escapes, so no item text — however
    hostile — can close the script element or open a tag; the escapes are still
    ordinary JSON string escapes, so ``JSON.parse`` yields the original text
    back unchanged. ``U+2028``/``U+2029`` are escaped too: they are legal in
    JSON strings but terminate a line in some JS parsers.
    """
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    for raw, escaped in (
        ("&", "\\u0026"),
        ("<", "\\u003c"),
        (">", "\\u003e"),
        ("\u2028", "\\u2028"),
        ("\u2029", "\\u2029"),
    ):
        text = text.replace(raw, escaped)
    return text


def _scale(points: Sequence[dict]) -> list[tuple[float, float]]:
    """Map projected coordinates into the SVG box under one shared scale.

    Both axes share a scale (:data:`ASPECT`) so the picture does not stretch the
    weaker component to fill the frame — that would exaggerate exactly the axis
    holding the least variance. The SVG y-axis points down, so PC2 is flipped
    to put larger values higher, matching every scatter plot ever drawn.
    """
    xs = [point["x"] for point in points]
    ys = [point["y"] for point in points]
    span = max(max(xs) - min(xs), max(ys) - min(ys)) if points else 0.0
    inner = min(_PLOT_WIDTH, _PLOT_HEIGHT) - 2 * _PLOT_PADDING
    factor = (inner / span) if span > 0.0 else 0.0
    mid_x = (max(xs) + min(xs)) / 2.0 if points else 0.0
    mid_y = (max(ys) + min(ys)) / 2.0 if points else 0.0
    return [
        (
            round(_PLOT_WIDTH / 2.0 + (point["x"] - mid_x) * factor, 2),
            round(_PLOT_HEIGHT / 2.0 - (point["y"] - mid_y) * factor, 2),
        )
        for point in points
    ]


def _marks(payload: dict) -> str:
    positions = _scale(payload["points"])
    rows = []
    for index, (point, (x, y)) in enumerate(zip(payload["points"], positions)):
        cluster = point["cluster"]
        slot = "none" if cluster is None else str(int(cluster) % len(_LIGHT_COLOURS))
        rows.append(
            f'<circle class="mark c{slot}" cx="{x}" cy="{y}" r="{_POINT_RADIUS}" '
            f'data-i="{index}" tabindex="0"></circle>'
        )
    return "\n".join(rows)


def _legend(payload: dict) -> str:
    counts: dict[int, int] = {}
    for point in payload["points"]:
        if point["cluster"] is not None:
            counts[int(point["cluster"])] = counts.get(int(point["cluster"]), 0) + 1
    if not counts:
        return '<p class="muted">Uncoloured: no clustering was requested (--clusters 0).</p>'
    chips = "\n".join(
        f'<li><span class="swatch c{cluster % len(_LIGHT_COLOURS)}"></span>'
        f"cluster {cluster} &middot; {counts[cluster]} items</li>"
        for cluster in sorted(counts)
    )
    return f'<ul class="legend">{chips}</ul>'


def _cluster_note(payload: dict) -> str:
    stability = payload["stability"]
    if stability is None:
        return (
            "No colouring was applied, so this map makes no claim about groups at all — "
            "only about where the projection puts each item."
        )
    return (
        f"Colour is a k-means partition into {payload['params']['clusters']} groups at seed "
        f"{payload['seed']}, stability verdict: {stability['verdict']} "
        f"({stability['measure']}, {stability['runs']} seed rerun(s), "
        f"{stability['resamples']} bootstrap resample(s)). k-means always returns exactly "
        "the k it was given, structure or no structure, so the verdict — not the picture — "
        "is what says whether these groups are real."
    )


def _provenance(payload: dict) -> str:
    variance = payload["explained_variance"]
    params = payload["params"]
    rows = [
        ("Atlas", payload["atlas"]),
        ("Model", payload["model"]),
        ("Endpoint", payload["endpoint"]),
        ("Items", f"{payload['items']} items in {payload['dimensions']} dimensions"),
        ("Algorithm", f"{payload['algorithm'].upper()} (thin SVD of the mean-centred matrix)"),
        ("PC1 explained variance", f"{variance['pc1'] * 100:.1f}%"),
        ("PC2 explained variance", f"{variance['pc2'] * 100:.1f}%"),
        ("Total kept", f"{variance['total'] * 100:.1f}% of the variance in this atlas"),
        ("Seed", str(payload["seed"])),
        ("Colour groups (k)", str(params["clusters"])),
        ("Stability runs", f"{params['runs']} seed rerun(s), {params['resamples']} resample(s)"),
        ("Sign convention", params["sign_convention"]),
        ("Aspect", params["aspect"]),
    ]
    return "\n".join(
        f"<div class='row'><dt>{_escape(term)}</dt><dd>{_escape(value)}</dd></div>"
        for term, value in rows
    )


def render_html(payload: dict) -> str:
    """Render ``payload`` as the self-contained artifact. A pure function of the payload.

    Byte-deterministic: every number is pre-rounded in the payload, the JSON
    block is key-sorted, and nothing here reads a clock, an environment
    variable, or an RNG.
    """
    data = {
        "points": payload["points"],
        "explained_variance": payload["explained_variance"],
        "params": payload["params"],
        "seed": payload["seed"],
    }
    return Template(_TEMPLATE).substitute(
        title=_escape(f"embeddings map — {payload['atlas']}"),
        atlas=_escape(payload["atlas"]),
        provenance=_provenance(payload),
        caveat=_escape(payload["caveat"]),
        cluster_note=_escape(_cluster_note(payload)),
        legend=_legend(payload),
        marks=_marks(payload),
        plot_width=str(_PLOT_WIDTH),
        plot_height=str(_PLOT_HEIGHT),
        light=_colour_rules(_LIGHT_COLOURS),
        dark=_colour_rules(_DARK_COLOURS),
        data=_json_block(data),
    )


def _colour_rules(colours: Sequence[str]) -> str:
    return " ".join(f"--c{index}:{value};" for index, value in enumerate(colours))


# --------------------------------------------------------------------------
# Writing the artifact
# --------------------------------------------------------------------------


def _write_artifact(payload: dict, target: Path, *, apply: bool, force: bool) -> dict:
    exists = target.exists()
    if not apply:
        return {
            "applied": False,
            "path": str(target),
            "exists": exists,
            "would_replace": exists and force,
        }
    if exists and not force:
        raise CliError(
            code=EXIT_USER_ERROR,
            message=f"an artifact already exists at {target}",
            remediation=("re-run with --force to overwrite it, or pass a different --html <path>"),
        )
    try:
        write_atomic(target, render_html(payload).encode("utf-8"))
    except OSError as err:
        raise CliError(
            code=EXIT_ENV_ERROR,
            message=f"failed to write the map artifact to {target}: {err}",
            remediation="check the target directory is writable and has free space",
        ) from err
    emit_diagnostic(f"map: wrote {target}")
    return {"applied": True, "path": str(target), "exists": exists, "would_replace": exists}


def _render_text(payload: dict) -> str:
    variance = payload["explained_variance"]
    params = payload["params"]
    html_info = payload["html"]
    heading = "# map"
    if html_info is not None and not html_info["applied"]:
        heading = "# map (dry run — nothing written)"
    elif html_info is not None:
        heading = "# map (applied)"

    lines = [
        heading,
        "",
        f"atlas:         {payload['atlas']}",
        f"model:         {payload['model']}",
        f"items:         {payload['items']} x {payload['dimensions']}",
        f"algorithm:     {payload['algorithm']}",
        f"params:        components={params['components']}, k={params['clusters']}, "
        f"runs={params['runs']}, resamples={params['resamples']}",
        f"seed:          {payload['seed']}",
        f"explained variance: PC1 {variance['pc1'] * 100:.1f}%, PC2 "
        f"{variance['pc2'] * 100:.1f}% (total {variance['total'] * 100:.1f}%)",
    ]
    if payload["stability"] is not None:
        lines.append(f"colouring:     {payload['stability']['verdict']}")
    lines.append("")

    if html_info is not None:
        lines.append(f"artifact:      {html_info['path']}")
        if html_info["applied"]:
            lines.append("wrote a self-contained single-file HTML map (no external requests).")
        elif html_info["exists"] and not html_info["would_replace"]:
            lines.append("a file already exists there; re-run with --apply --force to replace it.")
        else:
            lines.append("re-run with --apply to write the artifact. Nothing was written.")
        lines.append("")

    lines.append(payload["caveat"])
    lines.append("")
    lines.append("uncertain (belongs in the atlas manifest's uncertain[]):")
    lines += [f"- {entry}" for entry in payload["uncertain"]]
    return "\n".join(lines)


def cmd_map(args: argparse.Namespace) -> int:
    json_mode = bool(getattr(args, "json", False))
    html_target = getattr(args, "html", None)

    if bool(args.apply) and not html_target:
        raise CliError(
            code=EXIT_USER_ERROR,
            message="--apply has nothing to write: map only writes a file with --html",
            remediation=(
                "pass --html <path> to name the artifact to write, or drop --apply to see "
                "the projection summary"
            ),
        )

    payload = build_payload(args)
    if html_target:
        payload["html"] = _write_artifact(
            payload, Path(html_target), apply=bool(args.apply), force=bool(args.force)
        )

    emit_result(payload if json_mode else _render_text(payload), json_mode=json_mode)
    return 0


def register(sub: argparse._SubParsersAction) -> None:
    p = sub.add_parser(
        "map",
        help="Project an atlas to 2-D (PCA) and optionally write a self-contained HTML map.",
        description=(
            "Project a surveyed atlas onto its two leading principal components and report "
            "how much of the total variance that 2-D view keeps. With --html, write a "
            "single self-contained page carrying the plot, its full provenance, and the "
            "parameter-dependence caveat — previewed by default, written only with --apply."
        ),
    )
    add_atlas_argument(p)
    p.add_argument(
        "--html",
        default=None,
        metavar="<path>",
        help="Write the map as a self-contained single-file HTML artifact at this path "
        "(no external requests, light and dark themes). Previews only without --apply.",
    )
    p.add_argument(
        "--apply",
        action="store_true",
        help="Actually write the --html artifact. Without this flag the command performs no "
        "filesystem writes at all.",
    )
    p.add_argument(
        "--force",
        action="store_true",
        help="With --apply, overwrite an existing file at the --html path. Without it, an "
        "existing file is refused rather than replaced.",
    )
    p.add_argument(
        "--clusters",
        "-k",
        type=int,
        default=DEFAULT_CLUSTERS,
        help=f"Colour the map by a k-means partition into this many groups (default: "
        f"{DEFAULT_CLUSTERS}); 0 draws it uncoloured. The colouring is a structural claim, "
        "so it always ships with the stability verdict measured below.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Base RNG seed (default: {DEFAULT_SEED}). The PCA projection is deterministic "
        "and ignores it; the colour clustering and its stability reruns are derived from it, "
        "and it is recorded on every output either way.",
    )
    p.add_argument(
        "--runs",
        type=int,
        default=DEFAULT_RUNS,
        help=f"Seed reruns used to measure the colouring's stability (default: {DEFAULT_RUNS}). "
        "--runs 1 with --resamples 0 yields no evidence and is reported as "
        "'stability unverified'.",
    )
    p.add_argument(
        "--resamples",
        type=int,
        default=DEFAULT_RESAMPLES,
        help=f"Bootstrap resamples of the item set (default: {DEFAULT_RESAMPLES}); 0 skips the "
        "bootstrap. This catches a colour group held up by a handful of items.",
    )
    p.add_argument(
        "--max-iterations",
        type=int,
        default=DEFAULT_MAX_ITERATIONS,
        help=f"Cap on Lloyd iterations per colour clustering (default: {DEFAULT_MAX_ITERATIONS}).",
    )
    p.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"Centroid movement below which a colour clustering is called converged "
        f"(default: {DEFAULT_TOLERANCE}).",
    )
    p.add_argument("--json", action="store_true", help="Emit structured JSON.")
    p.set_defaults(func=cmd_map)


# --------------------------------------------------------------------------
# The artifact template
# --------------------------------------------------------------------------
#
# ``string.Template`` rather than an f-string or ``str.format``: CSS and JS are
# full of braces, and ``$name`` placeholders keep the template readable without
# doubling every one of them. Every substituted value is escaped by the builder
# above -- ``_escape`` for markup, ``_json_block`` for data -- so nothing
# derived from corpus text is ever interpolated raw.
#
# Nothing here points outside the file: no <link>, no src=, no href=, no
# @import, no url(), no fetch. Inline <svg> in HTML5 needs no xmlns, so even
# the SVG namespace URI is absent and the document contains no scheme at all
# except the endpoint recorded as provenance text.

_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>$title</title>
<style>
:root {
  color-scheme: light dark;
  --bg: #faf7f0; --panel: #fffdf8; --line: #d9d2c2;
  --ink: #242019; --muted: #6f6858; --grid: #e6e0d0;
  $light
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #0e1613; --panel: #14201c; --line: #2c3b33;
    --ink: #eaf1ec; --muted: #93a89b; --grid: #1e2c26;
    $dark
  }
}
:root[data-theme="light"] {
  --bg: #faf7f0; --panel: #fffdf8; --line: #d9d2c2;
  --ink: #242019; --muted: #6f6858; --grid: #e6e0d0;
  $light
}
:root[data-theme="dark"] {
  --bg: #0e1613; --panel: #14201c; --line: #2c3b33;
  --ink: #eaf1ec; --muted: #93a89b; --grid: #1e2c26;
  $dark
}
* { box-sizing: border-box; }
body {
  margin: 0; padding: 1.5rem; background: var(--bg); color: var(--ink);
  font: 15px/1.55 ui-sans-serif, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}
.wrap { max-width: 1100px; margin: 0 auto; }
h1 { font-size: 1.25rem; margin: 0 0 .2rem; letter-spacing: -.01em; }
h2 { font-size: .8rem; text-transform: uppercase; letter-spacing: .08em;
     color: var(--muted); margin: 0 0 .6rem; }
.sub { color: var(--muted); margin: 0 0 1.2rem; font-size: .9rem; word-break: break-all; }
.panel { background: var(--panel); border: 1px solid var(--line);
         border-radius: 10px; padding: 1rem 1.1rem; margin-bottom: 1rem; }
.cols { display: flex; flex-wrap: wrap; gap: 1rem; align-items: flex-start; }
.cols > .panel { flex: 1 1 320px; margin-bottom: 0; }
.plotwrap { overflow-x: auto; }
svg { display: block; max-width: 100%; height: auto; }
.frame { fill: none; stroke: var(--grid); stroke-width: 1; }
.axis { stroke: var(--grid); stroke-width: 1; stroke-dasharray: 3 4; }
.mark { stroke: var(--panel); stroke-width: 1.25; cursor: pointer; }
.mark:hover, .mark:focus, .mark.pin { stroke: var(--ink); stroke-width: 2; outline: none; }
.cnone { fill: var(--muted); }
.c0 { fill: var(--c0); } .c1 { fill: var(--c1); } .c2 { fill: var(--c2); }
.c3 { fill: var(--c3); } .c4 { fill: var(--c4); } .c5 { fill: var(--c5); }
.c6 { fill: var(--c6); } .c7 { fill: var(--c7); }
dl { margin: 0; }
.row { display: flex; gap: .75rem; padding: .3rem 0; border-top: 1px solid var(--line);
       font-size: .87rem; }
.row:first-child { border-top: 0; }
dt { flex: 0 0 11rem; color: var(--muted); }
dd { margin: 0; flex: 1 1 auto; word-break: break-word; }
.legend { list-style: none; margin: .4rem 0 0; padding: 0; display: flex;
          flex-wrap: wrap; gap: .5rem 1rem; font-size: .87rem; }
.swatch { display: inline-block; width: .8rem; height: .8rem; border-radius: 3px;
          margin-right: .4rem; vertical-align: -1px; }
.swatch.c0 { background: var(--c0); } .swatch.c1 { background: var(--c1); }
.swatch.c2 { background: var(--c2); } .swatch.c3 { background: var(--c3); }
.swatch.c4 { background: var(--c4); } .swatch.c5 { background: var(--c5); }
.swatch.c6 { background: var(--c6); } .swatch.c7 { background: var(--c7); }
.caveat { border-left: 3px solid var(--c0); }
.caveat p { margin: 0; }
.muted { color: var(--muted); font-size: .87rem; }
#inspect { margin-top: .6rem; min-height: 3.2rem; font-size: .87rem;
           white-space: pre-wrap; word-break: break-word; }
#inspect:empty::before { content: "Hover or focus a mark to read its item."; color: var(--muted); }
#theme { float: right; font: inherit; font-size: .8rem; cursor: pointer; color: var(--ink);
         background: var(--panel); border: 1px solid var(--line);
         border-radius: 6px; padding: .2rem .6rem; }
</style>
</head>
<body>
<div class="wrap">
<button id="theme" type="button" hidden>Toggle theme</button>
<h1>$title</h1>
<p class="sub">A projection of $atlas &mdash; read it with the provenance below, not
without it.</p>

<div class="panel plotwrap">
<svg viewBox="0 0 $plot_width $plot_height" role="img"
     aria-label="Two-dimensional PCA projection of the surveyed atlas">
<rect class="frame" x="0.5" y="0.5" width="718" height="458" rx="8"></rect>
<line class="axis" x1="0" y1="230" x2="720" y2="230"></line>
<line class="axis" x1="360" y1="0" x2="360" y2="460"></line>
$marks
</svg>
<div id="inspect"></div>
</div>

<div class="cols">
<div class="panel">
<h2>Provenance</h2>
<dl>
$provenance
</dl>
</div>
<div class="panel">
<h2>Colour</h2>
<p class="muted">$cluster_note</p>
$legend
</div>
</div>

<div class="panel caveat">
<h2>What this picture does not show</h2>
<p>$caveat</p>
</div>
</div>

<script id="map-data" type="application/json">$data</script>
<script>
(function () {
  var el = document.getElementById("map-data");
  if (!el) { return; }
  var data = JSON.parse(el.textContent);
  var inspect = document.getElementById("inspect");
  var pinned = null;
  function show(node) {
    var point = data.points[Number(node.getAttribute("data-i"))];
    if (!point) { return; }
    var group = point.cluster === null ? "uncoloured" : "cluster " + point.cluster;
    inspect.textContent = point.id + "  [" + group + "]  x=" + point.x + " y=" + point.y +
      "\\n" + point.text;
  }
  var marks = document.querySelectorAll(".mark");
  for (var i = 0; i < marks.length; i++) {
    marks[i].addEventListener("mouseenter", function (e) { if (!pinned) { show(e.target); } });
    marks[i].addEventListener("focus", function (e) { if (!pinned) { show(e.target); } });
    marks[i].addEventListener("click", function (e) {
      if (pinned) { pinned.classList.remove("pin"); }
      pinned = pinned === e.target ? null : e.target;
      if (pinned) { pinned.classList.add("pin"); show(pinned); }
    });
  }
  var button = document.getElementById("theme");
  button.hidden = false;
  button.addEventListener("click", function () {
    var root = document.documentElement;
    var dark = window.matchMedia("(prefers-color-scheme: dark)").matches;
    var now = root.getAttribute("data-theme") || (dark ? "dark" : "light");
    root.setAttribute("data-theme", now === "dark" ? "light" : "dark");
  });
})();
</script>
</body>
</html>
"""
