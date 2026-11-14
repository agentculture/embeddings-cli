"""Markdown catalog for ``embeddings explain <path>``.

Each entry is verbatim markdown. Keys are command-path tuples. The empty tuple,
``("embeddings",)``, and ``("embeddings-cli",)`` all resolve to the root entry
(the project is named ``embeddings-cli``; the installed command is
``embeddings``).

Keep bodies self-contained: an agent reading one entry should get enough
context without chaining reads.
"""

from __future__ import annotations

_ROOT = """\
# embeddings-cli

A tool for studying embedding spaces as **systems**, not just for querying
individual vectors. An individual embedding is hard to interpret — its
dimensions rarely correspond to clean human concepts. An embedding space,
however, has structure worth measuring: semantic regions and clusters,
neighborhoods and boundaries, recurring directions of meaning, dense and sparse
areas, gaps in what the model represents, and differences between models,
datasets, domains, and versions.

The project treats an embedding model as a **semantic landscape**. Rather than
only asking "which documents are closest to this query?", it asks "what kind of
world does this embedding model create?" — what it places together, what it
keeps apart, which distinctions it preserves, which it collapses, and where the
representation becomes weak or uncertain.

It also serves as the deterministic tool surface for a **Surveyor agent**, which
runs measured experiments against the space (sampling concepts, finding
clusters, proposing and testing region names, discovering and validating
semantic directions) and records an **Atlas** — an evidence-backed
interpretation of the space, not the space itself.

The installed command is `embeddings`.

## Verbs

- `embeddings whoami` — identity probe from `culture.yaml`.
- `embeddings learn` — structured self-teaching prompt.
- `embeddings explain <path>` — markdown docs for any noun/verb.
- `embeddings overview` — descriptive snapshot of the agent.
- `embeddings doctor` — check the agent-identity invariants.
- `embeddings cli overview` — describe the CLI surface.
- `embeddings survey <corpus>` — embed a corpus and write the atlas other verbs read.
- `embeddings regions` — cluster a surveyed atlas into semantic regions, with stability.
- `embeddings neighbors <text|--id id>` — k nearest atlas items by cosine distance.
- `embeddings directions discover` — discover candidate semantic directions and test them.
- `embeddings compare` — compare two atlases' k-NN and rank agreement on a shared probe set.
- `embeddings gaps` — find corpus-coverage gaps, and (with `--probe`) model-representation ones.
- `embeddings map` — PCA-project an atlas to 2-D, optionally writing an HTML artifact.
- `embeddings export` — write an atlas and its vectors as an embeddings-tumbler viewer data file.

## Exit-code policy

- `0` success
- `1` user-input error
- `2` environment / setup error
- `3+` reserved

## See also

- `embeddings explain whoami`
- `embeddings explain doctor`
- `embeddings explain survey`
"""

_WHOAMI = """\
# embeddings whoami

Reports the agent's identity from `culture.yaml`: nick (`suffix`), backend,
served model, and the package version. Read-only.

## Usage

    embeddings whoami
    embeddings whoami --json
"""

_LEARN = """\
# embeddings learn

Prints a structured self-teaching prompt covering purpose, command map,
exit-code policy, `--json` support, and the `explain` pointer. Every verb it
lists as a command is a verb the parser actually registers; nothing on this
CLI's roadmap remains unimplemented today, so there is no separate planned
list to keep apart from it.

## Usage

    embeddings learn
    embeddings learn --json
"""

_EXPLAIN = """\
# embeddings explain <path>

Prints markdown documentation for any noun/verb path. Unlike `--help` (terse,
positional), `explain` is global and addressable by path.

## Usage

    embeddings explain embeddings-cli
    embeddings explain whoami
    embeddings explain --json <path>
"""

_OVERVIEW = """\
# embeddings overview

Read-only descriptive snapshot of the agent: identity (from `culture.yaml`), the
verb surface, and the artifacts this repo carries. Accepts an ignored `target`
so a stray path never hard-fails.

## Usage

    embeddings overview
    embeddings overview --json
"""

_DOCTOR = """\
# embeddings doctor

Checks the agent-identity invariants `steward doctor` verifies:
prompt-file-present and backend-consistency (`colleague` → `AGENTS.colleague.md`), plus a
skills-present check. Exits 1 when unhealthy.

## Usage

    embeddings doctor
    embeddings doctor --json
"""

_CLI = """\
# embeddings cli

Noun group for CLI-surface introspection. `cli overview` describes the CLI
itself (distinct from the global `overview`, which describes the agent).

## Usage

    embeddings cli overview
    embeddings cli overview --json
"""

_SURVEY = """\
# embeddings survey <corpus>

Embeds a JSONL corpus through the embedding backend and writes the Atlas
every other reader verb (`regions`, `neighbors`, `directions`, `gaps`,
`compare`, `map`) reads. A write verb: previews by default and performs
**zero filesystem writes and zero backend calls** until `--apply` is given.
`--apply` onto an existing atlas is refused unless `--force` is also given,
which replaces the atlas whole rather than merging into it.

## Usage

    embeddings survey corpus.jsonl
    embeddings survey corpus.jsonl --apply
    embeddings survey corpus.jsonl --apply --force
    embeddings survey corpus.jsonl --json

## Key flags

- `--apply` — actually embed and write the atlas (default: preview only).
- `--force` — with `--apply`, replace an existing atlas at the target path.
- `--atlas <path>` / `-o` — directory to write the atlas into (default: `./atlas`).
- `--batch-size <n>` — texts per backend request (default: 64); each batch
  reports progress to stderr.
- `--retries <n>` — retries per batch on a transient (exit-2) backend
  failure (default: 2).
- `--seed <n>` — RNG seed recorded in the atlas manifest (default: 0).
- `--embed-url` / `--embed-model` — override the embedding backend endpoint
  and model.

Corpus format: one JSON object per line, each with a `text` field (and an
optional `id`; every other key is preserved as record metadata).
"""

_REGIONS = """\
# embeddings regions

Partitions a surveyed atlas into semantic regions with k-means, and reports
each region's membership, representative items, and stability across seed
reruns and bootstrap resamples. A region that survives only one parameter
setting is reported with `asserted: false` and a `finding` string — never
suppressed, since an unstable region is itself a finding.

## Usage

    embeddings regions
    embeddings regions --clusters 8 --seed 1
    embeddings regions --atlas path/to/atlas --json

## Key flags

- `--clusters <n>` / `-k` — number of regions to partition into (default: 5).
- `--seed <n>` — base RNG seed; every rerun/resample seed derives from it
  (default: 0).
- `--runs <n>` — seed reruns for stability evidence; `--runs 1` with
  `--resamples 0` reports "stability unverified" rather than a score.
- `--resamples <n>` — bootstrap resamples of the item set.
- `--representatives <n>` — items reported per region, nearest-centroid
  first (default: 3).
- `--atlas <path>` — atlas directory to read (default: `./atlas`).

Requires an atlas written by `embeddings survey <corpus> --apply`. Reads only
the atlas on disk; opens no socket.
"""

_NEIGHBORS = """\
# embeddings neighbors <text|--id id>

Finds the k nearest atlas items to a text query (embedded through the
backend, then cached) or to an existing atlas item id (looked up on disk, no
backend call at all), by cosine distance. Also reports a boundary
description: the sorted distance-gap profile across the returned neighbors.

## Usage

    embeddings neighbors "agentic culture"
    embeddings neighbors some-atlas-id --id
    embeddings neighbors "query text" --top-k 10 --json

## Key flags

- `--id` — treat the positional argument as an existing atlas item id
  rather than free text; this path never calls the embedding backend.
- `-k` / `--top-k <n>` — neighbors to return (default: 5).
- `--atlas <path>` — atlas directory to read (default: `./atlas`).
- `--embed-url` / `--embed-model` — override the embedding backend; the
  model defaults to the atlas's own model.

The positional argument is always free text unless `--id` is passed — never
guessed from its shape.
"""

_DIRECTIONS = """\
# embeddings directions

Proposes candidate semantic directions from paired items, and tests whether
each one generalizes to pairs it was not derived from. A direction is a
vector in the embedding space — never a coordinate index, and never given a
human-readable label by this tool.

## Sub-verbs

- `embeddings directions discover` — derive candidates from training pairs,
  test them on held-out pairs, and measure their stability.
- `embeddings directions overview` — this description (also printed by a
  bare `embeddings directions`).

## Usage

    embeddings directions overview
    embeddings directions discover --pairs pairs.jsonl
    embeddings directions discover --sample-pairs 64 --seed 0 --json

Candidates that fail to generalize are still listed, marked
`DOES NOT GENERALIZE`, and routed into `uncertain[]` — never dropped, since a
pair set with no transferable direction is itself a finding.
"""

_DIRECTIONS_DISCOVER = """\
# embeddings directions discover

Derives candidate semantic directions from the training half of a pair set
(the leading singular vectors of the training difference vectors), then
measures each candidate against the held-out half it never saw. Reports the
generalization result either way, plus a stability measurement across seed
reruns and bootstrap resamples of the item bipartition each direction
induces.

## Usage

    embeddings directions discover --pairs pairs.jsonl
    embeddings directions discover --sample-pairs 64 --seed 0
    embeddings directions discover --pairs pairs.jsonl --top 5 --json

## Key flags

- `--pairs <path>` — file of `(a, b)` atlas item-id pairs, one per line
  (a JSON object `{"a": "<id>", "b": "<id>"}` or a bare `<id>,<id>` row).
  Omit to sample pairs from the atlas instead.
- `--sample-pairs <n>` — random pairs to sample when `--pairs` is omitted
  (default: 64), seeded by `--seed`.
- `--holdout <fraction>` — fraction of the pair set held out to test
  generalization (default: 0.3); at least one pair always stays on each
  side of the split.
- `--top <n>` — how many candidate directions to propose (default: 3).
- `--seed <n>` — base seed for the split, the sampling, and every stability
  rerun (default: 0).
- `--runs <n>` / `--resamples <n>` — stability-evidence controls.
- `--include-vectors` — include each candidate's full vector in the output
  (off by default; it dominates the payload at realistic dimensionality).

Pairs name atlas item ids, not texts, so this verb never needs the
embedding backend.
"""

_COMPARE = """\
# embeddings compare

Compares two already-surveyed atlases relationally: k-NN set overlap
(Jaccard) and Spearman rank correlation on each model's own ordering of a
shared probe set, each reported against a chance baseline. Never assumes,
implies, or derives a value from a shared coordinate frame — no coordinate,
distance, or cosine crosses between the two models, and neither model is
ranked as better than the other.

## Usage

    embeddings compare --atlas-a atlas-model-a --atlas-b atlas-model-b
    embeddings compare --atlas-a a --atlas-b b --top-k 5 --json

## Key flags

- `--atlas-a <path>` / `--atlas-b <path>` — the two atlas directories to
  compare (both required).
- `-k` / `--top-k <n>` — neighborhood size the Jaccard measure is
  conditioned on (default: 10); the Spearman correlation runs over each
  model's complete ordering and is not conditioned on k.
- `--seed <n>` — recorded for provenance; the comparison itself is fully
  deterministic and consumes no randomness.

The two atlases must share a probe set: the same record ids, carrying
identical text. Reads only the two atlases on disk; opens no socket.
"""

_GAPS = """\
# embeddings gaps

Finds under-covered areas of a surveyed atlas. Reports two kinds of finding
as separate top-level keys, never merged: `corpus_coverage_findings` (sparse
regions and inter-region voids — says nothing about model capability) and
`model_representation_findings` (produced only by an active `--probe` step
that embeds new inputs at probe time).

## Usage

    embeddings gaps
    embeddings gaps --probe --probes probes.txt
    embeddings gaps --probe --holdout 10 --json

## Key flags

- `--clusters <n>` / `-k` — regions to partition into before measuring
  sparsity (default: 5).
- `--sparsity-quantile <f>` — share of regions, most isolated first,
  reported as sparse (default: 0.25).
- `--void-factor <f>` — multiple of the corpus's own median
  nearest-neighbor distance above which an inter-region midpoint counts as
  a void (default: 2.0).
- `--probe` — enable the probe step (**off by default**; this is the only
  path in this verb that reaches the embedding backend).
- `--probes <file>` — probe texts, one per line; requires `--probe`.
- `--holdout <n>` — hold out n atlas items and re-embed their text as
  probes; requires `--probe`.
- `--coherence-factor <f>` — fraction of the corpus-wide baseline spread
  below which probes count as cohering rather than scattering.

A sparse corpus region is not, on its own, evidence the model cannot
represent that area — only an active probe can test that.
"""

_MAP = """\
# embeddings map

Projects a surveyed atlas onto its two leading principal components (PCA,
deterministic, computed on numpy alone) and reports how much of the total
variance that 2-D view keeps. With `--html`, writes a single self-contained
page carrying the plot, its full provenance, and the parameter-dependence
caveat. A write verb: previews by default, `--apply` commits, `--force`
replaces an existing file.

## Usage

    embeddings map
    embeddings map --html atlas-map.html
    embeddings map --html atlas-map.html --apply
    embeddings map --clusters 0 --json

## Key flags

- `--html <path>` — write a self-contained single-file HTML artifact (no
  external requests); preview-only without `--apply`.
- `--apply` — actually write the `--html` artifact.
- `--force` — with `--apply`, overwrite an existing file at `--html`.
- `--clusters <n>` / `-k` — colour the map by a k-means partition into this
  many groups (default: 5); `0` draws it uncoloured. The colouring always
  ships with its own stability verdict.
- `--seed <n>` — base seed for the colour clustering (the PCA projection
  itself is deterministic and ignores it).

Distances in the picture are not faithful to distances in the space —
especially between apparent clusters — so the caveat travels inside the
artifact itself, not just in this doc.
"""


_EXPORT = """\
# embeddings export

Writes a surveyed atlas and its vectors as an **embeddings-tumbler** data set
(the viewer's data contract, schema version 1): `<name>.json` holding the
model, dimension, count, provenance, reduction record and one record per point,
plus `<name>.f32` beside it — every vector as little-endian float32, row-major,
exactly points x dimension x 4 bytes. Needs the atlas's `vectors.npy`; a
freshly-cloned study has none, so run `embeddings survey` first. Never
contacts the gateway. A write verb: previews by default, `--apply` commits,
`--force` replaces an existing `.json` or `.f32`.

## Usage

    embeddings export --out view.json
    embeddings export --out view.json --apply
    embeddings export --out view.json --labels-only --apply
    embeddings export --out view.json --group-by source --apply
    embeddings export --out view.json --top-k 64 --endpoint-name "lobes gateway" --apply

## Key flags

- `--out <path>` — the `.json` to write (default `viewer-data.json`); the block
  goes beside it as `<name>.f32`.
- `--labels-only` — labels are the record ids, no metadata is written, and the
  file states `text_included: false`. No corpus text reaches either file.
- `--group-by <key>` — write each point's `group` from that metadata key (a
  non-empty string or an integer); the viewer colours and names points by
  group. Records without the key stay ungrouped, a key no record carries is
  refused, and it cannot be combined with `--labels-only`. Without it no group
  is written.
- `--rotate` — exact orthogonal rotation of the mean-centred vectors onto their
  principal directions; every pairwise distance is kept.
- `--top-k <k>` — implies the rotation: it rotates exactly as `--rotate` does,
  then keeps only the first k directions (1 <= k < the atlas's dimension); the
  file records `source_dimension`, `directions_kept` and `variance_dropped`.
  Off unless given.
- `--endpoint-name <name>` — a *name* for the embedding route, written as
  `provenance.endpoint`: 1 to 64 ASCII letters, digits, spaces, `_` or `-`,
  starting with a letter or digit (no dots, colons or slashes, so no hostname,
  URL or port). The atlas's endpoint URL is never copied; any other value is
  refused, and without the flag the field is omitted.

Provenance copies the atlas's model, seed, `limitations`, `uncertain` and
`params`; any occurrence of the endpoint URL inside them is replaced with
`<embedding endpoint>`. `provenance.source` is `atlas: <folder name>`, never a
local path. Both files get mode 0644 reduced by the umask (never group or other
write), so a static web server can read them.

Writing: an existing `.json` is removed first (under `--force`), then the
`.f32` is written atomically, then the `.json` naming it. A failure at any step
leaves the full new pair or a `.f32` with no `.json` (exit 2) — never a `.json`
beside a block that is not its own.
"""


ENTRIES: dict[tuple[str, ...], str] = {
    (): _ROOT,
    ("embeddings-cli",): _ROOT,
    ("embeddings",): _ROOT,
    ("whoami",): _WHOAMI,
    ("learn",): _LEARN,
    ("explain",): _EXPLAIN,
    ("overview",): _OVERVIEW,
    ("doctor",): _DOCTOR,
    ("cli",): _CLI,
    ("cli", "overview"): _CLI,
    ("survey",): _SURVEY,
    ("regions",): _REGIONS,
    ("neighbors",): _NEIGHBORS,
    ("directions",): _DIRECTIONS,
    ("directions", "overview"): _DIRECTIONS,
    ("directions", "discover"): _DIRECTIONS_DISCOVER,
    ("compare",): _COMPARE,
    ("gaps",): _GAPS,
    ("map",): _MAP,
    ("export",): _EXPORT,
}
