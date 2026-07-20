# Semantic measurement: what a survey verb is allowed to claim

Status: decided. This note settles the three unknowns the survey-surface spec
marks blocking (`gaps` semantics, cross-model `compare` method, projection and
cluster stability) and scopes the determinism promise that an earlier draft of
the spec overstated. It lands ahead of the verbs it constrains, per acceptance
criterion 3 of issue #1.

It is a constraints document, not a roadmap. Every rule here is a limit on what
an output value may assert, and each of the seven planned verbs inherits those
limits whether or not it names them.

## The framing this note enforces

`embeddings-cli` studies embedding spaces **as systems**. The unit of
interpretation is structure — regions, neighborhoods, boundaries, recurring
directions, density — never an individual dimension and never an individual
vector read in isolation. Per-dimension interpretation is unstable across seeds,
training runs, and model versions, and it is misleading precisely because it
reads well; both `CLAUDE.md` and the originating issue mark it an explicit
anti-goal. Nothing in this surface may emit a claim of the form "dimension 412
encodes formality".

The second load-bearing distinction: the **Atlas is an evidence-backed
interpretation _of_ the space, not the space itself.** The vectors are the
space. The Atlas is what a survey concluded about it, under stated parameters,
with stated limitations. That is why `limitations[]` and `uncertain[]` are
required manifest fields rather than optional ones, why the committed subset is
the interpretation while `.npy` vector bulk is regenerable and git-ignored, and
why every structural claim carries the command that reproduces it. A reader who
distrusts a conclusion must be able to re-run it; a reader who trusts it must
still be able to see what it did not establish.

The three decisions below all follow from one posture: the CLI is the
deterministic, conservative half of the Surveyor loop. The agent may hypothesize
freely. The tool may not. When a measurement can support two readings, the tool
reports both or reports neither — it does not pick the interesting one.

## Baseline: the pre-work state

Verified during scoping at commit `be4f6e8`, before any survey-surface code
existed. Recorded here so that later drift is attributable.

| Property | State at `be4f6e8` |
|----------|--------------------|
| CLI verbs | Six introspection verbs only: `whoami`, `learn`, `explain`, `overview`, `doctor`, `cli overview` |
| Corpus / embedding / survey code | None. No module reads a corpus, calls an embedding backend, or computes a measurement |
| Disk writes | None. No module under `embeddings/` writes to the filesystem |
| Runtime dependencies | `dependencies = []` in `pyproject.toml` |
| Survey verbs | `survey`, `regions`, `neighbors`, `directions`, `gaps`, `compare`, `map` exist only as planned-verb listings in `README.md`, `learn.py`, `explain/catalog.py`, and `overview.py` |
| Rubric gate | `teken cli doctor . --strict` green; `doctor` reports `healthy: true` |

The zero-dependency property is inherited from the `culture-agent-template`
scaffold, not enforced by CI — `teken cli doctor --strict` checks CLI shape, not
dependencies. It is incompatible with being a measurement tool and is given up
deliberately, starting with numpy.

## 1. `gaps`: probe-based and dual-labeled

**Decision.** `gaps` emits two distinct finding kinds that are never merged into
a single number, list, or severity ranking:

- **`corpus coverage`** — derived from sparsity in the surveyed corpus. Says
  only that the corpus contains few or no items in some part of the space. Says
  **nothing** about model capability.
- **`model representation`** — derived **only** from actively probing a sparse
  region with generated or held-out inputs, embedded at probe time, and
  measuring where they land relative to the region and to each other. Requires a
  live or cached embedding of inputs that were not in the surveyed corpus.

A `gaps` run without a probe step yields `corpus coverage` findings and nothing
else. The kinds are separate JSON keys, not a `kind` field on a shared list, so
that an agent aggregating results cannot flatten them by accident.

**Why the separation is the highest-stakes rule in this document.** "The model
has a blind spot here" is exactly the claim a reader acts on — it justifies
swapping models, adding training data, or declaring a retrieval failure
understood. Conflating a corpus gap with a model gap manufactures that claim out
of nothing, and it does so in the direction of a more interesting result, which
is the direction errors travel unchallenged.

**Why absence cannot be inferred from silence.** An empty region of the surveyed
space has many sufficient explanations, and sparsity alone does not distinguish
between them:

- Nobody wrote documents about that topic — a fact about the corpus, its
  sampling, its recency, or its domain.
- The corpus contains such documents but they were filtered, truncated, or
  failed to parse — a fact about the ingestion path.
- The projection or density estimate used to call the region "sparse" put them
  elsewhere — a fact about the measurement parameters.
- The model genuinely collapses those inputs into a neighboring region, or
  scatters them without structure — the only explanation that is about the
  model.

Only the last is a model finding, and only a probe can isolate it: embed inputs
that _should_ land in the region and observe what the model actually does with
them. If they land coherently and separably, the region was a corpus artifact
and the model represents it fine. If they scatter, collapse onto an unrelated
neighborhood, or land at distances indistinguishable from noise, that is
evidence about the representation — and it is evidence, not proof, because it is
still conditioned on the probe inputs being fair instances of the concept.

**Consequences.** Probe inputs are recorded in the Atlas alongside the finding,
because a `model representation` claim is only as good as the probes behind it
and a reader must be able to judge them. A probe finding also carries the
distinguishing measurement it rests on (where probes landed, spread, distance to
the nearest populated region) rather than a bare verdict. And because probing
needs embeddings computed at probe time, `gaps` is the one reader verb that may
touch the backend; with the gateway down it still returns `corpus coverage`
findings and reports the probe step as unavailable rather than failing the whole
verb.

## 2. `compare`: relational only, no shared coordinate frame

**Decision.** `compare` v1 measures **neighborhood agreement on a shared probe
set embedded by both models**: k-NN set overlap (Jaccard) at stated `k`, plus
rank correlation (Spearman) over the ordered neighbor lists. No output value may
assume, imply, or be derived from a shared coordinate frame. CKA/RSA and
Procrustes alignment with a reported residual are named follow-ups; they are not
shipped in v1.

**Why raw coordinates are not comparable.** Two embedding models share no basis.
They differ in dimensionality, in scale, in the arbitrary rotation and
reflection their training landed on, and in the shape of the distance
distribution they induce. Coordinates from model A and model B are numbers in
different units of different things. Worse, cosine similarity — which _looks_
model-independent because it is normalized and bounded — is not on a common
ruler either: models differ systematically in how tightly they pack unrelated
text, so `0.82` from one model and `0.82` from another are not the same
statement about similarity. A "distance changed by 0.1 between models" output
would be arithmetic on incommensurable quantities, presented with the full
authority of a number.

**Why relational measures survive this.** Ranks and set membership are invariant
under exactly the transformations that differ between models — any monotone
rescaling of distance, any rotation, any reflection. "Which items are this
item's ten nearest neighbors" is a question each model answers in its own frame,
and the answers are directly comparable because they are drawn from the same
shared item set. That comparability is the whole basis of v1, and it is why the
shared probe set is not a convenience but a requirement.

**Stated assumptions.** These hold or the numbers mean less than they appear to,
so each is recorded in the comparison output:

- Both models embed **the same probe items**, identified identically. The metric
  is defined over a common index; without it there is nothing to intersect.
- The probe set is **representative of the region being compared**. Jaccard over
  an unrepresentative sample measures the sample.
- The probe set is **large enough that overlap at `k` is not dominated by
  chance**. Expected random overlap scales with `k/N`; small `N` with large `k`
  yields high agreement that means nothing, so the chance baseline is reported
  next to the observed value.
- Results are **conditioned on `k`**. Agreement typically falls as `k` grows;
  a single `k` is a slice, not a summary, and `k` is stamped on every output.
- Each model's own **preprocessing is applied as that model intends** —
  normalization, prompt or instruction prefixes, truncation. Comparing model A
  used correctly against model B used incorrectly measures the harness.

**What `compare` therefore says and does not say.** It says: these two models
agree (or disagree) about what is near what, at this `k`, on this probe set. It
does not say either model is better, more accurate, or more semantically
correct — agreement is symmetric and carries no ground truth. Where they
disagree, that is a located, re-runnable finding an operator can inspect, which
is the point.

## 3. Stability: params, seed, and N-seed reruns

**Decision.** Every measurement output records its full parameters and its seed.
Every structural claim — a region, a boundary, a direction, a gap — carries a
stability measure: agreement across **N seed reruns** plus **bootstrap
resampling** of the input items, reported with the measure used (Adjusted Rand
Index for clusterings; neighborhood or rank agreement for direction and
neighborhood claims). A result produced by a single run with no rerun evidence
is labeled **`stability unverified`** and must be rendered as such in both text
and `--json` output. It is not silently promoted to a finding.

**The principle.** _A region that only exists at one parameter setting is a
finding about the parameters, not about the model._ The same holds for a
direction that only appears at one seed and a boundary that dissolves under
resampling.

**Why this is not optional caution.** UMAP and t-SNE layouts, and k-means and
HDBSCAN partitions, reliably produce structures that _look_ meaningful:
well-separated blobs, clean corridors, apparent boundaries. They produce them
from noise as readily as from signal, and the specific structure shifts
substantially with `n_neighbors`, `min_dist`, perplexity, `k`, `min_cluster_size`
— and with the seed alone, parameters held fixed. A projection is a lossy
low-dimensional summary chosen to satisfy a layout objective; inter-cluster
distances in it are not faithful, and cluster count is a parameter as often as a
discovery. An unqualified "the space has seven regions" is a statement about a
`k` someone chose.

This bites hardest here because of what the CLI is for. The Surveyor agent's
value is generating hypotheses; the CLI's value is being the half of that loop
that does not hallucinate. If the tool reports a one-run artifact as a measured
fact, the agent builds an Atlas on it, cites it as evidence, and the split
between hypothesis and measurement — the reason to have a deterministic tool at
all — is gone. Stability labeling is the mechanism that keeps the tool's output
worth more than the agent's guess.

**Consequences.** `params` and `seed` are required keys on every structural
verb's `--json` payload. Stability reruns cost N times the clustering work but
no re-embedding (vectors come from the Atlas or the cache), so the default N can
be meaningful rather than token. Low-stability structures are still reported —
suppressing them would hide real ambiguity in the space, which is itself a
finding — but they are reported with their instability attached, and an unstable
region belongs in the manifest's `uncertain[]` section.

## 4. Determinism scope

An earlier draft of the spec promised that identical inputs and an identical
seed always produce identical output. That is unsatisfiable and is now scoped.

**Guaranteed byte-identical:**

- **Reader verbs over a fixed Atlas.** `regions`, `neighbors`, `directions`,
  `gaps` (its `corpus coverage` half), `compare`, and `map`, run against an
  unchanged Atlas with the same parameters and seed, produce byte-identical
  output. Readers never need the gateway; this is the guarantee tests assert.
- **Cache-served vectors.** A vector retrieved from the embed cache for a given
  `(model, text-hash)` is the same bytes every time. Any survey whose inputs are
  fully cache-served is reproducible end to end.

**Not guaranteed:**

- **Live re-embedding.** GPU inference is not bit-deterministic: kernel and
  algorithm selection, reduction order in parallel accumulation, batch-size- and
  load-dependent code paths, and vLLM's own batching and scheduling all mean the
  same text embedded twice may differ at float precision. The differences are
  tiny, but they are not zero, and downstream clustering can amplify a tiny
  input perturbation into a different partition near a decision boundary.

**The rule.** An Atlas written from live (non-cached) embeddings **must record
this in its `limitations[]`**. That entry is written automatically by `survey`,
not left to the operator, because the operator most likely to omit it is the one
who did not know it applied. An Atlas built entirely from cache hits does not
carry the entry.

This is why survey-level reproducibility is described as cache-scoped rather
than absolute, and why the honest reproducibility claim is layered: the Atlas is
the reproducibility boundary. Fix the Atlas and everything downstream is exact.
Cross that boundary back to the model and reproducibility becomes approximate —
which is precisely the kind of limit this tool is supposed to state out loud
rather than round away.

## What this constrains, in one list

- No claim about model capability without a probe (§1).
- No cross-model number that assumes a shared coordinate frame (§2).
- No structural claim without `params`, `seed`, and either a stability measure
  or the `stability unverified` label (§3).
- No reproducibility promise that extends past the Atlas boundary into live
  re-embedding (§4).
- No interpretation of an individual dimension, ever.
