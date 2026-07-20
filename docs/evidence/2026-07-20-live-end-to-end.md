# Live end-to-end run: all seven survey verbs against the lobes gateway

Date: 2026-07-20. Branch: `agent/t22`. Plan task: t22.

This is **operator-machine evidence**, not a CI assertion. Everything below was
run by hand on one workstation against a live embedding gateway. No test in
this repository requires a gateway, and this run adds none — CI stays hermetic.
The point of the document is to record what actually happened, including the
things that did not work.

## Summary

| Claim under test | Result |
|---|---|
| All seven verbs callable, producing measured output from a real corpus | Yes — all seven ran against a 152-item corpus surveyed live |
| Atlas round-trips: a reader reproduces what `survey` wrote | Yes — independent recomputation from the on-disk files matches to float32 epsilon |
| Reader determinism over a fixed atlas | Yes — five reader verbs byte-identical across repeated runs |
| Live survey records the float-variance limitation; cached survey does not | Yes — confirmed at 152-item scale, with byte-identical vectors |
| Every structural output is parameter-and-seed-stamped | Yes — `params` and `seed` on every payload |
| Every structural output supports `--json` | Yes |
| Every structural output cites the command that reproduces it | **No** — see finding F2 |
| Genuine cross-model `compare` | **Not demonstrable on this machine** — only one embedding model is served |

Two findings for the merging agent are recorded in [Findings](#findings). Neither
was fixed here, per the task's file boundary.

## Environment

Resolved with the authoritative resolver rather than assumed:

```console
$ lobes endpoint embedder --json
{"role": "embedder", "endpoint": "http://localhost:8001"}
```

| Property | Value |
|---|---|
| Endpoint | `http://localhost:8001/v1` (the CLI's built-in default; no flag passed) |
| Model | `Qwen/Qwen3-Embedding-0.6B` |
| Dimensions | 1024, `float32` on disk |
| Auth | Bearer token read from the environment by the client. `EMBEDDINGS_API_KEY` was exported from `COLLEAGUE_API_KEY` for the run. The value appears nowhere in this document, in any command shown, or in any captured output. |
| Embed cache | Pointed at a scratch directory via `EMBEDDINGS_CACHE_DIR` so the cold-cache and warm-cache runs below are honestly cold and warm, and the operator's real cache at `~/.embeddings/cache` is untouched. |

The gateway serves five models, but only one of them is an embedding model:

```console
$ curl -s -H "Authorization: Bearer $EMBEDDINGS_API_KEY" http://localhost:8001/v1/models
['sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP', 'Qwen/Qwen3-Embedding-0.6B',
 'Qwen/Qwen3-Reranker-0.6B', 'coolthor/gemma-4-12B-it-NVFP4A16',
 'nvidia/Gemma-4-31B-IT-NVFP4']
```

This single fact is what limits `compare` below.

## The corpus

Built from this repository's own committed prose by
[`build_corpus.py`](build_corpus.py), committed alongside this document so the
run reproduces:

```console
$ uv run python docs/evidence/build_corpus.py docs/evidence/corpus.jsonl
wrote 152 records to docs/evidence/corpus.jsonl
```

| Records | Source |
|---:|---|
| 7 | `README.md` |
| 22 | `CLAUDE.md` |
| 11 | `AGENTS.colleague.md` |
| 22 | `CHANGELOG.md` |
| 27 | `docs/design-notes/semantic-measurement.md` |
| 11 | `docs/skill-sources.md` |
| 9 | `docs/specs/2026-07-20-embeddings-survey-surface.md` |
| 25 | `docs/plans/2026-07-20-embeddings-survey-surface.md` |
| 18 | `LICENSE` |
| **152** | total, median 448 characters |

The builder splits each file into paragraphs, drops fenced code, tables and
rules, strips inline markdown, and keeps paragraphs of 180–1600 characters. The
source path and the nearest preceding heading ride along as record metadata.

Two choices worth stating, because they shape every measurement below:

- **The Apache-2.0 licence text is in the corpus deliberately.** It is legal
  boilerplate in a corpus that is otherwise engineering prose about one project.
  It is the one block with a genuinely different register, so it is the control:
  a clustering that cannot separate it is not finding much.
- **Everything else is topically close.** Eight of the nine sources are prose
  about this same CLI, written in the same voice, frequently restating each
  other — the plan restates the spec, which restates the design note. That is a
  hard corpus to partition, and the results reflect it honestly rather than
  flatteringly.

## 1. `survey`

### Dry run writes nothing

```console
$ uv run embeddings survey docs/evidence/corpus.jsonl
# survey (dry run — nothing written)

corpus:        docs/evidence/corpus.jsonl
items:         152 (152 unique texts)
model:         Qwen/Qwen3-Embedding-0.6B
endpoint:      http://localhost:8001/v1
atlas target:  atlas
cache:         0 hits, 152 to embed
batch size:    64
seed:          0

re-run with --apply to embed the corpus and write the atlas.
```

Exit 0, and no `atlas` directory existed afterwards. Demonstrates the
preview-by-default contract: no filesystem write and no backend call.

### `--apply` against a cold cache

```console
$ uv run embeddings survey docs/evidence/corpus.jsonl --apply
# survey (applied)

atlas:         atlas
items:         152 (152 unique texts)
model:         Qwen/Qwen3-Embedding-0.6B
endpoint:      http://localhost:8001/v1
vectors:       152 x 1024
cache:         0 hits, 152 embedded live
seed:          0

limitations recorded in the manifest:
- live-embedding float variance: some vectors in this atlas were embedded live
  rather than served from the embed cache. GPU inference is not bit-deterministic
  [...] This atlas is the reproducibility boundary: everything derived from it is
  exact, everything upstream of it is approximate.
```

Progress went to stderr, results to stdout — never mixed:

```text
survey: embedded 64/152 texts (batch 1/3)
survey: embedded 128/152 texts (batch 2/3)
survey: embedded 152/152 texts (batch 3/3)
survey: wrote atlas to atlas
```

152 real 1024-dimensional vectors in 0.9 s wall clock. On disk:
`atlas/atlas.json`, `atlas/records.jsonl`, `atlas/vectors.npy`.

Demonstrates: the live path works end to end, and the float-variance limitation
is written by the tool rather than left to the operator.

### `--force` refusal on an existing atlas

```console
$ uv run embeddings survey docs/evidence/corpus.jsonl --apply
error: an atlas already exists at atlas
hint: re-run with --force to replace it whole (the existing atlas is deleted, never merged), or pass a different --atlas path
$ echo $?
1
```

And the same failure under `--json`, on stderr, with no traceback:

```json
{"code": 1, "message": "an atlas already exists at atlas",
 "remediation": "re-run with --force to replace it whole (the existing atlas is deleted, never merged), or pass a different --atlas path"}
```

Demonstrates: exit-code policy (1 = user error), the `hint:` contract, and that
an existing atlas is never silently merged or clobbered.

### Cache behaviour and the limitations claim

Re-surveying the same corpus into a second atlas, now with a warm cache:

```console
$ uv run embeddings survey docs/evidence/corpus.jsonl --apply --atlas atlas-cached
# survey (applied)
...
cache:         152 hits, 0 embedded live
seed:          0
```

No `limitations` block printed. Comparing the two atlases:

```console
$ cmp atlas/vectors.npy atlas-cached/vectors.npy && echo IDENTICAL
IDENTICAL
$ cmp atlas/records.jsonl atlas-cached/records.jsonl && echo IDENTICAL
IDENTICAL
```

| Atlas | `cache_hits` | `embedded_live` | `limitations` entries |
|---|---:|---:|---:|
| `atlas` (cold) | 0 | 152 | 1 |
| `atlas-cached` (warm) | 152 | 0 | 0 |

**This confirms at 152-item scale what had only been checked on a 3-item
corpus.** A live survey records the float-variance limitation; a cache-served
re-survey records none and reproduces the vectors byte for byte.

### The float variance is real, and it moved a verdict

An earlier live survey in this same session — same texts, same model, same
endpoint, embedded live a second time — produced vectors that differed at float
precision. Downstream, the `regions` verdicts shifted:

| Measurement | First live survey | Second live survey |
|---|---:|---:|
| Partition ARI | 0.401 | 0.402 |
| Region 1 max Jaccard | 0.433 | 0.434 |
| Region 2 max Jaccard | 0.384 | 0.385 |
| Region 3 verdict | `unstable` (0.475) | `moderate` |

Region 3 crossed a verdict boundary on nothing but GPU float noise. This is not
a defect — it is precisely the amplification
`docs/design-notes/semantic-measurement.md` §4 predicts, observed in the wild,
and it is the reason the limitation entry exists. It also shows why the Atlas,
not the model, is the reproducibility boundary.

## 2. `regions`

```console
$ uv run embeddings regions
# regions

atlas:         atlas
model:         Qwen/Qwen3-Embedding-0.6B
items:         152 x 1024
algorithm:     kmeans
params:        k=5, runs=5, resamples=20, max_iterations=100, tolerance=1e-06
seed:          0
partition:     unstable

region 0 — 40 items
  representatives: skill-sources-001, changelog-017, changelog-022
  stability:       moderate

region 1 — 37 items
  representatives: spec-001, plan-001, agents-colleague-008
  stability:       unstable
  finding:         region 1 is not asserted as structure in the space: unstable
                   (max_jaccard_overlap 0.434) [...] At these parameters
                   (algorithm=kmeans, k=5, seed=0) it is a finding about the
                   parameters, not about the model.

region 2 — 34 items    stability: unstable (max_jaccard_overlap 0.385)
region 3 — 23 items    stability: moderate
region 4 — 18 items    stability: stable
  representatives: license-014, license-006, license-010

uncertain (belongs in the atlas manifest's uncertain[]):
- region 1: unstable — max_jaccard_overlap 0.434 [...]
- region 2: unstable — max_jaccard_overlap 0.385 [...]
- the 5-region partition itself: unstable — adjusted_rand_index 0.402 [...]
```

**Is the stability verdict honest about what it found?** Yes, and this is the
strongest single result in the run. The verb declined to assert the partition it
had just computed, and the one region it *did* call stable is the one the corpus
was built to contain.

Sweeping `k` confirms the verdict is responsive rather than boilerplate:

| `k` | Partition verdict | ARI | Stable regions |
|---:|---|---:|---:|
| 2 | unstable | 0.317 | 0 / 2 |
| 3 | unstable | 0.212 | 0 / 3 |
| 5 | unstable | 0.402 | 1 / 5 |
| 7 | unstable | 0.369 | 1 / 7 |
| 9 | unstable | 0.346 | 1 / 9 |
| 12 | unstable | 0.358 | 0 / 12 |

No `k` yields a stable partition. And every stable region found, at every `k`,
is pure licence text:

| `k` | Stable region size | Composition |
|---:|---:|---|
| 5 | 18 | 100% `LICENSE` |
| 7 | 16 | 100% `LICENSE` |
| 9 | 16 | 100% `LICENSE` |

The honest reading: on this corpus, the only large-scale structure this model
reproduces across seeds and resamples is the legal-register block. The
engineering prose — eight sources restating one project — does not partition
reproducibly at any `k` tried. The tool reports that instead of picking the
interesting result, which is what §3 of the design note demands.

## 3. `neighbors`

### Free text (requires the gateway)

```console
$ uv run embeddings neighbors "how stable is a k-means partition across seeds" -k 5
# neighbors (metric: cosine)

query:    'how stable is a k-means partition across seeds' (source: text)
atlas:    atlas
k:        5 (returned 5)
seed:     0

 1. plan-015          distance=0.268329  text='...regions clusters the Atlas into semantic regions and emits cluster membership, size, and representative items - output carries params and seed and a stability measure per region...'
 2. design-note-018   distance=0.283012  text='Decision. Every measurement output records its full parameters and its seed. Every structural claim — a region, a boundary, a direction, a gap — carries a stability measure...'
 3. design-note-013   distance=0.292794  text='Decision. compare v1 measures neighborhood agreement on a shared probe set...'
 4. plan-019          distance=0.302570  text='...compare reports k-NN set overlap (Jaccard) and rank correlation on a shared probe set...'
 5. design-note-020   distance=0.330010  text='Why this is not optional caution. UMAP and t-SNE layouts, and k-means and HDBSCAN partitions, reliably produce structures that _look_ meaningful...'

boundary: largest distance jump (gap=0.027440) is between neighbor 4 and neighbor 5
```

A second query in a different register:

```console
$ uv run embeddings neighbors "warranty disclaimer and limitation of liability" -k 3
 1. license-014  distance=0.410045  text='Disclaimer of Warranty. Unless required by applicable law...'
 2. license-018  distance=0.443167  text='Unless required by applicable law or agreed to in writing, software distributed under the License is distributed on an "AS IS" BASIS...'
 3. claude-md-018 distance=0.457684  text='doctor checks the same invariants steward doctor verifies...'
```

Demonstrates: the gateway path works and the results are semantically sensible —
a query about clustering stability retrieves the stability design note, a query
about warranty language retrieves the warranty clauses. The embeddings are real,
not a stub.

### By id (requires no gateway)

```console
$ uv run embeddings neighbors --id license-006 -k 5
 1. license-007  distance=0.288540
 2. license-011  distance=0.304097
 3. license-008  distance=0.323635
 4. license-014  distance=0.371589
 5. license-009  distance=0.388829
```

Every neighbor of a licence paragraph is another licence paragraph. Ran with no
backend call at all — the stored vector is looked up, not re-embedded.

## 4. `directions`

### `directions overview`

Ran clean, describing the noun, its method, how to read the output, and the
resolved atlas path (`present: yes`). It states the two rules that matter: a
direction is a vector, never a dimension index; and candidates that fail are
still listed rather than hidden. Output is a stable description, not a
measurement, so it is not reproduced in full here.

### `directions discover` on sampled pairs — an honest negative

```console
$ uv run embeddings directions discover
# directions discover

pairs:      64 (45 train, 19 held-out) from sampled
seed:       0
params:     holdout=0.3, pair_source=sampled, pairs=64, resamples=20, runs=5, sample_pairs=64, top=3

generalization test: heldout_mean_cosine >= 0.2 and agreement rate >= 0.75 on the held-out pairs

## candidate 0 — DOES NOT GENERALIZE (reported as a negative result)
held-out:   heldout_mean_cosine -0.008 (min -0.710) over 19 pair(s); agreement rate 0.474
stability:  unstable (adjusted_rand_index 0.315)

## candidate 1 — DOES NOT GENERALIZE (reported as a negative result)
held-out:   heldout_mean_cosine -0.028 (min -0.435) over 19 pair(s); agreement rate 0.526

## candidate 2 — DOES NOT GENERALIZE (reported as a negative result)
held-out:   heldout_mean_cosine -0.011 (min -0.252) over 19 pair(s); agreement rate 0.526

summary: 0 of 3 candidate(s) generalize to the held-out pairs.
```

Randomly sampled pairs encode no shared contrast, so no direction should
generalize — and none did. Held-out cosines sit at roughly zero, which is the
right answer.

### `directions discover` on a meaningful pair set — a positive result

To test that the verb can find a direction when one exists, a pair set encoding
a single deliberate contrast was built and committed as
[`pairs-legal-to-technical.txt`](pairs-legal-to-technical.txt): 18 pairs whose
`a` side is Apache-2.0 legal prose and whose `b` side is measurement design-note
prose. Every difference vector therefore encodes the same register shift.

```console
$ uv run embeddings directions discover --pairs docs/evidence/pairs-legal-to-technical.txt --top 2
pairs:      18 (13 train, 5 held-out) from file:docs/evidence/pairs-legal-to-technical.txt

## candidate 0 — GENERALIZES
held-out:   heldout_mean_cosine 0.506 (min 0.237) over 5 pair(s); agreement rate 1.000
train:      mean cosine 0.612 over 13 pair(s); energy share 0.406
stability:  stable (adjusted_rand_index 0.806): seed reruns 0.854 mean / 0.799 min
            over 10 pair(s); bootstrap 0.806 mean over 20 resample(s)

## candidate 1 — DOES NOT GENERALIZE (reported as a negative result)
held-out:   heldout_mean_cosine 0.060 (min -0.135) over 5 pair(s); agreement rate 0.800
stability:  unstable (adjusted_rand_index 0.450)

summary: 1 of 2 candidate(s) generalize to the held-out pairs.
```

**Demonstrates discrimination, which is the property that matters.** Given
noise the verb reports 0 of 3; given a real contrast it reports 1 of 2, with the
leading candidate generalizing on pairs it never saw (agreement 1.000) and
stable across reruns. A verb that said "generalizes" in both cases would be
worthless; this one distinguishes.

Note also that the atlas's own `limitations[]` entry propagates into the
reader's output — the float-variance caveat appears in `directions discover`
because the atlas it read carries it. Provenance survives the hop.

## 5. `compare`

`compare` needs two atlases. **A genuine cross-model comparison is not possible
on this machine, and this is the main thing this run could not demonstrate.**

The gateway serves exactly one embedding model. The reranker was tried as a
second model and is not served on the embeddings route:

```console
$ uv run embeddings survey docs/evidence/corpus.jsonl --apply --atlas atlas-reranker \
    --embed-model Qwen/Qwen3-Reranker-0.6B
survey: batch 1/3 failed (attempt 1/3): embedding backend unreachable at
  http://localhost:8001/v1/embeddings: HTTP Error 404: Not Found — retrying in 0.25s
survey: batch 1/3 failed (attempt 2/3): ... — retrying in 0.50s
error: embedding backend unreachable at http://localhost:8001/v1/embeddings: HTTP Error 404: Not Found
hint: start the embedding gateway at http://localhost:8001/v1 (resolve the real one with
  'lobes endpoint embedder'), or point elsewhere with --embed-url / the EMBEDDINGS_EMBED_URL
  environment variable
$ echo $?
2
```

Incidentally a clean demonstration of the retry-with-backoff path, exit code 2
for an environment error, and a remediation hint that names the resolver.

### What was demonstrated: the self-consistency sanity check

```console
$ uv run embeddings compare --atlas-a atlas --atlas-b atlas-cached
# compare (relational only — Jaccard conditioned on k=10)

model A:       Qwen/Qwen3-Embedding-0.6B  [atlas]
model B:       Qwen/Qwen3-Embedding-0.6B  [atlas-cached]
probe set:     152 shared item(s)
k:             10
seed:          0

## agreement

k-NN Jaccard:  mean 1.000  min 1.000  max 1.000  spread 0.000   [top-10 sets]
Spearman rank: mean 1.000  min 1.000  max 1.000  spread 0.000   [full ordering
               of all 152 probe items — NOT conditioned on k]

chance baseline at k=10, N=152: expected Jaccard 0.034, expected Spearman 0.000
(observed is +0.966 vs chance)

## coordinate frame

shared: no (model A is 1024-dimensional, model B is 1024-dimensional)
[...] Nothing in this output is a coordinate, a distance, or a cosine [...]

## notes

- both atlases were written by the same model (Qwen/Qwen3-Embedding-0.6B): this is
  a self-consistency sanity check, not a cross-model comparison, and agreement near
  1.0 is the expected result rather than a finding
```

Agreement is exactly 1.000 against a chance baseline of 0.034 — the correct
answer for two atlases with byte-identical vectors, and it validates the metric
plumbing (shared-probe matching, Jaccard, Spearman, chance baseline).

The part worth calling out: **the verb detects and labels its own triviality.**
Without being told, it noticed both atlases came from the same model and
demoted its own headline number to "expected result rather than a finding". That
is the honesty rule enforcing itself in code rather than in a doc.

The output also carries all five assumptions from design note §2 with a status
each, including two frankly marked `not verifiable by this tool`, and a "what
this does not say" block stating that agreement is not accuracy and that the
verb does not rank the two models.

### A second attempt, and why it also returned 1.000

To get a non-trivial number, a third atlas was built over the same corpus minus
the 18 licence items, then compared on the 134 shared items. It also returned
Jaccard 1.000. The reason is a deliberate design property, confirmed in
`embeddings/cli/_commands/compare.py`: neighbor orders are computed from
`atlas_a.vectors[rows_a]` and `atlas_b.vectors[rows_b]` — that is, **within the
shared probe set only**. Corpus composition outside the shared set cannot
perturb the comparison, which isolates model difference from corpus difference.
Correct behaviour, and it means that with one embedding model available,
`compare` on this machine can only ever return 1.000.

**Honest status: `compare` is exercised and correct, but its actual purpose —
locating where two models disagree — is unproven here and needs a second
embedding model on the gateway to test.**

## 6. `gaps`

### Without probing

```console
$ uv run embeddings gaps
# gaps

atlas:         atlas
items:         152 analysed of 152
params:        k=5, sparsity_quantile=0.25, void_factor=2.0, coherence_factor=0.75
seed:          0
partition:     unstable

Corpus sparsity alone says nothing about model capability. [...] Only an active
probe, embedding new inputs at probe time and measuring where they land, can
isolate the last one. The two finding lists are never merged, summed, or ranked
together.

## corpus coverage — 2 finding(s)
   what the surveyed corpus does and does not contain. Not about the model.

- [sparse_region] Region 0 holds 18 surveyed item(s) whose mean distance to their
  nearest neighbour is 0.323940, at or above the 0.271252 cut for this run.
  does not show: This does not show that the model cannot represent this area. [...]
  asserted: False (unstable)

- [sparse_region] Region 1 holds 34 surveyed item(s) [...] 0.271252 [...]
  asserted: False (unstable)

## model representation — 0 finding(s)
   what probing actually showed. Only ever produced by a live probe step.

no probe step ran (probing is off by default), so no claim about the model is made here.
run again with --probe and --probes <file> (or --holdout <n>) to test the model.
```

The two kinds are separate sections with separate counts, and the
model-representation list is emitted **empty rather than omitted** — an agent
aggregating this cannot flatten a corpus fact into a model claim by accident.
Both coverage findings also carry `asserted: False` because the underlying
partition is unstable.

### With probing — the first live run of this path

This is the first time the probe path could actually execute; before the gateway
fix it had never run. Probe inputs were written for the run and committed as
[`probes-legal.txt`](probes-legal.txt): eight held-out legal-boilerplate
sentences, none of which appears in the corpus, aimed at the sparsest region
(region 0, the 18 licence items).

```console
$ uv run embeddings gaps --probe --probes docs/evidence/probes-legal.txt

## model representation — 1 finding(s)
   what probing actually showed. Only ever produced by a live probe step.

- [probes scattered] The 8 probe(s) that landed nearest region 0 scatter: their
  mean pairwise distance is 0.53298, at or above the 0.438237 coherence threshold
  (0.75 x the corpus-wide baseline spread of 0.584316) — that is, no tighter than
  this model places arbitrary text. The model does not give these inputs a shared
  neighbourhood here. That is evidence about the representation in this area.
  does not show: This is evidence, not proof. It is conditioned on these 8 probe(s)
  being fair instances of what belongs in region 0, on the k=5 partition that
  defined the region, and on the stated coherence threshold. It does not show the
  model is incapable of representing this area, it does not generalise to any other
  region, and it is not a measurement of the corpus.

inconclusive [probes landed elsewhere]: None of the 8 probe(s) landed nearest
region 1; they were nearer other regions. That says the probes were not instances
of what this region holds, or that the region is unreachable by them — it says
nothing about how the model represents the region.
```

The probe path works: it embedded eight new inputs at probe time, measured where
they landed, and produced a `model representation` finding that a non-probe run
cannot produce.

**How much should be read into the finding: less than it first appears, and the
tool says so.** The probes span patent grants, trademark restrictions, liability
limits and attribution notices — genuinely different legal sub-topics. Their mean
pairwise distance (0.533) *is* tighter than the corpus-wide baseline (0.584) but
does not clear the 0.75× threshold (0.438). A plausible competing explanation is
that the probe set is topically diverse within the legal register, so the
measurement is partly reading probe diversity rather than model behaviour. That
is exactly the fair-instance caveat the output itself states. Recording it here
as an honest limit on the finding, not as a defect in the verb.

### With `--holdout`, where probes are fair instances by construction

```console
$ uv run embeddings gaps --probe --holdout 20

## model representation — 0 finding(s)
(the probe step ran but produced no finding; see inconclusive below)

inconclusive [probes landed elsewhere]: None of the 20 probe(s) landed nearest
region 0; they were nearer other regions.
inconclusive [too few probes landed to measure spread]: Only 1 probe(s) landed
nearest region 1, so there is no pairwise spread to compare against the corpus
baseline. A verdict from a single point would be a verdict from nothing.
```

A random holdout across the whole corpus rarely concentrates in a specific
sparse region, so "inconclusive" is the correct result. Two distinct
inconclusive reasons are reported rather than one vague one, and the verb
refuses to derive a verdict from a single probe.

## 7. `map`

### Preview writes nothing

```console
$ uv run embeddings map --html docs/evidence/map.html
# map (dry run — nothing written)

algorithm:     pca
params:        components=2, k=5, runs=5, resamples=20
seed:          0
explained variance: PC1 10.5%, PC2 9.0% (total 19.5%)
colouring:     unstable

artifact:      docs/evidence/map.html
re-run with --apply to write the artifact. Nothing was written.
```

No file existed afterwards.

### `--apply` writes the artifact

```console
$ uv run embeddings map --html docs/evidence/map.html --apply
# map (applied)
...
artifact:      docs/evidence/map.html
$ ls -l docs/evidence/map.html
-rw------- 1 spark spark 110817 Jul 20 19:02 docs/evidence/map.html
```

Committed as [`map.html`](map.html): 110,817 bytes on disk, 110,477 characters
of UTF-8 markup. Self-containment verified
mechanically:

| Check | Result |
|---|---|
| `src`/`href` references of any kind | none |
| Remote (`http:`, `https:`, `//`) references | none |
| `fetch(` / `XMLHttpRequest` / `WebSocket` | 0 occurrences each |
| `@import` | 0 occurrences |
| Inline `<svg>` plot with `<circle>` marks | present |
| `prefers-color-scheme` (light and dark) | present |

The page opens with no network access. Its provenance block carries the atlas
path, model, endpoint, item count and dimensionality, algorithm, per-component
and total explained variance, seed, `k`, stability run counts, the sign
convention, and the aspect rule — plus the caveat that PCA keeps only 19.5% of
the variance, so inter-cluster distances on the page are not faithful and "a gap
on this page is not evidence of a gap in the model".

Re-running `--apply` over the existing file is refused:

```console
$ uv run embeddings map --html docs/evidence/map.html --apply
error: an artifact already exists at docs/evidence/map.html
hint: re-run with --force to overwrite it, or pass a different --html <path>
$ echo $?
1
```

Worth stating plainly: **19.5% retained variance is a weak projection.** The
picture is close to uninformative as a 2-D summary of a 1024-dimensional space,
and the artifact says so itself rather than letting the reader infer structure
from the layout.

## The round-trip check

Does a reader verb load what `survey` wrote and reproduce it identically? Tested
by recomputing the measurement independently from the three files on disk —
reading `vectors.npy`, `records.jsonl` and `atlas.json` directly with numpy and
redoing the cosine ranking by hand — then comparing against the CLI's own
`--json` output.

```console
$ uv run embeddings neighbors --id license-006 -k 5 --json > nb.json
$ # independent recomputation from atlas/vectors.npy, atlas/records.jsonl
manifest model     : Qwen/Qwen3-Embedding-0.6B
vectors.npy shape  : (152, 1024) float32

rank CLI id                recomputed id             CLI dist    recomputed     delta
1    license-007           license-007               0.288540      0.288540  1.79e-07
2    license-011           license-011               0.304097      0.304097  8.94e-08
3    license-008           license-008               0.323635      0.323635  2.98e-08
4    license-014           license-014               0.371589      0.371589  1.19e-07
5    license-009           license-009               0.388829      0.388829  1.19e-07

ids match          : True
all 152 texts round-trip corpus -> atlas -> reader: True
```

Every neighbor id matches and every distance agrees to within float32 epsilon
(worst delta 1.8e-07). All 152 corpus texts survive the corpus → atlas → reader
path unchanged. **The Atlas round-trips.**

## The determinism check

Each reader verb was run twice with identical inputs and seed over the fixed
atlas, and the `--json` payloads compared byte for byte:

| Command | Result | Payload |
|---|---|---:|
| `embeddings regions --json` | byte-identical | 19,589 B |
| `embeddings neighbors --id license-006 -k 5 --json` | byte-identical | 3,463 B |
| `embeddings directions discover --json` | byte-identical | 18,953 B |
| `embeddings gaps --json` | byte-identical | 5,159 B |
| `embeddings map --json` | byte-identical | 90,122 B |

All five reader verbs are byte-identical across repeated runs. This is the
guarantee design note §4 makes for readers over a fixed Atlas, and it holds at
real corpus scale.

## Parameter and seed stamping

Checked mechanically across every structural verb's `--json` payload:

| Verb | `params` | `seed` | `--json` | Reproducing command |
|---|:-:|:-:|:-:|:-:|
| `regions` | yes | yes | yes | no |
| `neighbors` | yes | yes | yes | no |
| `directions discover` | yes | yes | yes | no |
| `gaps` | yes | yes | yes | no |
| `compare` | yes | yes | yes | partial |
| `map` | yes | yes | yes | no |

Two of the three clauses in the acceptance criterion hold without qualification.
The third does not — see finding F2.

## Findings

### F1 — `survey` accepts duplicate record ids silently

**Severity: low-to-moderate. Not fixed here** (the fix would land under
`embeddings/`, outside this task's file boundary).

The first corpus build derived record ids from filename stems. Because
`docs/specs/2026-07-20-embeddings-survey-surface.md` and
`docs/plans/2026-07-20-embeddings-survey-surface.md` share a stem, 9 ids were
duplicated — 152 records carrying only 143 distinct ids.

`survey --apply` accepted this without a warning, and reported
`items: 152 (152 unique texts)` — a count that is true about *texts* and easy to
misread as a statement about ids. The resulting atlas contained 152 records
under 143 ids.

The downstream consequence:

```console
$ uv run embeddings neighbors --id 2026-07-20-embeddings-survey-surface-001 -k 3
query:    '2026-07-20-embeddings-survey-surface-001' (source: atlas-item)
 1. 2026-07-20-embeddings-survey-surface-008  distance=0.145036
 ...
```

The verb silently resolved an ambiguous id to one of the two records with no
warning that a second record shared it. Nothing crashed and nothing was
incorrect in a narrow sense, but "look up item X" quietly became "look up
whichever X was written first". Ids are the join key for `--pairs`, `--id`
lookups, and `compare`'s shared-probe matching, so an ambiguous id space
undermines all three.

Suggested handling for the merging agent: have `survey` either reject a corpus
with duplicate ids (exit 1, with the offending ids in the remediation) or
disambiguate them and report that it did. Either is better than silence. The
committed builder now guards its own output and fails loudly, but that protects
this corpus, not the verb.

### F2 — no verb emits the command that reproduces its output

**Severity: moderate against the stated acceptance criterion. Not fixed here.**

Acceptance criterion 3 reads: "Every structural output is
parameter-and-seed-stamped, supports `--json`, and **cites the command that
reproduces it**." The first two clauses hold everywhere. The third does not.

No verb emits a literal reproducing command line — not in text output, not in
any `--json` payload, and not in the standalone HTML artifact. A search across
all six structural payloads found one command-ish key, `compare`'s
`"command": "compare"`, which is the verb name rather than a re-runnable
invocation.

What *is* present on every payload is `params`, `seed`, `atlas` and `model` —
enough information for a reader to reconstruct the command. So the criterion is
met in substance and missed in letter. The gap matters most for
[`map.html`](map.html), which is explicitly designed to travel on its own: its
provenance block is rich but a reader who wants to re-run it must reconstruct
the invocation from the parameter list by hand. The design note's own framing —
"why every structural claim carries the command that reproduces it" — sets the
stronger expectation.

Suggested handling: add a `reproduce` string to each payload (and a
corresponding line in the HTML provenance block) built from the resolved
parameters, so the printed command is the one that actually ran rather than the
one the operator typed.

### Observation — cache-served surveys still log "embedded"

Cosmetic. A fully cache-served survey prints progress reading
`survey: embedded 64/152 texts (batch 1/3)` on stderr even though the manifest
correctly records `embedded_live: 0` and `cache_hits: 152`. The stdout summary
is accurate; only the stderr progress wording implies backend calls that did not
happen. Worth a word change if someone is in the file anyway.

### Observation — the "where they disagree most" section under total agreement

Cosmetic. When two atlases agree completely, `compare` still prints a "where
they disagree most" heading followed by five items at Jaccard 1.000 with empty
`only A` / `only B` lists. Reading the list is unambiguous, but the heading
promises disagreements that do not exist. A "no disagreements at this k" line
would read better in the total-agreement case.

## Repository state after the run

Verified with the full gate set, unchanged from before the run:

| Gate | Result |
|---|---|
| `uv run pytest -n auto -q` | 584 passed in 1.55 s |
| `uv run black --check` | 55 files unchanged |
| `uv run isort --check-only` | clean |
| `uv run flake8` | clean |
| `uv run bandit -c pyproject.toml -r embeddings` | 0 findings at every severity |
| `uv run teken cli doctor . --strict` | exit 0 |
| `markdownlint-cli2 "docs/evidence/*.md"` | clean |

**No file under `embeddings/` or `tests/` was modified, and no test requiring a
gateway was added.** CI remains hermetic.

The scratch atlases (`atlas/`, `atlas-cached/`, `atlas-nolicense/`) were removed
after the run. They are regenerable from the committed corpus by the commands
below, and `*.npy` vector bulk is git-ignored by design.

## Reproducing this run

```bash
export EMBEDDINGS_API_KEY="$COLLEAGUE_API_KEY"   # value never printed

uv run python docs/evidence/build_corpus.py docs/evidence/corpus.jsonl
uv run embeddings survey docs/evidence/corpus.jsonl                 # dry run
uv run embeddings survey docs/evidence/corpus.jsonl --apply         # writes atlas/
uv run embeddings survey docs/evidence/corpus.jsonl --apply --atlas atlas-cached

uv run embeddings regions
uv run embeddings neighbors "how stable is a k-means partition across seeds" -k 5
uv run embeddings neighbors --id license-006 -k 5
uv run embeddings directions overview
uv run embeddings directions discover
uv run embeddings directions discover --pairs docs/evidence/pairs-legal-to-technical.txt --top 2
uv run embeddings compare --atlas-a atlas --atlas-b atlas-cached
uv run embeddings gaps
uv run embeddings gaps --probe --probes docs/evidence/probes-legal.txt
uv run embeddings gaps --probe --holdout 20
uv run embeddings map --html docs/evidence/map.html --apply --force
```

Exact numbers will differ slightly on a cold cache, because live GPU embedding
is not bit-deterministic — which is the limitation the atlas records for you,
and which was observed moving a stability verdict during this very run.
