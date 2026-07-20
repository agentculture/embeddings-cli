# Does Qwen3-Embedding-0.6B distinguish what `eidetic recall` needs distinguished?

Date: 2026-07-20. Branch: `agent/t23`. Plan task: t23.
Model under study: `Qwen/Qwen3-Embedding-0.6B`, 1024-dimensional, served at
`http://localhost:8001/v1`.

This is **operator-machine evidence**, not a CI assertion. Every number below
was produced by hand on one workstation against a live embedding gateway. No
test in this repository requires a gateway and this study adds none.

## The answer, in plain language

**No — not reliably, and the reason is not the one the pilot suggested.**

`eidetic recall` needs to tell a memory from its own contradiction: "port 8001"
from "port 8002", "we decided to" from "we decided against". This model does
encode that difference, but it encodes it at roughly **one tenth** the strength
with which it encodes *how the sentence is worded*. The practical consequence is
an ordering failure: a record that contradicts the truth but is worded like it
lands closer to a query than a record that states the truth in different words.
On the twenty facts probed here that inversion held for **19 of 20**.

The pilot that motivated this study read that result as "the model cannot
represent negation." **That reading does not survive.** A matched factorial
design that varies meaning and wording independently shows the effect is
overwhelmingly a *lexical-overlap* effect. Contradiction contributes a small,
consistently-signed, but nearly negligible amount on top. That is a sharper and
more actionable finding than the pilot's, and it changes the recommended fix:
you cannot solve this by choosing a model that is "better at negation", because
the model is not primarily failing at negation. You solve it by not asking a
bi-encoder to adjudicate near-duplicates at all.

Three further results worth stating up front:

- **Instruction-prefixing does not fix it.** Qwen3-Embedding's documented query
  prefix roughly triples the contradiction signal but inflates paraphrase
  distance at the same time, so the inversion survives (17–18 of 20).
- **Value substitution and negation behave differently.** Negation has a weak
  but consistently *discoverable* shared direction in this space; value
  substitution has none at all. A retrieval benchmark cannot see this
  distinction, and it determines which of the two is correctable.
- **Realistic record length makes the margins smaller, not safer.** At the
  length of real `~/.eidetic/memory` records, a contradicting record sits at
  **2.98%** of the unrelated-control distance.

Section [Limitations](#limitations) states what this does not establish. The
most important entry: this is 20 facts on one model, and the
lexical-versus-semantic split, while large, is measured on synthetic probes
written by the same author who predicted the result.

## How to reproduce everything

```bash
export EMBEDDINGS_API_KEY="$COLLEAGUE_API_KEY"   # value never printed
export EMBEDDINGS_CACHE_DIR="$PWD/.study-cache"  # keeps the operator cache clean

uv run python docs/studies/build_probe_corpus.py
uv run embeddings survey docs/studies/probe-corpus.jsonl --apply \
    --atlas docs/studies/atlas-plain
uv run embeddings survey docs/studies/probe-corpus-instruct.jsonl --apply \
    --atlas docs/studies/atlas-instruct
uv run embeddings survey docs/studies/probe-corpus-long.jsonl --apply \
    --atlas docs/studies/atlas-long
uv run python docs/studies/measure_pairs.py
```

Committed alongside this document:

| Artifact | What it is |
|---|---|
| `probes.py` | The probe sets: 20 facts x 4 matched variants, plus 20 controls |
| `build_probe_corpus.py` | Deterministic builder for all three corpora |
| `measure_pairs.py` | The ad-hoc pair-distance measurement (see the note below) |
| `probe-corpus*.jsonl` | The three corpora as surveyed |
| `atlas-plain/`, `atlas-instruct/`, `atlas-long/` | The committed Atlases, minus git-ignored `.npy` sidecars |
| `pairs-*.txt` | Pair files consumed by `directions discover` |
| `probes-heldout.txt` | Held-out probe inputs for `gaps --probe` |

**Which numbers came from the tool and which did not.** Every structural claim
in this study — regions, directions, gaps, neighbor rankings — is produced by an
`embeddings` verb, and each carries its command inline. The one exception is the
table of distances between named pairs of Atlas items: no verb emits that, so
`measure_pairs.py` reads the committed Atlases directly. It re-embeds nothing;
it reads vectors exactly as `survey` wrote them, which places it on the inside
of the Atlas reproducibility boundary described in
`docs/design-notes/semantic-measurement.md` section 4. Tables sourced from it
are marked *(ad-hoc)*.

**No verb prints its own reproducing command** (finding F2 in
`docs/evidence/2026-07-20-live-end-to-end.md`), so every command in this study
is written out by hand.

## Why the pilot could not settle the question

The pilot used four independent probe sets and reported mean cosine distance:

| Pilot set | n | Mean distance | Expected |
|---|---:|---:|---|
| value substitution | 5 | 0.041 | far |
| negation | 5 | 0.064 | far |
| paraphrase | 5 | 0.160 | close |
| unrelated controls | 5 | 0.580 | baseline |

Read naively this says contradictions sit four times closer than paraphrases —
a spectacular failure. But the design has a confound that fully accounts for it:
**the value pairs differed by one token while the paraphrases shared almost no
vocabulary.** Two very different claims predict exactly this table:

- *Semantic account.* The model does not represent negation or value identity,
  so contradictory statements collapse onto their sources.
- *Surface-form account.* The model represents both fine, but weights surface
  form so heavily that a one-token edit — whichever token — barely moves the
  vector.

These have different consequences. The first says swap the model. The second
says no bi-encoder of this family will help and the fix belongs elsewhere in the
pipeline. The pilot cannot distinguish them, and at n=5 per set it also cannot
report a distribution.

## The design that separates them

`probes.py` replaces the four independent sets with a **matched 2x2 factorial**
over twenty facts. Each fact carries one base statement and four variants:

| Variant | Meaning vs. base | Surface form vs. base |
|---|---|---|
| `para_high` | same | near-identical (one non-discriminating token changed) |
| `contra_high` | opposite | near-identical (only the discriminating token changed) |
| `para_low` | same | reworded, little shared vocabulary |
| `contra_low` | opposite | reworded, little shared vocabulary |

Worked example (fact `v01`):

| Variant | Text |
|---|---|
| base | the lobes embedding gateway listens on port 8001 |
| `para_high` | the lobes embedding gateway **runs** on port 8001 |
| `contra_high` | the lobes embedding gateway listens on port **8002** |
| `para_low` | port 8001 is where you reach the embedding service that lobes fronts |
| `contra_low` | port **8002** is where you reach the embedding service that lobes fronts |

`para_high` and `contra_high` are both one-token edits of the base. The only
difference between them is *which* token changed: a neutral one, or the
discriminating one. That is the razor. `contra_low` is built as `para_low` with
the discriminating element flipped, so the low-overlap cells are matched to each
other the same way.

Because all four variants concern the same fact in the same register at similar
length, topic and style are held fixed and only meaning and wording vary. The
two accounts now predict different orderings:

- Surface-form account: `para_high` ~ `contra_high` << `para_low` ~ `contra_low`
- Semantic account: `para_high` ~ `para_low` << `contra_high` ~ `contra_low`

Ten facts use value substitution (port, version, commit hash, date, threshold);
ten use negation or decision reversal. Twenty unrelated controls from unrelated
domains set the baseline. Total: 120 texts.

Measured token-level Jaccard overlap against the base confirms the design did
what it intended *(ad-hoc)*:

| Cell | Mean token Jaccard vs. base |
|---|---:|
| same meaning, high overlap | 0.7784 |
| contradicts, high overlap | 0.7486 |
| same meaning, low overlap | 0.1255 |
| contradicts, low overlap | 0.0895 |

The two high-overlap cells are matched to each other (0.78 vs 0.75) and the two
low-overlap cells likewise (0.13 vs 0.09), so overlap is a property of the
row, not of the meaning.

## Result 1 — the recall failure is real

Command: `uv run python docs/studies/measure_pairs.py`. Seed 0.
Atlas: `docs/studies/atlas-plain` (120 items x 1024 dims). *(ad-hoc table)*

Baselines first:

| Baseline | n | Mean | sd | Min | Median | Max |
|---|---:|---:|---:|---:|---:|---:|
| unrelated controls | 190 | 0.5875 | 0.0758 | 0.3859 | 0.5911 | 0.7576 |
| cross-fact bases | 190 | 0.5838 | 0.0726 | 0.2715 | 0.5946 | 0.7188 |

The full factorial:

| Cell | n | Mean | sd | Min | Median | Max |
|---|---:|---:|---:|---:|---:|---:|
| same meaning, high overlap | 20 | 0.0430 | 0.0725 | 0.0043 | 0.0181 | 0.3284 |
| contradicts, high overlap | 20 | 0.0521 | 0.0573 | 0.0056 | 0.0264 | 0.2071 |
| same meaning, low overlap | 20 | 0.2895 | 0.0954 | 0.1164 | 0.3057 | 0.4817 |
| contradicts, low overlap | 20 | 0.3349 | 0.1258 | 0.1324 | 0.3201 | 0.6324 |

**The operationally relevant comparison is row 2 against row 3.** A contradicting
near-duplicate sits at 0.0521; a correct paraphrase in different words sits at
0.2895 — 5.6 times further. Per fact, `contra_high` is closer to the base than
`para_low` for **19 of 20 facts** (all but `v09`). A contradiction sits at
**8.9%** of the unrelated-control distance.

That inversion is the failure mode. Distribution overlap makes the strength of
the claim precise rather than leaving it at the means:

| Comparison | Verdict | Overlap |
|---|---|---|
| contradiction+high-overlap vs paraphrase+low-overlap | OVERLAP | 10% of `contra_high` at or above min(`para_low`); 25% of `para_low` at or below max(`contra_high`) |
| within high overlap: contradiction vs paraphrase | OVERLAP | 100% / 95% — effectively coincident |
| within low overlap: contradiction vs paraphrase | OVERLAP | 100% / 100% — effectively coincident |

So the headline separation is large but **not clean**: the two distributions do
touch. One in ten contradictions is further away than the nearest paraphrase.
The claim is "strongly shifted distributions", not "separated distributions",
and no single distance threshold cleanly divides them.

### The same failure through `neighbors`

The distance table is abstract; this is what it does to retrieval. Commands:

```bash
uv run embeddings neighbors --atlas docs/studies/atlas-plain \
    "which port does the lobes embedding gateway listen on" -k 5
```

```text
 1. v01-para-high    distance=0.083992  'the lobes embedding gateway runs on port 8001'
 2. v01-contra-high  distance=0.089946  'the lobes embedding gateway listens on port 8002'
 3. v01-base         distance=0.093615  'the lobes embedding gateway listens on port 8001'
 4. v01-para-low     distance=0.168440  'port 8001 is where you reach the embedding service that lobes fronts'
 5. v01-contra-low   distance=0.176446  'port 8002 is where you reach the embedding service that lobes fronts'
```

**The wrong answer ranks second, above the correct record at third.** The margin
between them is 0.0037 — well inside the noise any operator would accept as
meaningless.

```bash
uv run embeddings neighbors --atlas docs/studies/atlas-plain \
    "does the test suite need the embedding gateway running" -k 5
```

```text
 1. n07-base         distance=0.030078  'the test suite requires a running embedding gateway'
 2. n07-para-high    distance=0.042639  'the test suite needs a running embedding gateway'
 3. n07-contra-high  distance=0.092268  'the test suite runs without any embedding gateway'
 4. v01-para-high    distance=0.283830  'the lobes embedding gateway runs on port 8001'
 5. n07-contra-low   distance=0.297009  'you can get a green run whether or not the embedder is up'
```

Here the ordering is correct at the top — and then at rank 4 a record **about an
entirely different fact** (port numbers) outranks the correct low-overlap
paraphrase of the answer, which does not make the top 5 at all. Topical
relevance and lexical overlap have swapped places.

Command: for every fact, look up its base by id and inspect its own neighborhood
(no backend call, vectors read from the Atlas):

```bash
uv run embeddings neighbors --atlas docs/studies/atlas-plain --id v01-base -k 4 --json
```

Aggregated over all 20 facts:

- For **19 of 20** facts the contradicting near-duplicate is among the base's
  **two** nearest neighbors in the entire space.
- For **5 of 20** facts the contradiction is the base's **single nearest
  neighbor** — the closest item in the space to a true statement is its own
  falsification.
- For the other 15, the high-overlap paraphrase is nearest. That 15-to-5 split
  is the small semantic signal showing up: within matched surface form the model
  does weakly prefer the paraphrase.

## Result 2 — the mechanism is lexical, and the pilot's reading is overturned

This is the study's most important result.

Command: `uv run python docs/studies/measure_pairs.py`. *(ad-hoc)*

Reading the factorial as main effects:

| Effect | Size |
|---|---:|
| **Wording** (low-overlap minus high-overlap, meaning held fixed) | **+0.2646** |
| **Meaning** (contradiction minus paraphrase, wording held fixed) | **+0.0273** |
| Ratio | **9.7x** |

Within-row detail:

| Held fixed | Meaning effect |
|---|---:|
| high overlap | +0.0091 |
| low overlap | +0.0454 |

And correlations across all 80 base-to-variant distances:

| Correlation | r |
|---|---:|
| lexical overlap vs distance | **-0.8405** |
| contradiction vs distance | +0.0851 |
| lexical overlap vs contradiction (design check) | -0.0483 |
| **partial** r(contradiction, distance) controlling for overlap | **+0.0821** |

The observed ordering is `para_high` (0.043) ~ `contra_high` (0.052) <<
`para_low` (0.290) ~ `contra_low` (0.335). **That is the surface-form
prediction, essentially exactly.** The semantic prediction is refuted.

Three things follow, and the third is the one that matters:

1. **Lexical overlap explains the pilot.** r = -0.84 for overlap against
   distance; +0.085 for contradiction. The design check confirms the two
   predictors are near-orthogonal by construction (r = -0.05), so the partial
   correlation barely differs from the raw one.
2. **The model is not blind to meaning.** The meaning effect is small but it is
   real and consistently signed: positive in both overlap rows, positive in both
   contradiction kinds, and visible in the 15-to-5 nearest-neighbor split. A
   model that genuinely could not represent negation would produce a meaning
   effect indistinguishable from zero. This one does not.
3. **The pilot's conclusion was right about the symptom and wrong about the
   cause.** Recall *will* rank a contradicting record above a correctly-worded
   one. But it is not because the model cannot tell them apart — it is because
   the model's similarity is dominated by how the sentence is written, and
   meaning is a rounding error on top. Choosing a model that scores better on
   negation benchmarks would not fix this, because the failure is a scaling
   problem, not a representation problem.

**The pilot's headline number is retracted.** "Contradictory facts sit 4x closer
than paraphrases" is an artifact of comparing a one-token edit against a full
rewrite. The defensible restatement: *distance in this space is governed by
lexical overlap at roughly ten times the weight of propositional content, and
because contradictions are usually lexically similar to what they contradict
while paraphrases are not, recall inverts.*

## Result 3 — value substitution and negation are not the same problem

A retrieval benchmark would treat these as one bucket. They are not, and the
difference determines whether the failure is correctable.

Command:

```bash
uv run embeddings directions discover --atlas docs/studies/atlas-plain \
    --pairs docs/studies/pairs-negation-contra.txt --top 2 --seed 0
uv run embeddings directions discover --atlas docs/studies/atlas-plain \
    --pairs docs/studies/pairs-value-contra.txt --top 2 --seed 0
```

Params on both: `holdout=0.3, resamples=20, runs=5, top=2`, seed 0. Each pair
set is 10 pairs (7 train, 3 held-out).

| Pair set | Result | Held-out cosine | Agreement | Direction stability |
|---|---|---:|---:|---|
| negation contradictions | **1 of 2 candidates GENERALIZES** | 0.241 (min 0.004) | 1.000 | `unstable` (ARI 0.479) |
| value contradictions | **0 of 2 generalize** | 0.026 / -0.007 | 0.667 | `unstable` (ARI 0.332) |
| all 20 contradictions pooled | 0 of 2 generalize | 0.030 / 0.069 | 0.500 / 0.833 | `unstable` (ARI 0.245) |
| paraphrase control | 0 of 2 generalize | 0.048 / -0.072 | 0.833 / 0.333 | `moderate` (ARI 0.597) |

Seed sweep, five seeds each, same commands with `--seed 0..4`:

| Seed | negation | value |
|---:|---|---|
| 0 | 1 of 2 generalize | 0 of 2 |
| 1 | 1 of 2 generalize | 0 of 2 |
| 2 | 1 of 2 generalize | 0 of 2 |
| 3 | 1 of 2 generalize | 0 of 2 |
| 4 | 1 of 2 generalize | 0 of 2 |

**The verdict is stable across all five seeds; the direction vector itself is
not.** Those are different claims and the study asserts only the first. Per
`docs/design-notes/semantic-measurement.md` section 3, the individual direction
carries its stability label — `unstable`, ARI 0.479 — and belongs in
[Uncertain](#uncertain), not in the settled findings. What is asserted here is
the *asymmetry between the two pair sets*, which reproduced at every seed tried.

The reading, stated conservatively:

- **Negation has a partially shared geometry.** Flipping "decided to" to
  "decided against" moves different sentences in a related-enough direction that
  a direction derived from seven pairs transfers to three it never saw, with
  agreement 1.000. The magnitude is small, but it is coherent.
- **Value substitution has none.** Swapping 8001 for 8002 and swapping
  `fe3e18b` for `2fdd056` move their sentences in unrelated directions. Held-out
  cosine sits at 0.026 and -0.007 — indistinguishable from noise, and matching
  the sampled-pairs negative control in
  `docs/evidence/2026-07-20-live-end-to-end.md`.
- **Pooling them destroys the negation signal** (0 of 2), which is why a study
  that lumps "contradiction" into one bucket would have concluded, wrongly, that
  the model has no contradiction geometry at all.

Why this matters operationally: a consistent direction is something you can
*correct for* — project it out, or use it as a re-ranking feature. Negation
offers a weak handle of that kind. Value substitution offers none, so a stale
"port 8002" record cannot be geometrically separated from a current "port 8001"
one by any linear method fit on other value swaps. Those two failure modes need
different mitigations.

### This is the finding a retrieval benchmark could not surface

A benchmark asks: for query q with relevant document d, is d in the top k? For
every query in [Result 1](#the-same-failure-through-neighbors), **it is**. The
correct record is in the top 3 every time. Both members of each probe pair are
topically about the query, so a benchmark scores a hit regardless of which one
ranks first — the wrong port number is retrieved *for a question about the port
number*.

The benchmark's relevance label is exactly the resolution this failure hides
beneath. What is broken is not whether the right neighborhood is found but the
**ordering within a neighborhood of near-duplicates**, and that ordering is what
a memory store depends on when it has both a current and a superseded record.
Nothing in a retrieval score is sensitive to it, and no amount of benchmark
improvement would register the difference between a model that gets this right
and one that gets it backwards.

The asymmetry in this section is a second-order version of the same point: it is
a fact about the *shape* of the errors, invisible to any metric that only counts
whether relevant documents were returned.

## Result 4 — the instruction prefix does not fix it

Qwen3-Embedding is trained to accept an instruction prefix on the query side.
The task description used (`probes.INSTRUCTION`):

```text
Instruct: Given a question about a project's recorded decisions and facts, retrieve the memory record that answers it
Query: <the text>
```

Three conditions were measured, all seed 0 *(ad-hoc)*. The asymmetric condition
is the realistic one: stored records unprefixed, query prefixed — the way the
prefix is actually meant to be used. It measures across `atlas-plain` and
`atlas-instruct`, which is legitimate only because both come from the same
model and therefore share a coordinate frame; section 2 of the design note
forbids this across models, and it is not done across models anywhere here.

| Condition | `para_high` | `contra_high` | `para_low` | `contra_low` | Inversions | Gap |
|---|---:|---:|---:|---:|---:|---:|
| plain (neither prefixed) | 0.0430 | 0.0521 | 0.2895 | 0.3349 | 19/20 | +0.2374 |
| symmetric (both prefixed) | 0.0402 | 0.0774 | 0.3446 | 0.4144 | 18/20 | +0.2672 |
| asymmetric (query prefixed) | 0.1279 | 0.1591 | 0.3689 | 0.4275 | 17/20 | +0.2098 |

"Gap" is mean(`para_low`) minus mean(`contra_high`); positive means the
inversion holds.

**Verdict: it helps a little and fixes nothing.** In detail:

- The **contradiction signal genuinely improves**. The meaning effect within
  high overlap rises from +0.0091 (plain) to +0.0372 (symmetric) to +0.0312
  (asymmetric) — roughly a threefold increase. The prefix does make the model
  attend more to propositional content.
- But **paraphrase distance inflates alongside it**, from 0.2895 to 0.3446 and
  0.3689. The prefix pushes everything apart, not just contradictions.
- So the **inversion survives**: 17 or 18 of 20 facts rather than 19, and the
  gap in the symmetric condition is *wider* than plain. Two facts flip in the
  asymmetric condition, which is a real improvement and nowhere near enough.

**Operational recommendation for eidetic: adopt the prefix, but not as a fix for
this.** It measurably improves contradiction sensitivity at no cost, so there is
no reason not to use it in the documented asymmetric form. It does not make
recall safe against superseded records, and shipping it as if it did would be
worse than not shipping it.

## Result 5 — at realistic record length the margins shrink

Real records in `~/.eidetic/memory` were read read-only for shape. Across 18
records (no private content is reproduced here or in `probes.py`):

| Property | Value |
|---|---|
| Content length | min 82, median 2,249, max 127,413 characters |
| Contain a status or polarity word | 11 / 18 |
| Contain a bare integer | 13 / 18 |
| Contain a commit-hash-like token | 6 / 18 |
| Contain an ISO date | 6 / 18 |
| Contain a branch or path slug | 5 / 18 |
| Contain an issue or PR number | 4 / 18 |

Real records are therefore **dense in exactly the discriminating tokens this
study varies**, and roughly 45 times longer than the bare probes. A single
changed token is a much smaller fraction of a long record's surface, so the
lexical-dominance effect should get *worse*, not better. `probe-corpus-long.jsonl`
tests that: the same 100 fact variants embedded in identical, information-free
session boilerplate at a median 1,645 characters.

Command:

```bash
uv run embeddings survey docs/studies/probe-corpus-long.jsonl --apply \
    --atlas docs/studies/atlas-long
```

| Cell | Short mean | Long mean | Ratio |
|---|---:|---:|---:|
| same meaning, high overlap | 0.0430 | 0.0114 | 0.26x |
| contradicts, high overlap | 0.0521 | 0.0175 | 0.34x |
| same meaning, low overlap | 0.2895 | 0.0960 | 0.33x |
| contradicts, low overlap | 0.3349 | 0.1120 | 0.33x |

| Condition | Inversions | Gap |
|---|---:|---:|
| short probes | 19/20 | +0.2374 |
| long records | 19/20 | +0.0785 |

Every distance compresses by roughly 3x — shared boilerplate dominates, as
expected — and **the inversion is completely unchanged at 19 of 20**. A
contradicting record now sits at mean distance 0.0175 against an unrelated
baseline of 0.5875: **2.98%**, down from 8.9% on the short probes.

The honest reading: length does not cause the problem and does not cure it. What
it does is compress the entire usable dynamic range, so any absolute distance
threshold an operator might tune on short text will be wrong on long records,
and the margin separating "correct memory" from "its contradiction" shrinks to
0.006 in absolute terms. This is a warning about threshold tuning, not a second
independent failure.

## Structural claims from the tool

### `regions`

```bash
uv run embeddings regions --atlas docs/studies/atlas-plain
```

Params: `algorithm=kmeans, k=5, runs=5, resamples=20, max_iterations=100,
tolerance=1e-06`, seed 0.

| Region | Size | Stability | Asserted |
|---:|---:|---|---|
| 0 | 38 | unstable | no |
| 1 | 35 | moderate | yes |
| 2 | 19 | unstable | no |
| 3 | 17 | unstable | no |
| 4 | 11 | unstable | no |

**Partition verdict: `unstable`, ARI 0.1935** across 5 seed reruns and 20
bootstrap resamples. Per section 3 of the design note, the partition is
therefore *not* asserted as structure in this space, and neither is any region
but region 1.

One composition fact is worth recording with that caveat firmly attached: **18
of 20 fact families sit entirely within a single region** — all five variants of
a fact, including both contradictions, cluster together. This is consistent with
everything above, but because it is derived from an unstable partition it is
**not asserted**; the `neighbors --id` measurement in Result 1 makes the same
point without depending on any clustering parameter, and that is the version to
rely on.

The 2-of-5 region split between controls and facts (regions 3 and 4 hold the 20
controls) confirms the corpus does contain the coarse structure it was built to
contain.

### `gaps`

```bash
uv run embeddings gaps --atlas docs/studies/atlas-plain --probe \
    --probes docs/studies/probes-heldout.txt
```

Params: `k=5, sparsity_quantile=0.25, void_factor=2.0, coherence_factor=0.75`,
seed 0. Probes: 10 held-out statements in the same fact slots, none in the
corpus.

- **corpus coverage — 7 findings.** Two sparse regions and five inter-region
  voids. Every one carries `asserted: False` because the underlying partition is
  unstable, and every one carries its own "does not show" clause.
- **model representation — 0 findings.** The probe step ran and returned
  `inconclusive [probes landed elsewhere]` for both sparse regions: the probes
  landed nearer other regions.

**No model-capability claim is made from `gaps` in this study.** The probes were
constructed to test contradiction, not to be fair instances of a sparse region,
so "inconclusive" is the correct outcome and the correct thing to report. The
corpus-coverage findings say something about a 120-item synthetic probe corpus
and nothing whatsoever about the model — recording them here as corpus facts,
per section 1 of the design note, which makes conflating the two the highest-stakes
error available.

### `compare`

**Not available.** The gateway serves exactly one embedding model
(`Qwen/Qwen3-Embedding-0.6B`); the reranker is not on the embeddings route. Any
cross-model number in this study would be fabricated, so there is none. The
question "would a different embedder do better on this?" is **unanswered here**
and is the single highest-value follow-up.

### Float variance, measured rather than asserted

`survey` writes a live-embedding float-variance limitation into every Atlas
manifest here. Its actual magnitude on this corpus was measured by surveying
`probe-corpus.jsonl` a second time through a cold, separate cache:

| Check | Result |
|---|---|
| `vectors.npy` byte-identical between runs | **No** — differ at byte 274,561 |
| Vectors differing in any component | 1 of 120 |
| Max absolute component delta | 7.451e-09 |
| Min cosine between run-1 and run-2 vectors | 0.9999998808 |
| Any cell mean changed | No — identical to 4 decimal places |
| Inversions changed | No — 19/20 both runs |
| `regions` partition verdict changed | No — `unstable` both runs, same 5 region sizes |

So the limitation is **real but did not move a single conclusion in this
study**. That is a weaker perturbation than the t22 run saw, where float noise
moved a region's stability verdict across a threshold. Both observations are
consistent: the effect is tiny and only matters when a value sits near a
decision boundary. Nothing in this study sits near one, except the
`directions discover` stability verdicts, which are already labeled unstable.

## What this means for eidetic

Stated as recommendations, with the confidence each deserves.

1. **Do not rely on vector distance alone to choose between a record and its
   contradiction.** This is the well-supported one. Whatever ranks first among
   near-duplicates is being chosen on wording, not on truth.
2. **Prefer non-semantic signals for supersession.** `eidetic` already has
   `supersedes`, `sweep`/shadowing, and freshness signals. This study is a
   strong argument for treating those as *primary* rather than as tie-breakers:
   they carry exactly the information the embedding does not.
3. **Adopt the instruction prefix in its documented asymmetric form** — measured
   improvement, no measured cost — while **not** treating it as a fix.
4. **A reranker is the natural place to put the missing discrimination.** The
   gateway already serves `Qwen/Qwen3-Reranker-0.6B`. A cross-encoder sees both
   texts jointly and is not subject to the bi-encoder bottleneck this study
   measures. **This is a hypothesis, not a finding — it was not tested here.**
5. **Do not tune an absolute distance threshold on short text.** Result 5 shows
   the usable range compresses roughly 3x at realistic record length.
6. **The hybrid default is doing more work than it appears to.** `recall`'s
   default blends vector and BM25. Since the vector half is largely tracking
   lexical overlap anyway, the two halves may be less complementary than the
   design assumes. **Untested here** and worth measuring directly.

## Limitations

Every entry below is populated from what was actually measured in this study.

- **Twenty facts, one author, synthetic probes.** n=20 per cell is enough to
  report a distribution and far too few to characterize a model. Worse, the
  probes were written by the same author who framed the hypothesis, after seeing
  the pilot. The factorial design constrains that bias — a matched design cannot
  be steered toward one cell without visibly unbalancing lexical overlap, and
  the measured overlaps confirm the cells are matched — but it does not
  eliminate it. Nothing here is a validated benchmark.
- **One model, no cross-model comparison.** Only one embedding model is served
  on this gateway. Nothing in this study establishes whether these numbers are
  characteristic of Qwen3-Embedding-0.6B specifically, of small bi-encoders
  generally, or of embedding retrieval as such. `compare` was not run and no
  cross-model number appears here.
- **Distributions overlap; the separation is not clean.** 10% of contradictions
  are further from their base than the nearest paraphrase is. Statements of the
  form "contradictions are closer than paraphrases" are true of the
  distributions and false of about one pair in ten.
- **The lexical-overlap confound is settled for this design, not in general.**
  Token-level Jaccard is a crude overlap measure and does not capture subword
  tokenization, which is what the model actually sees. A confound correlated
  with Jaccard but not identical to it could survive this design.
- **The direction results rest on 10 pairs each.** 7 train, 3 held-out. The
  negation-versus-value asymmetry reproduced at all five seeds tried, but three
  held-out pairs is a thin basis for "generalizes", and the direction's own
  stability measure is `unstable` (ARI 0.479).
- **Live-embedding float variance applies to all three Atlases.** All were
  written from live embeddings, so all carry the limitation in their manifests.
  Measured here at max component delta 7.45e-09 affecting 1 vector of 120, and
  it moved no conclusion — but re-running `survey` will not reproduce the
  vectors bit-for-bit, and only the committed Atlases reproduce the numbers
  exactly.
- **Real eidetic records were used for shape, not as a corpus.** 18 records is
  too few to survey, and their content is private. They informed the length and
  token-density conditions and validated that the probes resemble real records;
  no measurement in this study was computed over them.
- **The length condition uses uniform padding.** Every long record shares
  identical boilerplate, which maximizes the compression effect. Real records
  vary, so the 3x compression is an upper bound on what varied real text would
  show.
- **`gaps` produced no model-representation finding.** No claim about model
  capability is drawn from it. The corpus-coverage findings describe a synthetic
  120-item probe corpus and generalize to nothing.
- **The Atlas manifests' `uncertain[]` arrays are empty.** `survey` writes
  `limitations[]` but does not propagate reader-verb findings into `uncertain[]`,
  so the uncertain items live in the next section of this document rather than
  in the committed manifests. That is a tooling gap, not an assertion that
  nothing is uncertain.
- **No verb emits its reproducing command** (finding F2). Every command here was
  transcribed by hand and is therefore only as reliable as that transcription.

## Uncertain

Open questions this study raises and does not answer.

- **Is the negation direction real?** It generalizes at every seed tried
  (held-out cosine 0.241, agreement 1.000) but its own stability measure is
  `unstable` (ARI 0.479) on 10 pairs. Treat as a promising signal requiring
  50-plus pairs to confirm.
- **Does the value/negation asymmetry hold beyond these facts?** The value pairs
  here vary ports, versions, hashes and dates. Whether numeric substitution
  behaves like hash substitution is untested — they were pooled.
- **Would a reranker recover the distinction?** The most consequential open
  question, and the cheapest to answer: `Qwen/Qwen3-Reranker-0.6B` is already
  served.
- **Would a larger or different embedder change the picture?** Unanswerable on
  this machine.
- **Is the 15-to-5 nearest-neighbor split signal or noise?** Within matched
  surface form the model prefers the paraphrase 15 times of 20. Consistent with
  the small positive meaning effect, but 20 trials cannot separate that from
  chance.
- **Does subword tokenization explain the residual?** Whether the +0.082 partial
  correlation is propositional content or a subword-level surface effect Jaccard
  cannot see is not determined here.
- **The `regions` partition is unstable at every parameter setting tried.**
  Whether this 120-item probe corpus has any stable large-scale structure is
  unresolved; only region 1 reached `moderate`, and no `k` sweep was run.
- **Whether the hybrid vector-plus-BM25 default compounds or compensates.** If
  the vector half largely tracks lexical overlap, the two halves may correlate
  more than the design assumes. Directly measurable; not measured.

## Summary table

| Question | Answer | Confidence |
|---|---|---|
| Does the model distinguish what recall needs distinguished? | No, not reliably | High, within this design |
| Did the pilot's headline effect reproduce? | Yes — 19/20 inversions | High |
| Did the pilot's stated mechanism survive? | **No** — it is lexical, not semantic | High |
| Ratio of wording effect to meaning effect | 9.7x | High |
| Does the model represent contradiction at all? | Yes, weakly and consistently | Moderate |
| Does instruction-prefixing fix it? | No; improves signal ~3x, inversion survives | High |
| Is contradiction one failure mode or two? | Two — negation has a direction, value has none | Moderate (10 pairs each, 5 seeds) |
| Does realistic record length help? | No; margins compress ~3x | Moderate |
| Would another model do better? | **Unknown** — one model served | Not tested |
| Would a reranker fix it? | **Unknown** — hypothesis only | Not tested |
