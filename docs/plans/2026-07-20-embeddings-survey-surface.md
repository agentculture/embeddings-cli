# Build Plan — embeddings survey surface

slug: `embeddings-survey-surface` · status: `exported` · from frame: `embeddings-survey-surface`

> embeddings-cli ships its survey surface: 'embeddings survey' embeds a corpus through an OpenAI-compatible backend (defaulting to the local model-gear embed gear) into a persisted, typed Atlas with first-class limitations, and 'regions', 'neighbors', 'directions', 'gaps', 'compare', and 'map' read measured, reproducible, parameter-and-seed-stamped evidence back out of it.

## Tasks

### t1 — Add numpy as the first runtime dependency in pyproject.toml

- covers: c5, h5
- acceptance:
  - [project.dependencies] lists numpy and nothing else; uv sync succeeds and uv.lock updates
  - the full CI matrix passes with the dependency present, including the TestPyPI dry-run publish on the PR

### t2 — Write the design note docs/design-notes/semantic-measurement.md answering issue unknowns 1-3

- covers: c9, h8, c37, h24, c24, h16
- acceptance:
  - the note has a section per unknown: gaps semantics (probe-based, dual-labeled), compare method (k-NN Jaccard + rank correlation, no shared-frame numbers), stability policy (params+seed, N-seed reruns + bootstrap, 'stability unverified' label)
  - a determinism-scope section states byte-identity holds for reader verbs over a fixed Atlas and cache-served vectors, and that live re-embedding may vary at float precision
  - a baseline section records the pre-work state verified at commit be4f6e8: six introspection verbs, no corpus/embedding/survey code, no disk writes, dependencies=[]
  - markdownlint-cli2 passes on the new file

### t3 — Define the typed Atlas schema in embeddings/atlas/schema.py

- covers: c8
- acceptance:
  - the manifest dataclass/TypedDict requires schema_version, model, endpoint, params, seed, limitations[] and uncertain[]; constructing one without limitations or uncertain raises
  - a test asserts limitations and uncertain are non-optional schema fields, not conventions
  - SCHEMA_VERSION is a module-level constant the writer stamps and the reader compares against

### t4 — Add atomic write-temp-then-rename helpers in embeddings/_atomic.py

- covers: c34, h21
- acceptance:
  - write_atomic(path, bytes) and atomic_dir(path) materialize only complete results; a simulated interruption mid-write leaves no partial file and no partial directory
  - a test that raises between batches asserts the target path does not exist afterwards
  - concurrent writers of identical content-addressed payloads both succeed with no corrupt result

### t5 — Build the OpenAI-compatible embedding client in embeddings/backends/client.py

- covers: c7, h6, c40, h27
- acceptance:
  - an unreachable endpoint raises CliError with code 2, a message naming the endpoint and a hint naming the override flag/env; no traceback leaks
  - a test pointing the client at a reserved dead port asserts exit 2 and the hint text
  - the only outbound host is the configured endpoint — asserted by a transport-level test that fails on any other host; no telemetry calls exist
  - the endpoint flag's help text states that a non-local endpoint ships corpus content off-machine
  - the transport is injectable so tests never open a real socket

### t6 — Add the corpus loader in embeddings/corpus.py

- acceptance:
  - each JSONL line requires a 'text' field; 'id' is optional and defaults to the line index; all other keys are preserved verbatim as item metadata
  - a line missing 'text' fails with CliError code 1 naming the line number
  - a round-trip test asserts extra keys survive into atlas records unchanged

### t7 — Retarget AGENTS.colleague.md to the Surveyor resident prompt

- covers: c20, h13, c22, h14
- acceptance:
  - the prompt describes the Surveyor role: run measured experiments through the CLI, never interpret single dimensions, record evidence-backed Atlases with limitations
  - embeddings doctor still reports healthy true; culture.yaml suffix stays embeddings-cli and backend stays colleague
  - the change is additive to the prompt file only — CLAUDE.md is not re-scaffolded and acceptance criteria 1 and 2 are not re-opened
  - the prompt is sufficient to drive the CLI without out-of-band knowledge: it points at learn and explain rather than restating them

### t8 — Build the hermetic test harness in tests/_fakes.py

- covers: c38, h25
- acceptance:
  - a fake embedding transport returns deterministic vectors for given texts, with no socket opened
  - the full pytest suite passes on a machine with nothing listening on any embedding endpoint — this is what CI itself proves on every PR
  - the dead-port failure test uses a reserved local port rather than the real gateway port

### t9 — Build the home-dir embed cache in embeddings/backends/cache.py

- depends on: t4, t5
- acceptance:
  - cache entries live under ~/.embeddings/cache keyed by (model, sha256(text)), shared machine-wide and never inside a repo
  - each entry is written atomically via embeddings/_atomic.py; a killed run leaves no unreadable entry
  - a cache hit returns byte-identical vectors to what was stored; the cache path is overridable by flag/env for tests

### t10 — Build Atlas read/write in embeddings/atlas/io.py with a fail-closed version gate

- depends on: t3, t4
- covers: c33, h20, c36, h23
- acceptance:
  - write_atlas persists atlas.json (manifest), records.jsonl, and .npy vector sidecars into an atlas/ directory, atomically and path-overridable
  - the shipped .gitignore excludes atlas vector sidecars (*.npy); after a default survey --apply inside a repo, git status shows no .npy files while manifest, records, and measurements are committable
  - read_atlas on an atlas whose schema_version is unknown or newer exits 2 with a hint naming the version mismatch and the writing CLI version — it never best-effort-parses
  - a test bumps schema_version beyond supported and asserts exit 2 plus the mismatch hint

### t11 — Build seed-resampling stability measurement in embeddings/measure/stability.py

- depends on: t1
- covers: c11, h10
- acceptance:
  - every measurement result carries params and seed keys; stability is agreement across N seed reruns plus bootstrap resampling (ARI for clusterings)
  - a single-run result is labeled 'stability unverified' rather than reported as a fact
  - re-running a reader verb over a fixed Atlas with identical inputs and seed produces byte-identical output — asserted by test, scoped per the design note's determinism section

### t12 — Implement the survey verb in embeddings/cli/_commands/survey.py

- depends on: t6, t9, t10
- covers: c2, h2, c10, h9, c35, h22
- acceptance:
  - survey registers via register(sub) at the marked spot in _build_parser(), raises only CliError, and declares its own --json
  - teken cli doctor . --strict stays green, and a parse error inside survey renders error:/hint: with exit 1
  - without --apply survey previews corpus size, backend, and target path and performs zero filesystem writes — asserted by a test that snapshots the tree
  - with --apply onto an existing atlas it exits 1 with a hint naming --force; with --force the old atlas is replaced whole, never merged
  - embedding runs in configurable batches with transient-failure retry and stderr progress diagnostics, so a looping agent sees liveness

### t13 — Add the shared reader-verb atlas entry path in embeddings/cli/_commands/_atlas_arg.py

- depends on: t10
- covers: c39, h26
- acceptance:
  - a reader verb with no atlas at the resolved path exits 1 with hint 'run embeddings survey <corpus> --apply first, or pass --atlas <path>'
  - the same reader with the gateway down but an atlas present succeeds — reader verbs never require the gateway
  - both paths are asserted by test, and exit 1 (nothing surveyed) is distinguishable from exit 2 (environment broken)

### t14 — Implement the regions verb in embeddings/cli/_commands/regions.py

- depends on: t13, t11
- acceptance:
  - regions clusters the Atlas into semantic regions and emits cluster membership, size, and representative items
  - output carries params and seed and a stability measure per region; --json shape is asserted by test
  - a region that survives only one parameter setting is reported as a finding about the parameters, not a fact

### t15 — Implement the neighbors verb in embeddings/cli/_commands/neighbors.py

- depends on: t13
- acceptance:
  - neighbors <text> embeds the query and returns its k nearest Atlas items with distances and a boundary description
  - it accepts an Atlas item id as an alternative to free text, requiring no gateway call in that case
  - --json shape asserted by test; params and seed recorded

### t16 — Implement the directions noun group in embeddings/cli/_commands/directions.py

- depends on: t13, t11
- covers: c3, h3
- acceptance:
  - the group exposes both 'directions discover' and 'directions overview', copying cli.py including parser_class=type(p) on add_subparsers
  - 'embeddings directions' with no sub-verb prints the noun overview and exits 0; 'directions overview --json' returns the sections payload
  - discover proposes candidate directions and reports whether each generalizes to held-out pairs, with params, seed, and a stability measure

### t17 — Implement the gaps verb in embeddings/cli/_commands/gaps.py — probe-based and dual-labeled

- depends on: t13, t15
- acceptance:
  - every finding is labeled either 'corpus coverage' (from the surveyed corpus) or 'model representation' (only from actively probing a sparse region with generated or held-out inputs embedded at probe time)
  - the two kinds are never merged into one count or one list, and the output states in prose that corpus sparsity alone says nothing about model capability
  - a test asserts that with probing disabled, zero model-representation findings are emitted

### t18 — Implement the compare verb in embeddings/cli/_commands/compare.py

- depends on: t13
- acceptance:
  - compare reports k-NN set overlap (Jaccard) and rank correlation on a shared probe set embedded by both models
  - no output value assumes a shared coordinate frame; raw cross-model coordinates and cosines are never emitted as comparable
  - the assumptions of the method are stated in the output, and params and seed are recorded

### t19 — Implement the map verb in embeddings/cli/_commands/map.py

- depends on: t13, t14
- acceptance:
  - map --html writes a self-contained single-file HTML artifact (no external requests, both light and dark themes) following the league-of-agents replay precedent
  - the projection's algorithm, params, and seed are visible in the artifact itself, with the parameter-dependence caveat stated
  - map without --html emits a text or JSON summary; --apply gating matches the other write verbs

### t20 — Move all seven verbs from planned to implemented across the four roadmap surfaces

- depends on: t12, t14, t15, t16, t17, t18, t19
- covers: c4, h4
- acceptance:
  - learn.py (_TEXT and planned_commands), overview.py (_PLANNED_VERBS), explain/catalog.py, and README.md all move the same seven verbs in one change — this task is the sole owner of those four files to avoid same-wave merge collisions
  - explain/catalog.py gains a self-contained entry per verb; explain resolves every new verb path
  - a test checks learn --json's commands and planned_commands lists against the registered parser, so the two can never drift again

### t21 — Guard the stable contract and the CI gates

- depends on: t12
- covers: c15, h12, c12, h11
- acceptance:
  - _errors.py, _output.py, and cli/__init__.py semantics are unchanged: the existing introspection test suite passes unaltered throughout the work
  - a test asserts results go to stdout and errors/diagnostics to stderr with the hint: prefix in both text and --json mode, for every new verb
  - coverage stays at or above the fail_under=60 gate and the full CI matrix (tests, lint, bandit, markdownlint, rubric gate, version-check) is green

### t22 — Prove the end-to-end path against the live lobes gateway

- depends on: t20, t21
- covers: c1, h1, c23, h15, h7
- acceptance:
  - all seven verbs are callable and each produces measured output from a real corpus surveyed through the lobes gateway
  - the Atlas round-trips: survey writes it, a reader verb loads it and reproduces the written measurements identically
  - every structural output is parameter-and-seed-stamped, supports --json, and cites the command that reproduces it
  - this run is operator-machine evidence recorded in the study, not a CI assertion — CI stays hermetic

### t23 — Ship the first study: an Atlas of the eidetic recall embedder

- depends on: t22
- covers: c26, h18, c25, h17
- acceptance:
  - the committed Atlas answers whether Qwen3-Embedding-0.6B distinguishes what eidetic recall needs distinguished
  - every structural claim in it carries its reproducing command, params, seed, and stability measure
  - it demonstrates at least one model-specific finding a retrieval benchmark alone would not surface
  - its limitations[] and uncertain[] sections are populated from real measurement, not boilerplate

## Risks

- [unknown_nonblocking] lobes gateway per-request batch/token limits and the real magnitude of embed float variance are unmeasured — the gateway was down throughout scoping and challenge. survey's batch size stays configurable and t22's live run measures both. (task t22)
- [unknown_nonblocking] wave-4 reader verbs are file-disjoint by construction (one _commands/*.py each), but all of them register in cli/__init__.py's _build_parser. That one shared line-range is the merge choke point: the merging agent applies registrations serially and re-runs the rubric gate after each.
- [follow_up] clustering/projection algorithm choice (k-means vs HDBSCAN, UMAP vs t-SNE) is not fixed by the spec — t11's stability contract holds regardless, but t14/t19 authors must pick and record one, and a later change reshapes committed study evidence.
