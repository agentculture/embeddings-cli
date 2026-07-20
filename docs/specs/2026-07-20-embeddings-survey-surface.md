# embeddings survey surface

> embeddings-cli ships its survey surface: 'embeddings survey' embeds a corpus through an OpenAI-compatible backend (defaulting to the local model-gear embed gear) into a persisted, typed Atlas with first-class limitations, and 'regions', 'neighbors', 'directions', 'gaps', 'compare', and 'map' read measured, reproducible, parameter-and-seed-stamped evidence back out of it.
> instruction: implement survey first (it writes the Atlas the rest read), then reader verbs in dependency order; keep the four planned-verb listings in lockstep as each ships

## Audience

- The Surveyor agent (the colleague resident this repo declares), the operator driving studies, and agent readers of learn/explain — plus sibling lanes (eidetic-cli) consuming study results.

## Before → After

- Before: Today the CLI is the renamed template skeleton: six introspection verbs, no corpus/embedding/survey code, no disk persistence, dependencies=[] — the seven survey verbs exist only as planned-verb listings.
- After: An operator or the Surveyor agent runs 'embeddings survey corpus.jsonl' against the lobes gateway and gets a persisted Atlas; regions, neighbors, directions, gaps, compare, and map read measured evidence back out of it; every structural output is parameter-and-seed-stamped, supports --json, and cites the command that reproduces it.

## Why it matters

- Embedding models get picked on retrieval benchmarks and hidden behind vector databases as interchangeable infrastructure; they are not — each defines its own system of similarity and distinction. This CLI makes those differences visible, measurable, and reproducible.

## Requirements

- New verbs register as modules under embeddings/cli/_commands/ via register(sub) at the marked spot in _build_parser(); handlers raise CliError only (dispatch wraps everything else), and every sub-verb declares its own --json since argparse does not inherit the parent flag.
  - honesty: teken cli doctor --strict stays green after each new verb registers, and a parse error inside any new verb still renders error:/hint: with exit 1
- 'directions discover' makes directions a noun with action-verbs, so the group must also expose 'directions overview', copying cli.py including parser_class=type(p) on add_subparsers so nested parse errors keep the structured error contract.
  - honesty: 'embeddings directions' with no sub-verb prints the noun overview and exits 0, and 'directions overview --json' returns the sections payload — same shape tests as cli overview
- Each verb that ships moves from planned to implemented in all four surfaces that list the roadmap — learn.py (_TEXT and planned_commands), overview.py (_PLANNED_VERBS), explain/catalog.py (plus a new self-contained entry per verb), and README.md — the planned/implemented separation is load-bearing for agent readers.
  - honesty: after each verb ships, learn --json lists it under commands and not planned_commands, and explain resolves its path — asserted by a test that checks the two lists against the registered parser
- The first runtime dependency lands: pyproject dependencies=[] today and numpy is the floor for clustering/directions/density; nothing in CI enforces zero-deps (teken cli doctor --strict checks CLI shape, not dependencies), and publish.yml is path-filtered to pyproject.toml and embeddings/** so the dep change also rides the PyPI publish path on merge.
  - honesty: the first survey-surface PR adds numpy (only) to [project.dependencies], uv sync and CI pass, and the TestPyPI dry-run publish succeeds with the dependency present
- An unreachable embed backend is an environment failure: CliError code 2 with a remediation naming the endpoint and how to start or override the gear — the gear is down on this machine right now, so backend-down is the day-one failure path, and no traceback may leak.
  - honesty: with nothing listening on the configured endpoint, every embedding-touching verb exits 2 with error: naming the endpoint and hint: naming the override — covered by a test pointing the backend at a dead port
- The Atlas is a persisted, typed artifact with first-class limitations/uncertain sections (issue AC-4): survey writes it and regions/neighbors/directions/gaps/compare/map read it; nothing in the package writes to disk today, so format and location are greenfield.
  - honesty: the Atlas round-trips: survey writes it, a reader verb loads it and reproduces the written measurements identically; limitations and uncertain are required schema fields a test asserts non-optional
- A design note lands in docs/ answering issue unknowns 1-3 (gaps semantics, cross-model comparison method, projection/cluster stability) before the corresponding verbs are implemented (issue AC-3).
  - honesty: docs/ contains the design note answering gaps semantics, compare method, and stability policy, merged no later than the PR implementing the first of gaps/compare/regions
- Every write verb is dry-run by default with --apply to commit (issue AC-5, AgentCulture rule — agents call CLIs in loops); survey, the primary writer, previews corpus size and backend without --apply.
  - honesty: a test runs survey without --apply and asserts zero filesystem writes; with --apply it asserts the Atlas exists
- Every measurement output records its parameters and seed, and structural claims (a region, a direction, a gap) carry a stability measure rather than being reported as fact — UMAP/t-SNE layouts and k-means/HDBSCAN partitions shift substantially with parameters and seeds (issue unknown 3).
  - honesty: every structural verb's --json output carries params and seed keys, and re-running with identical inputs and seed produces identical output — asserted by test
- New verbs ship with tests including --json shape tests in the pattern of test_cli_introspection.py, and keep coverage above the fail_under=60 gate CI enforces via pytest --cov.
  - honesty: coverage stays at or above 60 and the full CI matrix (tests, lint, rubric gate, version-check) is green on every survey-surface PR
- Refines decision c30: the committable Atlas is the interpretation, not the bulk — atlas.json manifest, JSONL records, and measurements are committed; .npy vector sidecars are regenerable bulk (cache- or backend-rebuildable via survey --apply) and are excluded from git by a shipped .gitignore entry. A 10k-item corpus at 1024 dims is ~40MB of float32 per model — committing vectors would bloat the repo and contradict the Atlas-is-not-the-space distinction the issue marks load-bearing.
  - honesty: after a default survey --apply inside a repo, git status shows no .npy files; reader verbs still work locally from the sidecars; the committed subset alone lets a reviewer audit every structural claim
- All disk writes are atomic (write-temp-then-rename): the atlas directory materializes only complete, and each embed-cache entry is written atomically — a killed survey never leaves a half-written atlas or a corrupt cache entry, and concurrent surveys racing on the content-addressed cache are benign (same key, same content). survey batches gateway requests with a configurable batch size, retries transient failures, and emits progress diagnostics to stderr so looping agents see liveness.
  - honesty: a survey interrupted mid-run leaves no partial atlas dir and no unreadable cache entry — asserted by a test that simulates interruption between batches
- survey --apply refuses to overwrite an existing atlas at the target path: exit 1 with a hint naming --force; only --force replaces it (atomically). Prior committed study evidence is user data — silent clobbering by an agent in a loop is the data-loss path the c19 escalation named.
  - honesty: survey --apply onto an existing atlas exits 1 with a hint naming --force; with --force the old atlas is replaced whole, never merged — asserted by test
- The atlas.json manifest's schema_version is fail-closed: a reader verb pointed at an atlas with an unknown or newer schema_version exits 2 with a hint naming the version mismatch and the CLI version that wrote it — it never best-effort-parses. Old CLIs meeting new atlases is the lifecycle path the frame had no claim for.
  - honesty: a reader on an atlas with schema_version bumped beyond supported exits 2 with the mismatch hint — asserted by test
- The test suite is hermetic: CI has no lobes gateway, so no test may open a real socket to the embedding endpoint — the backend is faked via an injected transport/stub, and the dead-port failure test (h6) uses a reserved local port. The live end-to-end run (h15) is operator-machine evidence, recorded in the first study, not a CI assertion.
  - honesty: the full pytest suite passes on a machine with nothing listening on any embedding endpoint — verified by CI itself on every PR
- A reader verb with no atlas at the resolved path is a user error: exit 1 with hint 'run embeddings survey <corpus> --apply first, or pass --atlas <path>' — distinct from exit 2 (gateway down), so an agent can tell 'nothing surveyed yet' from 'environment broken'.
  - honesty: each reader verb run without an atlas exits 1 with the survey-first hint; the same verb with the gateway down but an atlas present succeeds (readers never need the gateway) — both asserted by test

## Honesty conditions

- the seven verbs exist, are callable, and each produces measured output from a real corpus through the lobes gateway — demonstrated end-to-end by the first study
- no survey-surface PR alters _errors.py/_output.py semantics — the existing introspection test suite (structured error shape, stdout/stderr split) passes unchanged throughout
- doctor stays healthy across the work; CLAUDE.md and the prompt files change only additively (the Surveyor retarget), never re-scaffolded
- learn/explain stay self-teaching for an agent reader (rubric-gated), and the Surveyor prompt can drive the CLI without out-of-band knowledge
- the end-to-end path works on this machine: survey a corpus through the lobes gateway, then each reader verb returns measured output from the persisted Atlas
- verified during scoping at commit be4f6e8: no corpus/embedding/survey code, no disk writes, dependencies=[]
- the first study (the eidetic-embedder Atlas) demonstrates at least one model-specific finding a retrieval benchmark alone would not surface
- the eidetic-embedder Atlas is committed with every structural claim carrying its reproducing command, params, seed, and stability measure
- a corpus line missing 'text' fails survey with exit 1 naming the line number; extra keys round-trip into atlas records unchanged — asserted by test
- the design note documents the determinism scope; every atlas written from live (non-cached) embeddings carries the float-variance limitation entry; reader-verb byte-identity is asserted by test
- code review + a test over the HTTP layer confirm the only outbound host is the configured endpoint; the endpoint flag's help text names the off-machine implication

## Success signals

- The first real study ships: an Atlas of the eidetic recall embedder (Qwen3-Embedding-0.6B via the lobes gateway) answering whether it distinguishes what recall needs distinguished — every claim in it traceable to a re-runnable embeddings command.

## Scope / boundaries

- _errors.py, _output.py, and cli/__init__.py are stable-contract: the survey surface extends them, never alters their semantics — results to stdout, errors/diagnostics to stderr with the hint: prefix, exit codes 0/1/2 with 3+ reserved, no traceback ever.
- Acceptance criteria 1 and 2 are already satisfied on disk and must not be re-opened: CLAUDE.md was retargeted to the real domain in 0.7.0 (commit be4f6e8) and backend colleague is consistent — AGENTS.colleague.md present, doctor reports healthy true. Remaining criteria are 3-6.
- Determinism claims are scoped: byte-identical reproducibility (h10) is guaranteed for reader verbs over a fixed Atlas and for cache-served vectors; a fresh re-embedding of the same text may vary at float precision (GPU/vLLM nondeterminism), so survey-level reproducibility is cache-scoped, and the Atlas limitations[] section states this automatically.
- Network egress is exactly one host: corpus text is sent only to the explicitly configured embeddings endpoint (default localhost lobes gateway); no telemetry, no other requests. Pointing at a remote provider is an explicit operator choice that ships corpus content off-machine — the help text for the endpoint flag says so.

## Non-goals

- embeddings-cli does not become a vector database or retrieval engine — eidetic-cli owns the memory store and already does vector recall (--mode approximate against the same embed gateway); the natural relation is evaluator: does eidetic's embedder distinguish what recall needs distinguished?
- No model serving: the lobes gateway (model-gear repo) hosts local models including the embed gear; embeddings-cli consumes its HTTP surface (POST /v1/embeddings) and never manages model processes.
- No judgment lane and no storage-data-quality lane: coherence-cli owns information-quality judgment, data-refinery-cli owns data quality in storage/retrieval; embedding-space coverage stays inventory and measurement here.
- The tool never assigns human-readable meaning to individual dimensions — the issue and CLAUDE.md both mark per-dimension interpretation unstable and misleading; the unit of interpretation is the space's structure (regions, directions, boundaries).

## Assumptions

- The first embedding backend is OpenAI-compatible /v1/embeddings only, defaulting to model-gear embed gear (Qwen/Qwen3-Embedding-0.6B at <http://localhost:8002/v1>) — the same endpoint eidetic-cli approximate/hybrid recall already defaults to — with endpoint and model overridable via flags/env in the EIDETIC_EMBED_URL style.
- Embedding results are cached on disk keyed by (model, text-hash) — for a bulk-embedding survey tool the cache is effectively not optional (issue unknown 4), and it makes compare and re-survey cheap.
- map --html ships as a self-contained single-file HTML artifact following league-of-agents' replay precedent — storybook-cli, the issue's suggested consumer, is itself an unimplemented template skeleton today, so integration with it is deferred until it exists.
- Corpus JSONL schema: each line needs a 'text' field; 'id' is optional (defaults to the line index); all other keys are preserved as item metadata in the Atlas records. The issue sketch names only 'survey corpus.jsonl' — the field contract is otherwise unstated anywhere.

## Scope exploration

- `s1` — `embeddings/cli/__init__.py (_build_parser, _dispatch)`: registration point is marked at __init__.py:94; _dispatch wraps non-CliError exceptions so no traceback leaks; parse errors route through _CliArgumentParser with the --json argv pre-scan
  - seeds: `c2`
- `s2` — `embeddings/cli/_commands/cli.py`: cli.py exists solely to satisfy the rubric overview_cli_noun_exists check and passes parser_class=type(p) at cli.py:40 — the pattern a directions noun group must copy
  - seeds: `c3`
- `s3` — `learn.py:49-57,91-99 + overview.py:38-46 + explain/catalog.py:48-51 + README.md:144`: the seven roadmap verbs are named under explicit not-yet-implemented headings in four places that must move in lockstep as each verb ships
  - seeds: `c4`
- `s4` — `pyproject.toml + .github/workflows/publish.yml + tests.yml:75`: dependencies=[] is an inherited template property, unenforced by any CI job; the rubric gate at tests.yml:75 checks CLI shape only
  - seeds: `c5`
- `s5` — `model-gear CLAUDE.md:37 + lobes/catalog.py:141 + eidetic-cli README.md:77-78`: model-gear serves Qwen/Qwen3-Embedding-0.6B at POST /v1/embeddings; eidetic-cli defaults recall embedding to <http://localhost:8002/v1> with env-var overrides
  - seeds: `c6`
- `s6` — `http://localhost:8002/v1 (live probe) + embeddings/cli/_errors.py`: curl to localhost:8002 was refused during scoping — backend-down is the common case, not the edge; the exit-code policy reserves 2 for environment/setup errors
  - seeds: `c7`
- `s7` — `embeddings/ package (no persistence today) + issue #1 AC-4`: no module under embeddings/ performs any disk write; Atlas storage format and location are open decisions the design note must settle
  - seeds: `c8`
- `s8` — `docs/ + issue #1 AC-3`: docs/ contains only skill-sources.md — the design note required before gaps/compare/regions ship does not exist yet
  - seeds: `c9`
- `s9` — `issue #1 AC-5`: the dry-run rule is an explicit acceptance criterion; survey is the main write verb the rule will bite on
  - seeds: `c10`
- `s10` — `issue #1 unknown 3`: the CLI's whole value is being the reproducible half of the Surveyor loop; parameter/seed stamping is what makes every structural claim re-runnable
  - seeds: `c11`
- `s11` — `tests/test_cli_introspection.py + pyproject.toml [tool.coverage.report] + tests.yml:32`: existing tests assert JSON payload shape, the structured error format (error:/hint:, exit 1), and rubric contracts — the template new-verb tests must follow
  - seeds: `c12`
- `s12` — `issue #1 unknown 4`: the issue marks the cache 'probably not optional'; without it every parameter tweak re-embeds the whole corpus
  - seeds: `c13`
- `s13` — `operator note + model-gear checkout (lobes/ package: lobes/catalog.py:141, lobes/cli/)`: operator confirms the gateway runtime is the lobes command (lobes-cli); the model-gear repo's package dir is lobes/, whose catalog defines the Qwen3-Embedding-0.6B gear
  - seeds: `c14`
- `s14` — `embeddings/cli/_errors.py + _output.py`: the three modules are marked stable-contract; the agent-first rubric parses against their exact behavior
  - seeds: `c15`
- `s15` — `eidetic-cli README.md:40,77`: eidetic recall already ships four modes including approximate (vector) on the same embed gateway — retrieval is occupied lane territory
  - seeds: `c16`
- `s16` — `model-gear README.md:338`: the lobes gateway routes /v1/embeddings to the Qwen3-Embedding-0.6B gear — serving is its lane
  - seeds: `c17`
- `s17` — `issue #1 lane boundaries (both siblings exist as local checkouts)`: the issue states the boundary explicitly to prevent drift; coherence-cli and data-refinery-cli both exist locally as active lanes
  - seeds: `c18`
- `s18` — `issue #1 'Why you exist' + CLAUDE.md`: both framing docs make per-dimension meaning an explicit anti-goal
  - seeds: `c19`
- `s19` — `uv run embeddings doctor --json (live) + CHANGELOG.md 0.7.0 + commit be4f6e8`: doctor passes prompt_file_present and skills_present live; the 0.7.0 changelog records the CLAUDE.md retarget and the prog fix
  - seeds: `c20`
- `s20` — `storybook-cli README.md + league-of-agents README.md:65,162`: storybook-cli's CLI surface is the untouched template (introspection verbs only); league-of-agents ships a working one-file, both-themes HTML replay
  - seeds: `c21`
- `s21` — `challenge pass / data-flow lens: issue #1 verb sketch (corpus.jsonl) + spec c1`: the corpus input field contract is unstated in issue and spec; seeded the schema assumption
  - seeds: `c32`
- `s22` — `challenge pass / adjacent-systems lens: .gitignore (read live — no atlas/ or *.npy entries) + decision c30`: c30 says committable directory but the gitignore would let 40MB-per-model .npy bulk into git; the interpretation/bulk split needs to be explicit
  - seeds: `c33`
- `s23` — `challenge pass / failure-mode + concurrency lens: survey write path (greenfield) + embed cache (c13)`: nothing writes to disk today so atomicity is free to design in; content-addressed cache keys make write races benign if per-entry writes are atomic
  - seeds: `c34`
- `s24` — `challenge pass / data-loss lens: atlas/ target path under repeated agent invocation`: dry-run default (c10) covers accidental writes but not deliberate --apply onto an existing study; overwrite protection is the missing containment
  - seeds: `c35`
- `s25` — `challenge pass / lifecycle lens: atlas.json schema_version (decision c30)`: c30 names the field but no claim said what a version mismatch does; fail-closed mirrors the sibling-store precedent
  - seeds: `c36`
- `s26` — `challenge pass / counter-evidence lens: honesty h10 vs GPU/vLLM float nondeterminism`: h10 as written ('identical inputs and seed produce identical output') is unsatisfiable for live re-embedding; scoping it to fixed-atlas readers + cached vectors keeps it honest
  - seeds: `c37`
- `s27` — `challenge pass / overlooked-actor lens: tests.yml (no gateway service container) vs honesty h15`: h15 promises the end-to-end works 'on this machine' but CI is an actor with no gateway; hermetic tests reconcile the two
  - seeds: `c38`
- `s28` — `challenge pass / failure-path lens: reader verbs before any survey has run`: the frame fixed backend-down (exit 2) but had no claim for the more common first-contact failure: no atlas exists yet; also surfaces that readers must not require the gateway at all
  - seeds: `c39`
- `s29` — `challenge pass / security lens: outbound HTTP surface (greenfield) + bandit config in pyproject`: single-egress is easy to hold now and hard to retrofit; bandit runs in CI but checks code patterns, not egress policy
  - seeds: `c40`
- `s30` — `challenge pass / operations-reversibility lens: publish.yml (Trusted Publishing on push to main)`: clean pass: merge auto-publishes and a published version is immutable, but the TestPyPI dry-run on PRs is the existing containment; no new claim needed
- `s31` — `challenge pass / cheap-probe lens: teken cli doctor . --strict (live run)`: baseline rubric green before any survey verb lands — deviations after are attributable to the new work
- `s32` — `challenge pass / unexamined surface: live /v1/embeddings response shape and limits`: not probed — gateway down; shape assumed from the OpenAI-compatible contract and model-gear's tests; parked with the batch-limit unknown rather than claimed

## Decisions

- The local embedding gateway is the 'lobes' command (lobes-cli runtime) — that is what serves the OpenAI-compatible /v1/embeddings endpoint the CLI consumes; 'model-gear' is the repo, 'lobes' is the runtime/gateway.
- gaps is probe-based and dual-labeled: corpus-coverage findings come from the surveyed corpus; model-representation findings only from actively probing sparse regions with generated/held-out inputs embedded at probe time. Every finding is labeled which kind it is — the two are never conflated.
- compare v1 is relational neighborhood agreement on a shared probe set: k-NN set overlap (Jaccard) plus rank correlation. No output ever assumes a shared coordinate frame; CKA/Procrustes are follow-ups.
- stability policy v1 is seed-resampling: every structural verb records params + seed; stability is agreement across N seed reruns plus bootstrap resampling (e.g. ARI for clusterings); single-run output is labeled 'stability unverified'.
- the Atlas is a repo directory artifact: an atlas/ directory (path overridable) holding an atlas.json manifest (schema_version, model, endpoint, params, required limitations[] and uncertain[]), vectors as .npy sidecars, per-item records as JSONL — committable, diffable, shareable study evidence.
- Surveyor is the agent's persona: AGENTS.colleague.md is retargeted to the Surveyor resident prompt as part of this work; the mesh nick/suffix stays embeddings-cli.
- The embed cache lives in a home-dir store (~/.embeddings/cache), eidetic-style: keyed by (model, text-hash), shared across every repo and study on the machine, never near git; CI starts cold by design.

## Resolved vagueness

- [unknown_blocking] Cross-model comparison method for 'compare': neighborhood agreement (k-NN overlap / rank correlation), representational similarity on a shared probe set (CKA/RSA), or explicit Procrustes alignment with reported residual — models share no coordinate frame, so the method and its stated assumptions must be settled in the design note before compare ships (issue unknown 2). — resolved: compare v1 = k-NN neighborhood agreement (Jaccard set overlap + rank correlation) on a shared probe set; no shared-frame numbers; CKA/Procrustes deferred (decision c28).
- [unknown_blocking] Which stability measure structural claims carry (resampling survival, seed variation, cross-algorithm agreement) and which clustering/projection algorithms ship first — parameter-dependence is inherent (issue unknown 3); the design note must pick the measure before regions/directions/map report structure as findings. — resolved: stability v1 = seed-resampling: params+seed on every output; agreement across N seed reruns + bootstrap resampling; single-run output labeled 'stability unverified' (decision c29).
- [unknown_blocking] Atlas storage format and location: repo artifact (JSON/JSONL + assets) vs a store; single file vs directory. Blocks survey, which writes it, and everything downstream that reads it (issue unknown 5). — resolved: Atlas = repo directory artifact: atlas/ dir with atlas.json manifest (required limitations[]/uncertain[]), .npy vector sidecars, JSONL records; path overridable (decision c30).
