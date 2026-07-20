# Changelog

All notable changes to this project will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/). This project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.8.1] - 2026-07-20

Review triage for PR #3 — Qodo and SonarCloud.

### Fixed

- **A short embed response silently misaligned the Atlas.** `EmbeddingClient.embed()` returned one vector per response entry with no length check, and `get_or_embed()` paired them to inputs with `zip()`, so a backend returning a short batch attached the wrong vector to the wrong text for every entry after the gap. The contract is now enforced once, where request and response are both in scope: `data` must be a list of exactly `len(texts)` entries, each a non-empty sequence of real numbers of consistent dimension (`bool` rejected — it is an `int` in Python but not a coordinate). The cache re-checks the count *before* its first write, because a poisoned cache entry outlives the misbehaving backend and is served on every later run.
- **A missing `vectors.npy` surfaced as an `IndexError` frames from the cause.** Now distinguished deliberately: *absent* sidecars exit 1, because `.npy` is git-ignored by design and "records present, vectors absent" is the expected state of a freshly-cloned study — the hint says so, so a cloner does not go hunting a bug that does not exist. A sidecar that is *present but* malformed or too short for its records is real corruption and exits 2.
- **`atomic_dir` leaked a full scratch copy on a failed replace.** `os.replace` sat in an `else:` clause with no cleanup, so an `ENOTEMPTY` left a complete copy of the just-built atlas as a dotted directory nobody would look at. Found via a Qodo aside rather than its headline.

### Changed

- **`cmd_compare` and `_parse_pairs_file` split into named helpers** (cognitive complexity 19 and 18, over the 15 allowed) — coherent sub-concepts, not metric-gaming, with no test assertion touched.
- **31 `pytest.raises` blocks tightened** so only the statement under test can raise. A multi-statement block passes if *any* line raises, so setup could mask the assertion — a real false-green hazard in a suite whose job is guarding an error contract. No test was found to be passing for the wrong reason; this is a rename-proof guard against a hazard not yet exploited.
- **The atomic helpers no longer claim power-loss durability.** They fsync the temp file but not the parent directory after `os.replace`, which is correct for their actual scope — process death, not power loss — since a per-write directory fsync would mean thousands of synchronous flushes per survey for a property no caller needs. The docstrings now say what is promised, what is not, and where the fix would go.
- **`CLAUDE.md` described a repo that no longer exists** — it claimed the survey surface was unimplemented, that there were no runtime dependencies, and that Atlas persistence was an open question. The numpy claim in particular generated a false-positive review finding against our own stale doc.

## [0.8.0] - 2026-07-20

### Added

- **The first real study** — `docs/studies/2026-07-20-eidetic-embedder.md`, asking whether the embedder behind `eidetic recall` distinguishes what recall needs distinguished. The answer is no, and not for the reason first suspected: the model encodes a fact-versus-contradiction difference at roughly a tenth the strength with which it encodes *wording*, so a contradicting record phrased like the query outranks a true record phrased differently (19 of 20 probed facts). A matched 2x2 factorial separating meaning from lexical overlap overturned the pilot's "cannot represent negation" reading — the wording effect is 9.7x the meaning effect — which changes the fix, since choosing a model "better at negation" would not help. Instruction-prefixing roughly triples contradiction sensitivity but does not remove the inversion.
- **Live end-to-end evidence** — `docs/evidence/2026-07-20-live-end-to-end.md`, all seven verbs exercised against the running gateway over a 152-paragraph corpus, with the builder committed so every number re-runs. Notable: `regions` found no stable partition at any k and declined to assert the one it computed, and two live surveys of identical text moved a region's verdict on GPU float variance alone — the amplification the design note predicts, observed.
- **Every structural output now cites the command that reproduces it** — a `reproduced_by` key in `--json`, a `reproduce:` line in text mode, and a row in the `map --html` artifact, which is the case that matters most since that file travels alone. Credential-shaped flag values are redacted defensively.
- **The survey surface ships.** All seven verbs are implemented, registered, and callable: `survey` (embed a JSONL corpus into a persisted Atlas), `regions`, `neighbors`, `directions discover` / `directions overview`, `compare`, `gaps`, and `map`. The package gains its first runtime dependency (numpy) and its first persistence layer.
- **The Atlas** — a typed, versioned directory artifact (`atlas.json` manifest + `records.jsonl` + `.npy` vector sidecars). `limitations[]` and `uncertain[]` are required schema fields with no defaults, so a manifest that forgets to state what it does not know cannot be constructed. Readers are fail-closed: an unknown or newer `schema_version` exits 2 naming both versions, checked before any record is parsed.
- **Measurement apparatus** (`embeddings/measure/`) — Adjusted Rand Index, N-seed rerun agreement, and bootstrap resampling on numpy alone. A single-run result reports `score: null` and the literal verdict `stability unverified`, never a numeric sentinel that a threshold comparison would read as a measured-but-bad score.
- **An embedding backend** (`embeddings/backends/`) — an OpenAI-compatible client with an injectable transport, single-host egress, and a machine-wide content-addressed cache at `~/.embeddings/cache`. An unreachable gateway is a clean exit 2 naming the endpoint and the override, never a traceback.
- **Atomic writes throughout** (`embeddings/_atomic.py`) — an interrupted `survey` leaves no partial atlas and no unreadable cache entry.
- **A data-driven CLI contract guard** — walks the registered parser rather than a hand-written verb list, so every verb added later inherits the stdout/stderr, `error:`/`hint:`, and `--json` assertions automatically. A companion guard derives the roadmap surfaces from the parser too, so `learn`/`overview`/`explain`/`README` can no longer drift from what is actually callable.
- **The Surveyor resident prompt** — `AGENTS.colleague.md` retargeted from the generic template to the Surveyor role.
- **The converged spec for the survey surface** — `docs/specs/2026-07-20-embeddings-survey-surface.md`, worked backwards from issue #1 through the devague `/scope` → `/think` → `/challenge` arc. Records 20 scope entries with file/line provenance, 10 requirements each carrying a testable honesty condition, the lane boundaries and non-goals, and six operator decisions: probe-based dual-labeled `gaps`, k-NN neighborhood agreement for `compare` v1, seed-resampling stability, the Atlas as a repo directory artifact, the `lobes` gateway as the embedding backend, and Surveyor as the agent persona.
- **The buildable plan** — `docs/plans/2026-07-20-embeddings-survey-surface.md`: 23 file-disjoint tasks in 8 dependency waves, every one of the spec's 52 coverage targets claimed by a confirmed task with TDD-shaped acceptance criteria, plus three first-class plan risks.
- **Nine findings from the rigorous `/challenge` pass**, adjudicated into the spec: the corpus JSONL field contract, atomic write-temp-then-rename for all disk writes, `survey --apply` refusing to overwrite an existing atlas without `--force`, a fail-closed `schema_version` gate, hermetic tests (CI has no gateway), a distinct exit-1 path for a missing atlas, and single-host network egress.

### Fixed

- **The default embedding endpoint was wrong, and the client sent no credentials.** It defaulted to `http://localhost:8002/v1`, taken from the build brief and corroborated by eidetic-cli's README and its recall/remember skills — but all four are downstream of one stale claim, and eidetic's own code says `:8101`. The gateway publishes host port **8001** and rejects unauthenticated requests. The authoritative resolver is `lobes endpoint embedder`, now recorded beside the constant and pinned by a test. Auth is environment-only (`EMBEDDINGS_API_KEY`, falling back to `COLLEAGUE_API_KEY`) so a token cannot reach shell history or a process listing, the field is `repr=False`, and a 401/403 is classified structurally and given its own remediation — bad credentials need a different fix than a dead host.
- **Duplicate corpus record ids were accepted silently**, and `neighbors --id` then resolved an ambiguous id to whichever record loaded first. Ids are the join key for `neighbors --id`, `directions --pairs`, and compare's probe matching, so a collision made all three quietly wrong. Now refused at load time naming both the offending line and the first occurrence.

### Changed

- **Determinism claims are now scoped honestly** — byte-identical reproducibility holds for reader verbs over a fixed Atlas and cache-served vectors; live re-embedding may vary at float precision (GPU/vLLM nondeterminism), and an atlas written from live embeddings records that in its `limitations[]`.
- **The Atlas commits its interpretation, not its bulk** — the manifest, records, and measurements are committable; `.npy` vector sidecars are regenerable and git-ignored. The embed cache moves to a machine-wide home-dir store at `~/.embeddings/cache`.

## [0.7.0] - 2026-07-20

### Added

- **The project domain**, replacing the inherited `culture-agent-template` framing. `README.md` now describes what `embeddings-cli` is for: studying embedding spaces as systems — semantic regions, neighborhoods, directions, gaps, density, and cross-model differences — treating an embedding model as a semantic landscape, and serving as the deterministic tool surface for a Surveyor agent that records an evidence-backed Atlas.
- **Planned-verb sections** in `learn`, `explain`, `overview`, and `README.md` naming the target survey surface (`survey`, `regions`, `neighbors`, `directions`, `gaps`, `compare`, `map`) under explicit *not yet implemented* headings, so an agent reading the CLI never mistakes a roadmap verb for a callable one.
- `test_prog_matches_declared_console_script` — reads the console script out of `pyproject.toml` and asserts the parser's `prog` matches, guarding the pairing rather than the literal so a future rename that updates only one side fails loudly.
- `command` and `planned_commands` keys in the `learn --json` payload.

### Changed

- **`prog` is now `embeddings`**, matching the console script declared in `pyproject.toml`. Every usage, help, and error line is now copy-pasteable.
- **Retargeted the CLI's self-documentation** — `learn`, `explain/catalog.py`, and `overview` described this repo as "a clonable template for AgentCulture mesh agents" and explained how to clone it to mint a new agent. They now describe the embeddings/Surveyor domain.
- Command invocations throughout docstrings, help strings, the `explain` catalog, and the `doctor` status line now read `embeddings ...`; `embeddings-cli` is reserved for the dist/repo/nick. The `explain` catalog keys both spellings to the root entry.
- `CLAUDE.md` — replaced the roadmap placeholder with the real domain, recorded the naming convention, and noted the design constraints the survey surface will force (runtime dependencies, rubric shape of `directions discover`, Atlas persistence).

### Fixed

- **The CLI documented commands that did not exist.** `pyproject.toml` declares the binary `embeddings`, but `prog` and every help/catalog string said `embeddings-cli` — so `embeddings-cli whoami`, printed in the README quickstart and in the CLI's own `learn`/`explain` output, failed with "No such file or directory". Left over from the scaffold rename.

## [0.6.0] - 2026-07-18

### Added

- **Four devague-origin skills re-vendored into `.claude/skills/`**
  (cite-don't-import), synced to the fixed devague source
  (devague#74/#75/#76):
  - `challenge` — a risk-scaled blind-spot discovery pass that runs between
    `/think` and `/spec-to-plan`, routing findings back through the existing
    deterministic moves as human-adjudicated proposals.
  - `scope` — the idea→scope leg that surveys the surfaces an idea touches
    before framing, seeding the Announcement Frame with provenance-backed
    boundary/non-goal/assumption claims.
  - `deviate` — stops an in-flight `assign-to-workforce` run when execution
    must diverge from the confirmed plan and records the divergence as a
    first-class, append-only deviation record.
  - `summarize-delivery` — closes the loop after an `assign-to-workforce`
    run with a planned-vs-actual accountability artifact.

  These four originate in `devague` and are re-broadcast via guildmaster; see
  `docs/skill-sources.md` for provenance.

## [0.5.0] - 2026-06-24

### Added

- **Memory-discipline "Conventions and workflow" section in `CLAUDE.md`** — a
  per-task *recall-before / remember-after* convention (scope localized to this
  repo's nick) so the vendored `remember` / `recall` skills are actually used,
  not just present: `/recall` before non-trivial work to build on prior
  decisions instead of re-deriving them, and `/remember` when a non-obvious
  decision, constraint, fix-and-why, or hard-won gotcha surfaces. The section
  documents this repo's memory as **in-repo and public** — records resolve to
  `<repo-root>/.eidetic/memory` (committed, team- and mesh-shared). Inserted
  idempotently (skipped if already present), slotted under an existing
  "Conventions and workflow" heading when one exists, else appended.

### Changed

- **Refreshed the `remember` + `recall` wrappers from eidetic-cli 0.10.0**
  (cite-don't-import) — picks up eidetic's **project-local store default**: the
  files backend now resolves per record by visibility — PUBLIC records inside a
  git repo go to `<repo-root>/.eidetic/memory` (committed, team-shared), PRIVATE
  records (or any record outside a repo) go to `$HOME/.eidetic/memory` (never
  committed), an explicit `EIDETIC_DATA_DIR` still wins, and recall reads both
  stores and merges. Also carries the 0.9.3 hardening (interactive-stdin guard,
  `help` as a search term, SIGPIPE-safe suffix parsing). **Recipe policy
  override (the wrappers here are NOT byte-verbatim):** the injected default
  visibility is flipped from eidetic's `private` to **`public`**, so a plain
  `/remember` lands the note in `./.eidetic/memory` in this repo, kept as part
  of the repo — pass `--visibility private` to route a record to `$HOME`
  instead. `remember` drives `eidetic remember` (idempotent upsert of one JSON
  record or an NDJSON batch on stdin); `recall` drives `eidetic recall` with
  four search modes (exact / approximate / keyword / hybrid). Each `SKILL.md` is
  localized only in the illustrative `--scope <nick>` examples (Provenance keeps
  "First-party to eidetic-cli"). Runtime dep: the `eidetic` CLI on PATH (else a
  local eidetic-cli checkout with `uv`) — **`eidetic >= 0.10.0`** for the
  in-repo routing; on an older CLI the public records still work but are stored
  in `$HOME/.eidetic/memory` instead of in-repo. Propagated by rollout-cli's
  `eidetic-memory` recipe.

## [0.4.0] - 2026-06-23

### Added

- **Vendored the `remember` + `recall` memory skills from eidetic-cli**
  (cite-don't-import) — the write/read halves of eidetic's shared
  `~/.eidetic/memory` surface, so this agent (Claude and its colleague backend)
  can persist facts across sessions and recall them later, sharing one store.
  `remember` drives `eidetic remember` (idempotent upsert of one JSON record or
  an NDJSON batch on stdin, dedup by id + content hash); `recall` drives
  `eidetic recall` with four search modes — exact / approximate / keyword /
  hybrid — each hit carrying text, full provenance metadata, a relevance score,
  and a freshness signal. The `.sh` wrappers are byte-verbatim from eidetic-cli
  (their first-party origin); each `SKILL.md` is localized only in the
  illustrative `--scope <nick>` examples (Provenance keeps "First-party to
  eidetic-cli"). Both default to this agent's PRIVATE scope, reading the suffix
  from `culture.yaml`. Runtime dep: the `eidetic` CLI on PATH (else a local
  eidetic-cli checkout with `uv`). Propagated by rollout-cli's `eidetic-memory`
  recipe.

## [0.3.4] - 2026-06-20

### Fixed

- Identity docs and self-description strings still claimed `backend: claude`
  (prompt file `CLAUDE.md`), but this template was promoted to a colleague
  resident in #14/#15: `culture.yaml` declares `backend: colleague` (Qwen) with
  `AGENTS.colleague.md` as the resident prompt. Corrected the stale claim in
  `CLAUDE.md` (Identity section), `README.md`, `docs/skill-sources.md`, and the
  two CLI description strings (`overview` artifacts and `explain doctor`). The
  `doctor` backend→prompt-file mapping and the tests were already on
  `colleague`; this aligns the prose and self-description with them.

## [0.3.3] - 2026-06-20

### Fixed

- pyproject.toml: correct the `license` field and PyPI classifier from MIT to
  Apache-2.0 to match the `LICENSE` file. The README License section was already
  corrected in 0.3.2, but the package metadata was missed; the built wheel now
  reports `License-Expression: Apache-2.0`.

## [0.3.2] - 2026-06-18

### Added

- ask-colleague skill: `monitor`/`guide`/`stop` pilot verbs plus a `--watch`
  flag to dispatch, watch the live feed of, send mid-flight guidance to, and
  cooperatively stop a running colleague flight (re-vendored from colleague).

### Changed

- README: correct the License section from MIT to Apache 2.0 to match the
  `LICENSE` file.

## [0.3.1] - 2026-06-13

### Changed

- CLAUDE.md: add a convention to reach for the `ask-colleague` skill reflexively
  for explore/review/write/grade — read-only `review`/`explore` are always safe;
  side-effecting `write` needs the user's go-ahead.

## [0.3.0] - 2026-06-13

### Added

- AGENTS.colleague.md resident prompt file (backend colleague <-> AGENTS.colleague.md)

### Changed

- Promote agent identity to a colleague resident: culture.yaml backend
  claude -> colleague with a pinned model. The `doctor` backend-consistency
  map gains `colleague` -> AGENTS.colleague.md.

## [0.2.1] - 2026-06-12

### Changed

- **Re-vendored the `ask-colleague` skill from colleague (now 1.7.0, up from the
  0.39.2 sync)** — the wrapper had drifted multiple releases behind origin. Picks
  up the `clean` verb (reap stale/corrupt `colleague/*` branches + orphaned
  `.colleague/` artifacts a crashed run left behind), the `--json` flag on every
  verb (result JSON on stdout, diagnostics/digest on stderr), the
  `_colleague_via_uv` local-dev resolution that honors `--repo`, and the
  tri-state (0/1/2) exit-code contract. `scripts/ask-colleague.sh` + `prompts/`
  are byte-identical to the origin; `SKILL.md` diverges only in the one
  consumer-identifying Provenance clause (`embeddings-cli vendors from
  guildmaster`). `docs/skill-sources.md` sync row updated to
  `2026-06-12 (colleague 1.7.0, direct)`. Refs: colleague#183, #186.

## [0.2.0] - 2026-06-06

### Added

- **`ask-colleague` skill** (`.claude/skills/ask-colleague/`) — the first-party front door to the `colleague` CLI (the renamed `convertible`). On top of `explore` / `review` / `write` it adds a `feedback` verb (grade a finished work item — the ROI loop), and `write` now **previews by default** in a throwaway worktree (no side effects) unless `--apply` / `--pr` is given. Reach for it reflexively — `review` for a diverse second opinion on a committed diff before opening a PR, `explore` for a fresh read of an unfamiliar area.

### Changed

- **Replaced the `outsource` skill with `ask-colleague`.** `outsource` was renamed to `ask-colleague` upstream ([colleague#148](https://github.com/agentculture/colleague/pull/148)). Because guildmaster has not re-broadcast the rename yet (its kit still ships the old `outsource`), `ask-colleague` is vendored **directly from the sibling `colleague` checkout** rather than from guildmaster — a tracked local divergence recorded in `docs/skill-sources.md`, parallel to the `agex` → `devex` one. Vendored verbatim except one consumer-identifying clause in the Provenance paragraph.
- **Ledger + CLAUDE.md + `.gitignore`:** point `docs/skill-sources.md` and the CLAUDE.md Skills section at `colleague` / `ask-colleague`, swap the *optional* runtime prerequisite `convertible` → `colleague` (env prefix `CONVERTIBLE_*` → `COLLEAGUE_*`, with the legacy names kept as a deprecated fallback), and gitignore the `.colleague/` run-artifact dir the skill writes (plus the stale `.agex/`).

## [0.1.4] - 2026-05-31

### Added

- **Vendor the `outsource` skill** (`.claude/skills/outsource/`) from
  guildmaster's canonical copy (origin
  [`agentculture/convertible`](https://github.com/agentculture/convertible),
  re-broadcast via guildmaster — guildmaster
  [#51](https://github.com/agentculture/guildmaster/pull/51)). Every agent
  cloned from this template now inherits the ability to hand a scoped task to a
  *different* engine/mind: `explore` (read-only investigation), `review` (a
  diverse second opinion on the committed diff), and `write` (delegate a small
  implementation). `explore`/`review` run isolated in a throwaway `git worktree`;
  `write` refuses a dirty tree. Fulfils
  [#8](https://github.com/agentculture/embeddings-cli/issues/8).
- **Ledger + CLAUDE.md:** record `outsource` in `docs/skill-sources.md`
  (origin = convertible, re-broadcast via guildmaster; vendored verbatim — it
  already carries `type: command`) and document its *optional* runtime
  dependency on the `convertible` CLI (the skill exits with an install hint if
  absent, so a clone that never uses it is unaffected).

### Changed

### Fixed

## [0.1.3] - 2026-05-31

### Changed

- Expanded the clone-and-rename instructions in `CLAUDE.md`: added `README.md` to
  the rename targets and a portable `git grep` discovery command so a cloner can
  find every occurrence of the template name (hard-coded in ~100 places across the
  package, including the CLI command files and `_ISSUES_URL` in
  `embeddings/cli/__init__.py`) rather than renaming by hand.
- Synced `README.md`'s "Make it your own" checklist with `CLAUDE.md`: it now lists
  `README.md` itself as a rename target and points to `CLAUDE.md`'s discovery
  command as the authoritative procedure, so the two onboarding checklists no
  longer drift.

## [0.1.2] - 2026-05-30

### Changed

- Renamed the PR-lifecycle CLI references `agex` / `agex-cli` to `devex` (same
  tool, new name) across `CLAUDE.md`, `docs/skill-sources.md`, `.gitignore`, and
  the vendored `cicd`, `assign-to-workforce`, and `communicate` skills — the
  `cicd` scripts now invoke `devex pr`.
- Logged the vendored-skill in-place patch as a local divergence in
  `docs/skill-sources.md`; the matching canonical rename is tracked upstream for
  guildmaster in
  [agentculture/guildmaster#48](https://github.com/agentculture/guildmaster/issues/48)
  so a future re-sync reconciles cleanly.
- Aligned the documented `devex` version floor to `>=0.21` across the vendored
  `cicd` `SKILL.md` and `workflow.sh` install hint (were `>=0.1`), matching
  `docs/skill-sources.md` and the `await`-era feature set; flagged upstream on
  guildmaster#48.

### Fixed

- SonarCloud now reports code coverage — added `relative_files = true` to
  `[tool.coverage.run]` so `coverage.xml` emits repo-relative paths that map to
  `sonar.sources=embeddings` (absolute / `.venv` paths were dropped
  as unmappable). Mirrors the sibling `convertible` setup.

## [0.1.1] - 2026-05-26

### Changed

- **CI gates on the SonarCloud quality gate**
  ([issue #3](https://github.com/agentculture/embeddings-cli/issues/3)) —
  added `sonar.qualitygate.wait=true` to `sonar-project.properties` so a failing
  gate fails the `test` job when `SONAR_TOKEN` is set. Token-less repos and fork
  PRs remain green (the scan step is guarded by `if: env.SONAR_TOKEN != ''`).

## [0.1.0] - 2026-05-26

### Added

- **Onboarded into the AgentCulture mesh** ([issue #1](https://github.com/agentculture/embeddings-cli/issues/1)).
- **Agent-first CLI** cited from teken's (`afi-cli`) `python-cli` reference
  (`teken cli cite`) — verbs `whoami`, `learn`, `explain`, `overview`, `doctor`,
  and the `cli` noun group. Runtime is self-contained (`dependencies = []`);
  `teken>=0.8` is a dev dependency only. Passes the seven-bundle agent-first
  rubric (`teken cli doctor . --strict`). `doctor` checks the agent-identity
  invariants (prompt-file-present, backend-consistency, skills-present).
- **Mesh identity**: `culture.yaml` (`suffix: embeddings-cli`,
  `backend: claude`) and the matching `CLAUDE.md` prompt file.
- **Canonical guildmaster skill kit** (11 skills) vendored under
  `.claude/skills/` (cite-don't-import): `agent-config`, `assign-to-workforce`,
  `cicd`, `communicate`, `doc-test-alignment`, `pypi-maintainer`, `run-tests`,
  `sonarclaude`, `spec-to-plan`, `think`, `version-bump`. Every `SKILL.md`
  carries `type: command` (load-bearing for the culture/claude backend);
  `cicd` / `communicate` consumer-identifying prose adapted, all script bodies
  verbatim. Provenance in `docs/skill-sources.md`. Three skills (`think`,
  `spec-to-plan`, `assign-to-workforce`) originate in `devague`, re-broadcast
  via guildmaster.
- **Build + deploy baseline**: `pyproject.toml` (hatchling), `tests/` (pytest,
  xdist, coverage), `.github/workflows/{tests,publish}.yml` (CI rubric/lint gate,
  PyPI Trusted Publishing), `.flake8`, `.markdownlint-cli2.yaml`,
  `sonar-project.properties`, and `.claude/skills.local.yaml.example`.

### Changed

### Fixed
