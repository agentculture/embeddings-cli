# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

`embeddings-cli` studies **embedding spaces as systems**, not individual
vectors. A single embedding is hard to interpret — its dimensions rarely map to
clean human concepts. An embedding space, though, has structure worth measuring:
semantic regions and clusters, neighborhoods and boundaries, recurring
directions of meaning, dense/sparse/ambiguous areas, gaps in what the model
represents, and differences between models, datasets, domains, and versions.

The framing: treat an embedding model as a **semantic landscape**. Rather than
only asking "which documents are closest to this query?", ask "what kind of
world does this model create?" — what it places together, what it keeps apart,
which distinctions it preserves, which it collapses, where concepts form
continuous transitions, and where the representation goes weak or uncertain. The
goal is explicitly *not* to assign human-readable meaning to each dimension;
that is unstable and misleading.

It also serves as the deterministic tool surface for a **Surveyor agent**, which
doesn't invent an interpretation from one projection but runs experiments
against the space — sample concepts, find clusters, propose region names, test
those names against independent examples, discover candidate directions, measure
whether they generalize, identify gaps and unstable boundaries, and record the
evidence behind every conclusion. The output is an **Atlas**: an evidence-backed
interpretation of the space, not the space itself.

### What exists today

**The survey surface ships.** All seven verbs are implemented, registered, and
callable — `survey`, `regions`, `neighbors`, `directions discover` /
`directions overview`, `compare`, `gaps`, `map` — alongside the introspection
verbs `whoami`, `learn`, `explain`, `overview`, `doctor`, `cli overview`.
`export` is the producer step for the sibling
[`embeddings-tumbler`](https://github.com/agentculture/embeddings-tumbler)
viewer: it writes an Atlas with its vectors as the viewer's data contract
(schema version 1, `<name>.json` + float32 `<name>.f32`). It never copies the
Atlas's endpoint URL — `provenance.endpoint` is a route name from
`--endpoint-name` or is omitted — and `tests/_viewer_contract.py` is an
independent checker of that contract written in this repo, not imported from
the viewer. The
package persists an Atlas (`embeddings/atlas/`), talks to an embedding gateway
(`embeddings/backends/`), and measures stability (`embeddings/measure/`).

**`numpy` is a runtime dependency, deliberately.** The template's
"no third-party runtime dependencies" property was inherited, not chosen, and
it is incompatible with being a measurement tool: clustering, direction
discovery, density measurement, and the projection behind `map` all need it.
The dependency was spec'd and confirmed before it landed. Nothing in CI ever
enforced zero-deps — `teken cli doctor --strict` checks CLI *shape*, not
dependencies — so this was a choice, and it is now made. **Static analysers
that flag `numpy` under `embeddings/` as a rule violation are reading a rule
this file no longer states.** Adding a *further* runtime dependency is still a
deliberate decision, not a routine one: prefer numpy-only implementations
(that is why ARI, Spearman, k-means, and PCA are hand-written here rather than
pulled from scipy/scikit-learn).

The planned/implemented separation is still load-bearing in the four surfaces
that advertise the verb list — `README.md`, `learn.py`, `explain/catalog.py`,
`overview.py`. Nothing remains planned today, and
`tests/test_roadmap_lockstep.py` derives the truth from the registered parser,
so those four can no longer drift from what is actually callable. Keep new
sections of this file grounded in checked-in reality and mark anything ahead of
it `(planned)`.

### Design constraints the domain forced

Settled during the build; kept here because the reasoning still binds:

- **Rubric shape of the new verbs.** `directions discover` is a noun with an
  action-verb, so `directions` also exposes an `overview` sub-verb to pass the
  rubric (see `cli.py` for the pattern). `survey`, `compare`, and `map` read as
  global verbs. Every new sub-verb needs its own `--json`.
- **Atlas persistence.** `survey` writes the state every reader verb loads. The
  Atlas is a directory: `atlas.json` (manifest), `records.jsonl`, and `.npy`
  vector sidecars. The manifest, records, and measurements are the *committable
  interpretation*; the sidecars are regenerable bulk and are git-ignored. That
  split is the "the Atlas is not the space" distinction made physical, so a
  freshly-cloned study legitimately has records but no vectors.

It is a sibling to [`guildmaster`](https://github.com/agentculture/guildmaster)
(the **skills supplier**), [`steward`](https://github.com/agentculture/steward)
(**alignment** — `steward doctor`, the sibling-pattern baseline), and
[`teken`](https://github.com/agentculture/teken) (the **afi-cli** "Agent First
Interface" scaffolder this CLI is cited from) within the Organic Development
framework.

## Naming: the project is `embeddings-cli`, the command is `embeddings`

`pyproject.toml` declares one console script — `embeddings = "embeddings.cli:main"`
— so `embeddings` is what a user types. `embeddings-cli` is the dist name, the
repo name, and the agent nick in `culture.yaml`; it is *not* a runnable command.

The scaffold rename originally left these disagreeing (`prog` and every help
string said `embeddings-cli`, making the CLI's own documented commands
uncopyable). That is fixed, and
`test_prog_matches_declared_console_script` guards the pairing — it reads the
console script out of `pyproject.toml` and asserts `prog` matches, so a rename
that updates only one side fails loudly.

The convention when writing strings: **invocations** use `embeddings ...`;
**titles and project references** use `embeddings-cli`. The `explain` catalog
keys both `("embeddings",)` and `("embeddings-cli",)` to the root entry so
either spelling resolves.

## The CLI

The CLI is cited (cite-don't-import) from teken's `python-cli` reference
(`teken cli cite`); `teken` (a.k.a. `afi-cli`) is a dev dependency only. The
runtime package's only third-party dependency is **numpy** — see
*What exists today* above for why that is deliberate. The agent-first rubric is
enforced in CI by `teken cli doctor . --strict`.

### Structure and the contract each part holds

Everything hangs off three small modules that are marked *stable-contract* — the
agent-first rubric depends on their exact behavior, so changing them changes the
CLI's public contract:

- **`embeddings/cli/_errors.py`** — `CliError(code, message, remediation)` and
  the exit-code policy in one place: `0` success, `1` user error, `2`
  environment/setup error, `3+` reserved. Every failure inside the CLI raises
  `CliError`; nothing else.
- **`embeddings/cli/_output.py`** — `emit_result` / `emit_error` /
  `emit_diagnostic`. The invariant agents parse against: **results to stdout,
  errors and diagnostics to stderr, never mixed**, in both text and `--json`
  mode. Text-mode errors render as `error: <msg>` then `hint: <remediation>`;
  the `hint:` prefix is required by the rubric.
- **`embeddings/cli/__init__.py`** — `main()` → `_build_parser()` →
  `_dispatch()`. `_dispatch` catches `CliError` and wraps *any* other exception
  into one, so no Python traceback ever reaches stderr. Argparse's own errors
  are routed through the same format by `_CliArgumentParser.error()`; because
  parse-time errors happen before `args.json` exists, `main()` pre-scans raw
  argv for `--json` and stashes the answer in the class-level `_json_hint`.

### Adding a verb or noun group

Each module under `embeddings/cli/_commands/` exposes a `register(sub)` that
adds its parser and calls `p.set_defaults(func=...)`. Register it in
`_build_parser()` (there's a marked spot). Handlers return `None` or an `int`
exit code and raise `CliError` on failure.

Rubric constraints worth knowing before you design a command:

- **Every command takes `--json`** — including each sub-verb of a noun group,
  since argparse won't inherit the parent's flag.
- **Any noun with action-verbs must also expose `overview`.** `cli.py` exists
  only to satisfy this and is the pattern to copy for a new noun group — note
  that it passes `parser_class=type(p)` to `add_subparsers` so nested parse
  errors keep the structured error contract.
- **Descriptive verbs must not hard-fail on a missing target path.**
  `overview` accepts an optional positional `target` and ignores it, so
  `overview /no/such/path` still exits 0.
- **`learn` must stay ≥200 chars** and mention purpose, command map, exit codes,
  `--json`, and `explain`.

### `explain`

`embeddings/explain/catalog.py` is a dict keyed by command-path tuples
(`("cli", "overview")`) holding verbatim markdown; `()` and `("embeddings-cli",)`
both resolve to the root. Entries are meant to be self-contained — an agent
reading one shouldn't need to chain reads. Add an entry whenever you add a verb.

### Identity resolution

`whoami.py` parses `culture.yaml` **without a YAML dependency** (line-wise,
first agent block only) to keep runtime deps empty, and finds the file by
walking up from `__file__` — deliberately *not* the caller's CWD, so the
identity is always this agent's own. A wheel install ships no `culture.yaml`
alongside the package, so both `whoami` and `doctor` degrade to literal defaults
rather than failing. `whoami.find_culture_yaml` / `read_agent_fields` are reused
by `doctor`, and `whoami.report` by `overview` — keep them importable.

`doctor` checks the same invariants `steward doctor` verifies —
**prompt-file-present** and **backend-consistency** (`claude` → `CLAUDE.md`,
`colleague` → `AGENTS.colleague.md`, `acp` → `AGENTS.md`, `gemini` →
`GEMINI.md`) — plus a skills-present check, and reports the rubric-shaped
`{healthy, checks: [{id, passed, severity, message, remediation}]}`.

## Identity

Declared in `culture.yaml`:

```yaml
agents:
- suffix: embeddings-cli
  backend: colleague
  model: sakamakismile/Qwen3.6-27B-Text-NVFP4-MTP
```

`backend: colleague` fixes the resident prompt file to **`AGENTS.colleague.md`**
— this agent is a colleague resident served by a local Qwen model, so the mesh
runtime reads `AGENTS.colleague.md`, while `CLAUDE.md` (this file) stays the
Claude Code guidance file. `AGENTS.colleague.md` is still the generic resident
prompt inherited from the template; retarget it to the Surveyor role when that
work starts. Changing `backend` here without adding the matching prompt file
breaks `doctor` and CI.

## Commands

```bash
uv sync                                  # create .venv and install dev deps
uv run pytest -n auto                    # full suite, parallel
uv run pytest tests/test_cli.py -v       # one file
uv run pytest -k whoami -v               # one test by name
uv run pytest --cov=embeddings --cov-report=term   # coverage (fails under 60%)

uv run embeddings whoami                 # NOTE: `embeddings`, not `embeddings-cli`
uv run embeddings learn --json
uv run embeddings doctor

uv run black embeddings tests            # line length 100
uv run isort embeddings tests            # profile=black
uv run flake8 embeddings tests
uv run bandit -c pyproject.toml -r embeddings
markdownlint-cli2 "**/*.md" "#node_modules" "#.local" "#.claude/skills" "#.teken"
uv run teken cli doctor . --strict       # the agent-first rubric gate CI runs
```

CI (`.github/workflows/tests.yml`) runs tests + SonarCloud, the full lint set
above, and a `version-check` job. `.github/workflows/publish.yml` publishes to
PyPI via Trusted Publishing on push to `main`, and does a TestPyPI dry-run on
PRs; both are path-filtered to `pyproject.toml` and `embeddings/**`. The
`pypi` / `testpypi` GitHub environments and a PyPI Trusted Publisher must be
configured before the publish job can succeed.

## Conventions and workflow

- **Every PR bumps the version** — even docs/config/CI. Use the `version-bump`
  skill; the `version-check` CI job comments on and blocks the PR otherwise.
- **PRs** go through the `cicd` skill (`devex pr` + SonarCloud gating). Sign
  online posts as `- embeddings-cli (Claude)` — the `cicd` / `communicate`
  scripts resolve the nick from `culture.yaml` automatically, so don't hand-sign
  in bodies those scripts author.
- **Reach for `ask-colleague` reflexively.** Treat it as the teammate at the
  next desk, not a last resort — its value is a *second, independent mind* (a
  different backend/model), not a stronger one. Before presenting or opening a
  PR on a non-trivial committed diff, run `review`; for a fresh read of an
  unfamiliar area whose answer is independent of your current context, run
  `explore`. Both are read-only (throwaway worktree, zero side effects), so the
  reflex is always safe. The side-effecting `write --apply` / `write --pr` needs
  the user's go-ahead. Colleague's output is a second opinion to verify and own,
  never authority.
- **The vendored `.claude/skills/` are cited verbatim** — do not reformat or
  edit their scripts. Re-sync from the upstream tracked in
  `docs/skill-sources.md`, which also records four live local divergences
  (`agex`→`devex`, `ask-colleague` from `colleague`, and four devague-origin
  skills vendored directly rather than via guildmaster).
- **Tooling prerequisites:** `devex` (>=0.21) and `agtag` (>=0.1) on PATH for
  the `cicd` / `communicate` skills; `colleague` on PATH is *optional* and only
  needed when `ask-colleague` is actually invoked.
- **Per-machine config:** copy `.claude/skills.local.yaml.example` to
  `skills.local.yaml` (git-ignored) to point skills at your sibling checkouts.

### Memory discipline — recall before, remember after

The `recall` / `remember` skills are backed by the `eidetic` store. The vendored
wrappers here resolve the scope from `culture.yaml`'s `suffix` and default to
**this agent's personal, private scope** (`--scope embeddings-cli
--visibility private`), which routes to `$HOME/.eidetic/memory` — outside the
repo, never committed. Both the `claude` and `colleague` backends resolve the
same suffix, so they share these records. Pass `--visibility public` to
contribute to the shared public pool instead (public records inside a git repo
route to `<repo-root>/.eidetic/memory` and *are* committed, so choose
deliberately). `/recall` reads both stores and merges.

- **`/recall` before you start** a non-trivial task — prior decisions, gotchas,
  "have we done this before?" — so you build on what's known instead of
  re-deriving it.
- **`/remember` when something worth keeping surfaces** — a non-obvious decision
  and its rationale, a constraint, a fix and *why*, a gotcha that cost time.
  Capture it as it happens, not at the end when it's faded. Don't store what the
  repo already records (code structure, git history, this file, `CHANGELOG.md`);
  store what you'd otherwise have to re-derive.

## Layout

```text
embeddings/               agent-first CLI (cited from teken's python-cli reference)
  cli/                    parser + error/output contract; _commands/ holds the verbs
  explain/                markdown catalog for `explain`, keyed by path tuple
tests/                    pytest smoke (test_cli.py) + introspection (test_cli_introspection.py)
.claude/skills/           vendored guildmaster skill kit (cite-don't-import, verbatim)
docs/skill-sources.md     skill provenance ledger + re-sync procedure + divergences
culture.yaml              mesh identity (suffix + backend)
AGENTS.colleague.md       resident prompt for the colleague backend
.github/workflows/        tests + lint + version-check; PyPI Trusted Publishing
```
