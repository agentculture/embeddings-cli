# embeddings-cli

**embeddings-cli is a tool for studying embedding spaces as systems, not just
querying individual vectors.**

An individual embedding is difficult to interpret. Its dimensions rarely
correspond to clean human concepts, and inspecting the raw numbers tells us very
little.

But an embedding space contains larger structures that can be studied:

- semantic regions and clusters;
- neighborhoods and boundaries;
- recurring directions of meaning;
- dense, sparse, and ambiguous areas;
- gaps in what the model represents;
- differences between models, datasets, domains, and versions.

`embeddings-cli` treats an embedding model as a **semantic landscape**.

Instead of only asking:

> Which documents are closest to this query?

it asks:

> What kind of world does this embedding model create?

The CLI can embed a representative corpus, survey the resulting space, identify
regions and directions, compare different embedding models, and generate an
interpretable map of the model's semantic coverage.

```bash
embeddings survey corpus.jsonl
embeddings regions
embeddings neighbors "agentic culture"
embeddings directions discover
embeddings gaps
embeddings compare model-a model-b
embeddings map --html
```

The goal is not to assign a human-readable meaning to every dimension. That is
usually unstable and misleading.

The goal is to understand the **geometry of meaning produced by the model**:

- what it places together;
- what it keeps apart;
- which distinctions it preserves;
- which distinctions it collapses;
- where concepts form continuous transitions;
- where the representation becomes weak or uncertain.

## Status

The survey surface above is the **target design, not yet implemented**. What
ships today is the agent-first CLI skeleton this project is built on — identity
and introspection verbs (`whoami`, `learn`, `explain`, `overview`, `doctor`,
`cli overview`), each with `--json`. See [CLI](#cli) for what actually runs.

## The Surveyor agent

`embeddings-cli` can also serve as the deterministic tool surface for a
**Surveyor agent**.

The Surveyor does not invent an interpretation from a single projection. It
performs experiments against the space:

1. samples representative concepts;
2. searches for natural clusters;
3. proposes names for regions;
4. tests those names against independent examples;
5. discovers candidate semantic directions;
6. measures whether those directions generalize;
7. identifies gaps, overlaps, and unstable boundaries;
8. records the evidence behind every conclusion.

The agent may describe a region, but the CLI provides the measurements that make
the description inspectable and reproducible.

## Outputs

A study can produce an **Atlas** containing:

- the datasets and models used;
- representative points and neighborhoods;
- hierarchical region maps;
- discovered semantic directions;
- cluster and boundary evidence;
- coverage and density measurements;
- model comparisons;
- known limitations and uncertain interpretations.

The Atlas is not the embedding space itself. It is an evidence-backed
interpretation of that space.

## Why it matters

Embedding models are often treated as interchangeable infrastructure. They are
selected using retrieval benchmarks and then hidden behind a vector database.

But every embedding model defines its own system of similarity and distinction.

Two models may disagree about:

- whether two concepts are related;
- which property of a sentence is most important;
- how domains connect;
- how multilingual meanings align;
- whether a semantic change is gradual or abrupt;
- which concepts are represented precisely and which are compressed together.

`embeddings-cli` makes those differences visible.

It turns embeddings from an opaque retrieval component into something that can
be explored, evaluated, compared, and understood.

## Quickstart

```bash
uv sync
uv run pytest -n auto              # run the test suite
uv run embeddings whoami           # identity from culture.yaml
uv run embeddings learn            # self-teaching prompt (add --json)
uv run teken cli doctor . --strict # the agent-first rubric gate CI runs
```

The project is named `embeddings-cli`; the installed command is **`embeddings`**.

## CLI

Shipping today:

| Verb | What it does |
|------|--------------|
| `whoami` | Report this agent's nick, version, backend, and model from `culture.yaml`. |
| `learn` | Print a structured self-teaching prompt. |
| `explain <path>` | Markdown docs for any noun/verb path. |
| `overview` | Read-only descriptive snapshot of the agent. |
| `doctor` | Check the agent-identity invariants (prompt-file-present, backend-consistency). |
| `cli overview` | Describe the CLI surface itself. |

Planned (the survey surface): `survey`, `regions`, `neighbors`, `directions`,
`gaps`, `compare`, `map`.

Every command supports `--json`. Results go to stdout, errors/diagnostics to
stderr (never mixed). Exit codes: `0` success, `1` user error, `2` environment
error, `3+` reserved.

## License

Apache 2.0 — see [`LICENSE`](LICENSE).
