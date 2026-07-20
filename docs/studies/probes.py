"""Probe sets for the eidetic-embedder study.

Each fact below is a distinction a memory store MUST preserve. If the embedder
collapses one, ``eidetic recall`` returns a confidently wrong memory -- so these
are failure modes with real consequences, not abstract geometry.

The pilot for this study used four independent sets (value substitution,
negation, paraphrase, unrelated controls) and found contradictory statements
sitting roughly four times closer than paraphrases. That design has a fatal
confound: the value pairs differed by one token while the paraphrases shared
almost no vocabulary. "The model cannot represent negation" and "surface form
dominates the discriminating token" both predict the pilot result, and they are
different claims with different consequences.

This file replaces the four independent sets with a **matched 2x2 factorial**
over a single list of facts. Every fact carries four variants of one base
statement:

===================  ==================  =========================================
variant              meaning vs. base    surface form vs. base
===================  ==================  =========================================
``para_high``        same                near-identical (one non-discriminating
                                         token changed)
``contra_high``      opposite            near-identical (only the discriminating
                                         token changed)
``para_low``         same                reworded, little shared vocabulary
``contra_low``       opposite            reworded, little shared vocabulary
===================  ==================  =========================================

``contra_low`` is deliberately built as ``para_low`` with the discriminating
element flipped, so the two low-overlap cells are matched to each other the same
way the two high-overlap cells are.

The design separates the two explanations because they predict different
orderings of the four cells' distance-to-base:

* **Surface-form account.** Distance tracks lexical overlap, so
  ``para_high`` ~ ``contra_high`` << ``para_low`` ~ ``contra_low``.
* **Semantic account.** Distance tracks meaning, so
  ``para_high`` ~ ``para_low`` << ``contra_high`` ~ ``contra_low``.

Because all four variants of a fact are about the same topic, in the same
register, at similar length, topic and style are held fixed across the
comparison and the only things varying are meaning and wording.

The twenty facts split evenly into two contradiction kinds:

* ``value`` -- a substituted value (port, version, hash, date, threshold). The
  stale-fact failure: the record is about the right subject and states the wrong
  number.
* ``negation`` -- a reversed polarity or decision. The "we decided the opposite"
  failure.

Ten of the facts are shaped after real ``~/.eidetic/memory`` records, which lean
heavily on commit hashes, branch names, dates, and done/not-done status. Those
records were read read-only for shape only; no private record content is
reproduced here or in the study.
"""

from __future__ import annotations

# Each entry: id, kind, and the base statement plus its four matched variants.
FACTS = [
    # ---------------------------------------------------------------- value --
    {
        "id": "v01",
        "kind": "value",
        "base": "the lobes embedding gateway listens on port 8001",
        "para_high": "the lobes embedding gateway runs on port 8001",
        "contra_high": "the lobes embedding gateway listens on port 8002",
        "para_low": "port 8001 is where you reach the embedding service that lobes fronts",
        "contra_low": "port 8002 is where you reach the embedding service that lobes fronts",
    },
    {
        "id": "v02",
        "kind": "value",
        "base": "eidetic defaults its embed backend to Qwen3-Embedding-0.6B",
        "para_high": "eidetic sets its embed backend to Qwen3-Embedding-0.6B",
        "contra_high": "eidetic defaults its embed backend to Qwen3-Embedding-4B",
        "para_low": "the embedding model you get out of the box is Qwen3-Embedding-0.6B",
        "contra_low": "the embedding model you get out of the box is Qwen3-Embedding-4B",
    },
    {
        "id": "v03",
        "kind": "value",
        "base": "the coverage gate fails the build under 60 percent",
        "para_high": "the coverage gate breaks the build under 60 percent",
        "contra_high": "the coverage gate fails the build under 80 percent",
        "para_low": "anything below 60 percent coverage and CI will not go green",
        "contra_low": "anything below 80 percent coverage and CI will not go green",
    },
    {
        "id": "v04",
        "kind": "value",
        "base": "survey writes the atlas only when the apply flag is passed",
        "para_high": "survey saves the atlas only when the apply flag is passed",
        "contra_high": "survey writes the atlas only when the commit flag is passed",
        "para_low": "nothing lands on disk from a survey run without that apply switch",
        "contra_low": "nothing lands on disk from a survey run without that commit switch",
    },
    {
        "id": "v05",
        "kind": "value",
        "base": "the release bumped the version from 0.7.0 to 0.8.0",
        "para_high": "the release moved the version from 0.7.0 to 0.8.0",
        "contra_high": "the release bumped the version from 0.6.0 to 0.7.0",
        "para_low": "this cut took us up to 0.8.0, coming from 0.7.0",
        "contra_low": "this cut took us up to 0.7.0, coming from 0.6.0",
    },
    {
        "id": "v06",
        "kind": "value",
        "base": "the embedding vectors this gateway returns are 1024 dimensional",
        "para_high": "the embedding vectors this gateway emits are 1024 dimensional",
        "contra_high": "the embedding vectors this gateway returns are 768 dimensional",
        "para_low": "each vector coming off that endpoint carries 1024 floats",
        "contra_low": "each vector coming off that endpoint carries 768 floats",
    },
    {
        "id": "v07",
        "kind": "value",
        "base": "the message-tags work landed in commit fe3e18b",
        "para_high": "the message-tags work shipped in commit fe3e18b",
        "contra_high": "the message-tags work landed in commit 2fdd056",
        "para_low": "commit fe3e18b is the one that carried tag parsing into the protocol layer",
        "contra_low": "commit 2fdd056 is the one that carried tag parsing into the protocol layer",
    },
    {
        "id": "v08",
        "kind": "value",
        "base": "the event catalog task was completed on 2026-04-16",
        "para_high": "the event catalog task was finished on 2026-04-16",
        "contra_high": "the event catalog task was completed on 2026-04-15",
        "para_low": "we closed out the catalog work on 2026-04-16",
        "contra_low": "we closed out the catalog work on 2026-04-15",
    },
    {
        "id": "v09",
        "kind": "value",
        "base": "the mesh event work lives on branch feat/mesh-events",
        "para_high": "the mesh event work sits on branch feat/mesh-events",
        "contra_high": "the mesh event work lives on branch fix/init-rubric-naming",
        "para_low": "you will find those commits under feat/mesh-events",
        "contra_low": "you will find those commits under fix/init-rubric-naming",
    },
    {
        "id": "v10",
        "kind": "value",
        "base": "survey sends 64 texts per request to the backend",
        "para_high": "survey ships 64 texts per request to the backend",
        "contra_high": "survey sends 32 texts per request to the backend",
        "para_low": "each outgoing request carries a batch of 64 texts",
        "contra_low": "each outgoing request carries a batch of 32 texts",
    },
    # ------------------------------------------------------------- negation --
    {
        "id": "n01",
        "kind": "negation",
        "base": "we decided to add numpy as a runtime dependency",
        "para_high": "we chose to add numpy as a runtime dependency",
        "contra_high": "we decided against adding numpy as a runtime dependency",
        "para_low": "numpy is going into the runtime requirements; that was the call",
        "contra_low": "numpy is staying out of the runtime requirements; that was the call",
    },
    {
        "id": "n02",
        "kind": "negation",
        "base": "the compare verb ships CKA in version one",
        "para_high": "the compare verb includes CKA in version one",
        "contra_high": "the compare verb defers CKA past version one",
        "para_low": "CKA is part of what that comparison does at its first release",
        "contra_low": "CKA is left out of what that comparison does at its first release",
    },
    {
        "id": "n03",
        "kind": "negation",
        "base": "the map page depends on a third party charting library",
        "para_high": "the map page relies on a third party charting library",
        "contra_high": "the map page avoids depending on a third party charting library",
        "para_low": "rendering that visualisation pulls in an outside plotting package",
        "contra_low": "rendering that visualisation pulls in no outside plotting package",
    },
    {
        "id": "n04",
        "kind": "negation",
        "base": "probing is enabled by default in the gaps verb",
        "para_high": "probing is on by default in the gaps verb",
        "contra_high": "probing is disabled by default in the gaps verb",
        "para_low": "you get a probe step out of that command without asking for one",
        "contra_low": "you get no probe step out of that command unless you ask for one",
    },
    {
        "id": "n05",
        "kind": "negation",
        "base": "the atlas commits its vector sidecars to git",
        "para_high": "the atlas checks its vector sidecars into git",
        "contra_high": "the atlas keeps its vector sidecars out of git",
        "para_low": "those float arrays are tracked in version control alongside the manifest",
        "contra_low": "those float arrays are left untracked, unlike the manifest",
    },
    {
        "id": "n06",
        "kind": "negation",
        "base": "the stacked motor setup pull requests are ours to merge",
        "para_high": "the stacked motor setup pull requests are ours to land",
        "contra_high": "the stacked motor setup pull requests are not ours to merge",
        "para_low": "we can close those branches out ourselves once review is done",
        "contra_low": "we must leave those branches for the user to close out after review",
    },
    {
        "id": "n07",
        "kind": "negation",
        "base": "the test suite requires a running embedding gateway",
        "para_high": "the test suite needs a running embedding gateway",
        "contra_high": "the test suite runs without any embedding gateway",
        "para_low": "you cannot get a green run unless the embedder is up",
        "contra_low": "you can get a green run whether or not the embedder is up",
    },
    {
        "id": "n08",
        "kind": "negation",
        "base": "SonarCloud reports coverage correctly when the paths are absolute",
        "para_high": "SonarCloud reports coverage properly when the paths are absolute",
        "contra_high": "SonarCloud reports coverage as zero when the paths are absolute",
        "para_low": "absolute paths in that report file are fine by the quality gate",
        "contra_low": "absolute paths in that report file silently zero out the quality gate",
    },
    {
        "id": "n09",
        "kind": "negation",
        "base": "every pull request in this repository has to bump the version",
        "para_high": "every pull request in this repository must bump the version",
        "contra_high": "no pull request in this repository has to bump the version",
        "para_low": "you cannot merge here without touching the version number",
        "contra_low": "you can merge here without touching the version number",
    },
    {
        "id": "n10",
        "kind": "negation",
        "base": "a genuine cross model comparison was demonstrated on this machine",
        "para_high": "a real cross model comparison was demonstrated on this machine",
        "contra_high": "a genuine cross model comparison was never demonstrated on this machine",
        "para_low": "we did manage to put two different embedding models side by side here",
        "contra_low": "we never managed to put two different embedding models side by side here",
    },
]

# The four matched variants, in the order the study's tables report them.
VARIANTS = ("para_high", "contra_high", "para_low", "contra_low")

# Unrelated sentences establishing the corpus-wide baseline distance. They share
# neither topic nor vocabulary with the facts above, and are drawn from several
# unrelated domains so the baseline is not itself a single tight cluster.
CONTROLS = [
    "sourdough needs a long cold ferment before it develops any sourness",
    "the tide tables shift with the lunar cycle over the course of a month",
    "cast iron should not be left to soak overnight in the sink",
    "migratory geese navigate partly by sensing the earth's magnetic field",
    "the cello section sits to the conductor's right in most modern orchestras",
    "basalt columns form as thick lava cools and contracts into hexagons",
    "a badly seasoned wok will make everything you cook in it stick",
    "the postal service raised the price of a first class stamp again",
    "beekeepers watch for swarming behaviour in late spring",
    "cross country skis need a grip wax matched to the snow temperature",
    "the lighthouse keeper's cottage was sold to a family from the mainland",
    "olive trees can keep producing fruit for several hundred years",
    "the bridge deck was resurfaced during the two week closure",
    "sled dogs eat far more calories per day in winter than in summer",
    "a well built dry stone wall uses no mortar at all",
    "the harbour seals haul out on the sandbank at low tide",
    "wool shrinks when it is agitated in water that is too hot",
    "the observatory closes to visitors whenever the humidity climbs",
    "sourwood honey has a distinctly different flavour from clover honey",
    "the ferry timetable changes completely once the season ends",
]

# Qwen3-Embedding is trained to accept an instruction prefix on the *query* side
# of a retrieval pair. This is the task description used for the instruction
# condition; the study reports both a symmetric variant (both sides prefixed)
# and the realistic asymmetric variant (query side prefixed, stored record not).
INSTRUCTION = (
    "Given a question about a project's recorded decisions and facts, "
    "retrieve the memory record that answers it"
)


def instruct(text: str) -> str:
    """Wrap TEXT in the Qwen3-Embedding instruction-prefixed query format."""
    return f"Instruct: {INSTRUCTION}\nQuery: {text}"


# Real records in ``~/.eidetic/memory`` have a median content length of roughly
# 2,250 characters, against roughly 50 for the bare probe statements above. A
# discriminating token is therefore a far smaller fraction of a real record's
# surface than of a probe's, so the length condition tests whether the effect
# measured on short probes gets worse at realistic record length.
#
# The padding below is deliberately generic session boilerplate: it carries no
# information about any fact, and it is identical across every padded record, so
# the only thing that varies within a padded pair is still the discriminating
# token. No content from the real store is reproduced -- only its length and its
# register are imitated.
CONTEXT_PREFIX = (
    "Session note recorded during a working block on the agent tooling stack. "
    "Context for whoever picks this up next: the branch was cut from main "
    "earlier in the day, the working tree was clean at the start, and the usual "
    "gate set was run before and after the change. The suite was green, the "
    "linters were clean, and the coverage report was regenerated. Nothing in "
    "this note supersedes anything in the repository itself; where the two "
    "disagree, the repository wins and this note should be corrected. The "
    "operator was working from a single workstation with the local gateway up. "
    "Timings below are wall clock and were not averaged over repeats. The "
    "surrounding tooling did not change during the block, so any behaviour "
    "difference recorded here is attributable to the change under discussion "
    "rather than to the environment. "
)

CONTEXT_SUFFIX = (
    " Follow-ups noted at the time: re-check the affected paths once the next "
    "release lands, confirm the behaviour still holds after the dependency "
    "refresh, and fold anything surprising back into the guidance file rather "
    "than leaving it in a session note. No action was taken on those "
    "follow-ups during this block. The note was written at the end of the "
    "session rather than incrementally, so the ordering of the observations "
    "above reflects how they were recalled and not necessarily the order in "
    "which they happened. Nothing here was verified a second time. If a claim "
    "in this note matters, re-derive it from the repository before relying on "
    "it, because session notes go stale faster than the code they describe and "
    "this one carries no expiry."
)


def padded(text: str) -> str:
    """Embed TEXT in generic session boilerplate at realistic record length."""
    return f"{CONTEXT_PREFIX}The point worth recording is that {text}.{CONTEXT_SUFFIX}"


def texts():
    """Yield (record_id, text, fact_id, kind, role) for every probe text."""
    for fact in FACTS:
        yield f"{fact['id']}-base", fact["base"], fact["id"], fact["kind"], "base"
        for variant in VARIANTS:
            yield (
                f"{fact['id']}-{variant.replace('_', '-')}",
                fact[variant],
                fact["id"],
                fact["kind"],
                variant,
            )
    for index, text in enumerate(CONTROLS, start=1):
        yield f"c{index:02d}", text, "", "control", "control"
