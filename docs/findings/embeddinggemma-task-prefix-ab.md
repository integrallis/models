# Does EmbeddingGemma's task prefix improve router classification?

**Pre-registered 2026-09-27, before any arm was run.** Written first so the decision rule cannot move
after the numbers land.

## Why ask

`models-router` classifies a prompt by nearest neighbour against an index of labelled exemplars, both
sides embedded by EmbeddingGemma-300M Q8_0. **No prefix is applied on either side** — verified by
reading the path: the only `prefix` occurrences in `models-router` are KV-cache token counts.

EmbeddingGemma is documented (**read, not measured**) to expect a task-specific prefix, e.g.
`task: classification | query: ` for classification and `task: search result | query: ` /
`title: none | text: ` for retrieval. The GGUF carries **no** template metadata to confirm this: all
41 keys were dumped, and none is a prompt or prefix key (`pooling_type = 1`, mean). So the file
cannot settle it and only measurement can.

Note what is *not* being claimed. Embedding both sides bare is self-consistent — exemplars and
queries are the same kind of text, embedded identically, so the comparison is already apples to
apples. This is not a bug report. The open question is only whether the trained-for prefix separates
the space better.

## Arms

Same corpus, same 481 held-out prompts, same model file, SQ4, threshold 0.0, one host.

| arm | index side | query side |
|---|---|---|
| **A** (baseline, already measured) | none | none |
| **B** symmetric classification | `task: classification \| query: ` | `task: classification \| query: ` |
| **C** asymmetric retrieval | `title: none \| text: ` | `task: search result \| query: ` |

A is **0.9044 (435/481)**, measured here.

## Decision rule

The arms are **paired** — the same 481 items scored by each — so the marginal binomial SE is the
wrong test and would be misleadingly wide (≈1.37pp near p=0.90). The correct test is on the
discordant pairs only:

- **McNemar, exact binomial on discordant pairs, two-sided, α = 0.05.**
- Adopt an arm over A only if it is significant by that test **and** its point estimate is higher.
- If both B and C qualify, take the higher point estimate.
- If neither qualifies, **change nothing** and record the negative result here. A null is a result:
  it says the prefix is not worth the extra tokens on every query, which is itself a latency finding.

Per-task accuracy is reported for diagnosis but is **not** a decision input — ten tasks at n≈50 each
would let a subgroup be cherry-picked. The single held-out accuracy decides.

## What this cannot test

One corpus, one embedder, one host. A win here is evidence about EmbeddingGemma on this corpus and is
not transferable to Nomic or E5, whose prefixes are documented separately in
`asymmetric-embedding-prefixes.md`. A null here does not mean prefixes are inert for those models.

## Results

**Measured here** 2026-09-27, one host, all three arms on the same 481 held-out prompts, same model
file, SQ4, threshold 0.0. Every arm's index carries `corpusSha256=639d305a…` and 1929 prompts, so the
prefix is the only difference between them.

| arm | accuracy | delta | discordant (A-only / arm-only) | McNemar exact p | verdict |
|---|---|---|---|---|---|
| **A** no prefix | 0.9044 (435/481) | — | — | — | baseline |
| **B** symmetric `task: classification \| query: ` | **0.9459 (455/481)** | **+0.0416** | 8 / 28 | **0.0012** | **ADOPT** |
| **C** asymmetric retrieval | 0.8399 (404/481) | −0.0644 | 46 / 15 | 0.0001 | REJECT, significantly **worse** |

B is adopted by the pre-registered rule: significant at α = 0.05 with a higher point estimate. Arm A
was re-run from its cache before the comparison and reproduced 435/481 exactly.

### Both directions are informative

The headline is not only that a prefix helps. It is that **the wrong prefix costs more than no prefix
at all** — C is 6.4 points below baseline, a larger move than B's gain. Running an instruction-tuned
embedder with no prefix is a mild, silent tax; running it with the prefix for the wrong task is a much
larger one. That asymmetry is the reason the prefixes belong in the manifest and not at the call site:
the expensive mistake is the one a caller makes by guessing.

C's loss is concentrated in `chat`, which falls from 41/50 to **17/50**. Embedding queries as search
queries and exemplars as titled documents puts the two sides in regions the model keeps apart, and the
tasks with no strong surface signature are then the ones that land on whatever happens to be nearest.

### Per-task (diagnosis only, never a decision input)

| task | A | B | C | B − A |
|---|---|---|---|---|
| `chat` | 41/50 | 42/50 | 17/50 | +0.020 |
| `code` | 49/50 | 49/50 | 49/50 | +0.000 |
| `creative` | 46/48 | 47/48 | 42/48 | +0.021 |
| `extraction` | 40/54 | 50/54 | 42/54 | +0.185 |
| `math` | 49/50 | 49/50 | 48/50 | +0.000 |
| `reasoning` | 33/42 | 38/42 | 33/42 | +0.119 |
| `sql` | 47/50 | 46/50 | 41/50 | -0.020 |
| `summarization` | 53/58 | 58/58 | 56/58 | +0.086 |
| `tool-use` | 27/29 | 26/29 | 26/29 | -0.034 |
| `translation` | 50/50 | 50/50 | 50/50 | +0.000 |
B's gains concentrate where intent rather than surface form separates the classes: `extraction`
+0.185, `reasoning` +0.119, and `summarization` +0.086 (to 58/58). The four tasks whose form is
already distinctive — `code`, `math`, `translation`, and near-ceiling `sql` — move by at most one
prompt. B's two regressions (`sql` −1, `tool-use` −1) sit inside the discordant counts the McNemar
test already accounts for.

### What this still does not test

One corpus, one embedder, one host, one quantizer. It says nothing about Nomic or E5, whose prefixes
differ and whose effect remains unmeasured — see `asymmetric-embedding-prefixes.md`.

It also does not separate *which* part of B's prefix earns the gain: the arm changed
`task: classification` and the bare `query: ` scaffold together, so either could be carrying it. A
follow-up splitting them is cheap and is **not run**.

The prefix adds roughly 7 tokens to every exemplar and every query. On the exemplars that is a one-off
build cost. Per query it is a small constant, and it was **not measured here as latency**.
