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

_To be filled in after the arms run. Nothing recorded yet._
