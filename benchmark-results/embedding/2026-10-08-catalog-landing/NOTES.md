# Landing the embedding qualifications, and what is held back

Date: 2026-10-08. Gate: `oracle-equivalence-v1`, 8 pinned probes, worst-probe cosine >= 0.999 and
unit length within 1e-3. Oracle: llama.cpp `llama-embedding`.

## The counting correction

43 artifacts were measured. The sweep script's tally counts **rows**, and retried artifacts append
a second row, so it reported 30 passes where there are **25 unique** ones. The five duplicates
(`all-MiniLM` Q4_0, Q5_0, Q6_K, Q8_0, F16) agree to every digit across both runs, so nothing is in
conflict — the count was simply wrong. `raw/verdicts-working-tree.tsv` is deduplicated last-wins.

Unique: **25 REPRODUCED, 9 NOT_REPRODUCED, 9 UNSUPPORTED** (`jina-bert-v2`, an architecture this
runtime does not implement).

## Why only 17 of the 25 can land now

Eight of the twenty-five were measured on a working tree that is not any released library, and the
catalog's `backendVersion` is a claim a user can check. The split is read from each artifact's own
GGUF tensor table (`header_types.py`, output in `raw/artifact-headers.tsv`), not from its filename:

- **4 require Q5_1**, which no released version loads at all: `all-MiniLM-L6-v2 Q4_K_S` and
  `granite-embedding-107m-multilingual` Q4_K_S, Q5_K_S and Q5_K_M. The headers show 4 to 30 Q5_1
  tensors apiece.
- **4 require the bidirectional `lfm2` encoder**: every LFM2.5-Embedding artifact that passes.

Those eight are held, not dropped. They land against the first release that carries the Q5_1 and
`lfm2` work, with that release's version in `backendVersion`.

## The 17, verified against the released library rather than the working tree

The remaining seventeen are `bert`, `nomic-bert` and `qwen3` files whose tensor types a released
0.3.54 already serves, so the claim "qualified under models-0.3.54" is checkable. It was checked:
a detached worktree at `v0.3.54` was built from source and the gate re-run on all seventeen, with
the new pinned references on the classpath and nothing else from the working tree.

**17 of 17 REPRODUCED, and 0 of 17 disagree with the working-tree run** — same verdict and same
printed cosine for every artifact (`raw/verdicts-released-0.3.54.tsv` against
`raw/verdicts-working-tree.tsv`). That is what says the uncommitted lfm2 and Q5_1 work does not
touch these paths, rather than it being argued from which files were edited.

`granite-embedding-107m-multilingual` declares `pooling_type = 2`; its CLS pooling is the `bert`
encoder's own, which predates this work. The `Pooling.CLS` added here is for the decoder-side
caller-chosen pooling that `lfm2` needs, and no `bert` file reaches it.

## What landing takes

`tools/generate-embedding-qualifications.mjs` in modeljars derives every entry from the committed
report and refuses one whose `artifactSha256` or `artifactSizeBytes` disagrees with
`catalog/models.json` — a qualification that disagrees with the catalog is a claim about a
different file than the one users download. Dry-run against these seventeen: `qualifiedModels
5 -> 22`, every digest matched.

Remaining, in order, and **not yet done**:

1. Commit the reports and the pinned references in this repo, on a branch that merges to `main`.
   `modelsRevision` must name a commit reachable on `main`, so this cannot be finished from a
   working tree.
2. `npm run catalog:embeddings -- --models-repo ../models --models-revision <sha> --backend-version
   models-0.3.54 --report-list <list>`.
3. `npm run catalog:profiles` — `catalog/model-profiles.json` is generated and gated.
4. The manifest must advance `generatedAt`; `tools/qualification-generation-gate.mjs` enforces it.

## Where this leaves the standing mandate

The catalog is **76 unique qualified model ids**, not the 44 the runbook claimed — count the union
of model ids across the six manifests, never their sum, since two ids are qualified on both the RAG
and tool workloads. The runbook has been corrected and now derives the number.

Landing these 17 takes it to **93**. The 8 held artifacts take it to **101**, so 100 is reachable
from measurements already taken, on the next library release. Nothing here needs new fleet spend.
