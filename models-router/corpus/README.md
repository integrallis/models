# Router task-classification corpus

Labelled prompts used to build the task classifier that `ModelRouter` routes on.

## What is here

| File | Purpose |
| --- | --- |
| `benchmark-prompts.tsv` | The corpus: `split<TAB>task<TAB>source<TAB>prompt` |
| `sources.json` | Provenance and licence for every dataset drawn from |
| `farm_benchmarks.py` | Rebuilds the corpus from the public benchmarks |
| `bakeoff.py` | Measures embedding models as classifiers on the held-out split |

The corpus stays in the repository and is **not** shipped in the jar. What ships is
the index built from it, since only embeddings are needed at query time. Keeping the
text here means the labels stay reviewable and the index stays rebuildable.

## Where the prompts come from

Only the *input* side of each benchmark is taken; reference answers are discarded.
Two tasks have no benchmark of user-style requests, so their rows are documents or
sentence pairs wrapped in an instruction — `sources.json` marks those `wrapped: true`.

Splits are 80/20 train/eval, assigned by a seeded shuffle. Eval prompts never appear
in training, and a prompt that would carry two different task labels is dropped from
both rather than assigned to one.

## Rebuilding

```bash
pip install pyarrow
python3 farm_benchmarks.py --per-task 250
```

Whole parquet shards are downloaded rather than paged through the rows API: paging a
category-filtered source only ever sees its first few pages, which silently starves
tasks whose rows sit late in the file.

## Choosing the embedding model

```bash
python3 bakeoff.py                      # every candidate
python3 bakeoff.py --models qwen3-emb-0.6b,embeddinggemma-300m
```

Accuracy is measured by classifying each held-out prompt as the task of its nearest
training neighbour. The run also reports the distances at which correct and incorrect
matches sit, which is what the classifier's unclassified threshold has to separate —
that threshold is read off these measurements rather than picked.
