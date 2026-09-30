# CLM-8B on our JevBench cohort: measured, and the catalog recommendation

Run 2026-09-25 on a RunPod A40 (46 GB, driver 570.211.01, CUDA 12.8). **Same 120 items, same
scorer, as every arm in `../ACCURACY.md`** -- JevBench's own `composite_v12`, `metrics` and
`scoring`, imported rather than reimplemented.

## Recommendation: do not add CLM to the catalog as a decision model

| arm | Intelligence | Calibration | accuracy | Speed |
|---|---|---|---|---|
| **Harriet** (Qwen3.5-4B, letter logits, CPU) | **88.0** | 86.2 | 0.9000 | -- |
| CLM-8B `clm-latest` (their reference heads, A40) | **46.8** | 32.6 (ECE 0.3370) | 0.4917 | 91.6 |
| CLM `clm-raw` (their own no-head ablation) | 33.3 | 51.9 (ECE 0.2404) | 0.3333 | 91.7 |

Zero-shot, out of the box, on typed decisions, CLM scores **roughly half** our Intelligence and is
badly calibrated -- ECE 0.337 against a readout whose entire calibration story is renormalising
over the answer letters. `clm-raw` lands on 0.3333 in every tier, which is what chance looks like
here, so their projection heads are genuinely doing work (33.3 -> 46.8); the work just does not
reach far enough.

Catalog inclusion is not "we measured it", it is "it runs on our stack against an oracle" --
`backend: pure-java` with an `artifactSha256` and a reference-delta bound, per
`reranking-qualifications.json` and `embedding-qualifications.json`. For CLM that means porting
Qwen3-8B last-token pooling plus converting two MLP heads out of a `torch.save` checkpoint. **The
evidence does not justify that port**, and this file exists so nobody spends a week finding out
again.

## Where it is genuinely good, which is exactly where they say

Per family:

| family | CLM | | family | CLM |
|---|---|---|---|---|
| tool_selection | **0.917** | | adequacy | 0.417 |
| fact | 0.667 | | routing | 0.417 |
| policy | 0.667 | | intent | 0.375 |
| extraction | 0.417 | | **ordinal** | **0.250** |

By question type: `noul` 0.583, `choice` 0.486, **`score` 0.250**.

`tool_selection` at 0.917 is the bi-encoder's sweet spot and it matches their README's tool-calling
claim: a fixed, reusable action set whose options are self-describing names, where embedding the
option independently of the state costs nothing. Their cached action embeddings make that case both
accurate and nearly free.

`ordinal` at 0.250 is chance for four levels, and it is the same number **we** scored on ordinal
before rubrics reached the prompt (0.2500 -> 0.9167, `../ACCURACY.md`). A scaled cosine between a
state and four level descriptions has no mechanism for *order*: "many users blocked" and
"irreversible data loss" are similar strings, and nothing lets either one look at the state to see
which threshold it crossed. That is structural, not a tuning problem.

The most-predicted labels were `no` (27), `other` (11), `1` (11) -- a bi-encoder drifting toward
whichever option text is most generic.

## Two caveats, stated rather than buried

**This is the zero-shot arm.** Their README's headline numbers -- DeepSWE 81.6%, Terminal-Bench 2.1
87.6% -- are explicitly "with lightweight fine-tuning" per task, and they ship a second checkpoint
(`deepswe-clm-heads-8k`) for one of them. 46.8 is what the published reference head does on unseen
typed questions, which is also exactly what a catalog entry would ship. It is not a claim about
what CLM can reach when trained on a cohort.

**We substituted the encoder transport.** vLLM 0.30 ships a CUDA 13 torch and this host's driver is
12.8, so the engine would not initialise. Rather than fight the version matrix we drove
`Qwen/Qwen3-8B` in-process with `transformers` and last-token pooling -- which is the whole contract
CLM's `Embedder` has, since it L2-normalises whatever the endpoint returns (`clm/embedder.py`).
Everything downstream is theirs: `Engine`, `HeadPair`, `schema`.

That substitution was validated against **their own documented example** before any score was
trusted:

    engine.rank("What causes tides on Earth?",
                ["The Moon's gravitational pull.", "Photosynthesis in plants.",
                 "Because the Earth is round."])

README states the Moon at rank 1, prob ~0.997. We measured rank 1 at **0.9932**, identical
ordering. So the pipeline is faithful and the low scores are CLM's behaviour, not our plumbing.
That check is why the first smoke run -- five easy-intent items all answering `billing_question` --
was investigated rather than published: it looked exactly like a broken encoder and turned out to
be the model.

## Architecture, for the record

From `clm/heads.py` and the checkpoint itself:

    cfg: model Qwen/Qwen3-8B, hidden_size 4096, projection_dim 512,
         width 1536, depth 3, activation gelu, layernorm True, residual False
    logit_scale (exp): 100.81
    trained parameters: 18,887,680

Score is `exp(logit_scale) * cos(state_head(s), action_head(c))`, softmaxed over a question's
options. The state head sees the state with the question's instructions appended; the action head
sees each option's own description.

## Pinned artifacts

| | |
|---|---|
| CLM head | `Contrastive-LM/CLM-v0.1-8B` / `CLM_v0.1-8B.pt` |
| head sha256 | `b2b4a8c9c2d39263eff78a351eb909a342ce9b3bf21a3f07c1d1bf15f1c4eda5` |
| head bytes | 75,557,149 |
| encoder | `Qwen/Qwen3-8B` revision `b968826d9c46dd6066d109eabc6255188de91218` |
| `contrastive-lm` | 0.1.0 |
| torch / transformers | 2.8.0+cu128 / 4.57.1 |
| cohort | `datasets/public/easy.jsonl` + `original.jsonl` from `fstandhartinger/jevbench` (MIT), 120 items: 72 choice, 36 noul, 12 score |

## Reproducing

    python3 -m venv venv && venv/bin/pip install contrastive-lm
    # vLLM is not needed and its cu13 torch will not run on a 12.8 driver:
    venv/bin/pip install "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu128
    venv/bin/pip install "transformers==4.57.1"
    venv/bin/pip uninstall -y torchvision torchaudio torchcodec   # built for torch 2.13, break the
                                                                  # transformers import chain
    cat easy.jsonl original.jsonl > jevbench-120.jsonl
    venv/bin/python clm_arm.py jevbench-120.jsonl clm-latest.tsv \
        --local-encoder Qwen/Qwen3-8B --model clm-latest
    venv/bin/python clm_arm.py jevbench-120.jsonl clm-raw.tsv \
        --local-encoder Qwen/Qwen3-8B --model clm-raw
    python3 ../options-first/score_arm.py jevbench-120.jsonl clm-latest.tsv "clm-latest"

Validate the encoder substitution first, with `ranktest.py`. If the Moon is not rank 1 at ~0.99,
stop: the pooling is wrong and every score after it is noise.

Raw arms: `clm-latest.tsv`, `clm-raw.tsv`.

## What is still open

CLM is tagged `text-ranking` on the Hub, and `tool_selection` 0.917 is a real signal. Our catalog
has a **reranking** workload (`reranking-qualifications.json`, policy
`reranking-exact-artifact-and-latency-v3`) that this cohort does not exercise. Whether CLM is worth
having as a *reranker* is a different question with a different benchmark, and it is **not measured
here**. Nothing above should be read as an answer to it.
