# Candidate inventory: what is published, what failed, what has never been looked at

Regenerate with:

```
python3 scripts/fleet/candidate-inventory.py --catalog ../model-jars/catalog --format md
python3 scripts/fleet/candidate-inventory.py --catalog ../model-jars/catalog \
    --check-doc docs/CANDIDATE-INVENTORY.md
```

The catalog carries far more candidates than qualifications, and until this inventory existed the
difference was undocumented as a set. "Tried and failed" and "never looked at" are different facts
with different consequences, and publishing nulls is a house rule — a null nobody can enumerate is
not published.

| bucket | models | meaning |
| --- | --- | --- |
| **QUALIFIED** | 101 | a manifest row with `qualified` true |
| **REJECTED** | 1 | a manifest row with `qualified` false; the verdict is the finding |
| **EVALUATED_NOT_LANDED** | 19 | committed evidence names it, no manifest row claims it |
| **NOT_EVALUATED** | 336 | nothing in the catalog or the evidence tree mentions it |
| | **457** | catalog candidates |

A model that appears in several manifests is counted once, and **qualified anywhere wins**: four
models sit as unqualified rows in `tool-qualifications.json` while being QUALIFIED on another
track, and counting those as rejections would misreport them. An ad-hoc count made before this
tool existed reported 5 rejections for exactly that reason; the correct figure is the one above.

## Rejected — the verdict is the finding

```
REJECTED (1):
  h2oai_h2o_danube3_500m_chat_gguf_q4_k_m                  FAILED_MODEL_CONTRIBUTION_GATE
```

## Evaluated, not landed

Committed evidence names these, but no manifest row claims them. Two distinct reasons, and the
distinction matters:

- **Measured and failed.** The max-output-tokens sweep entries are published nulls: raising the cap
  from 256 to 768 cleared truncation entirely and did not change a single verdict, so those models
  genuinely do not contribute on their corpora. `benchmark-results/2026-10-09-max-tokens-sweep`
  holds the raw artifacts and `derive.py` re-derives every number. `fin_r1_7b_q4_k_m` is the
  exception and is recorded as **inconclusive**, not failed: it still truncated 56% of answers at
  the higher cap while every answer it finished was correct, so the cap was still the binding
  constraint and a third cap is owed.
- **Measured and passed on an unreleased build.** The embedding entries reproduce against the
  llama.cpp oracle but their reports name a `-dev` version, and a dev build is not publishable
  evidence. They land when re-measured against a release.

```
EVALUATED_NOT_LANDED (19):
  bartowski_mathstral_7b_v0_1_gguf_q4_k_s                  evidence under 2026-10-09-max-tokens-sweep
  bartowski_qwen2_5_math_7b_instruct_gguf_q4_k_m           evidence under 2026-10-09-max-tokens-sweep
  fin_r1_7b_q4_0                                           evidence under 2026-10-09-max-tokens-sweep
  fin_r1_7b_q4_k_m                                         evidence under 2026-10-09-max-tokens-sweep
  huatuogpt_o1_7b_q4_0                                     evidence under 2026-10-09-max-tokens-sweep
  huatuogpt_o1_7b_q4_k_s                                   evidence under 2026-10-09-max-tokens-sweep
  liquidai_lfm2_5_embedding_350m_gguf_q4_k_m               evidence under embedding
  liquidai_lfm2_5_embedding_350m_gguf_q5_k_m               evidence under embedding
  liquidai_lfm2_5_embedding_350m_gguf_q6_k                 evidence under embedding
  phi_4_mini_instruct_q4_0                                 evidence under 2026-10-09-max-tokens-sweep
  phi_4_mini_instruct_q4_k_m                               evidence under 2026-10-09-max-tokens-sweep
  qwen2_5_math_1_5b_instruct_q4_0                          evidence under 2026-10-09-max-tokens-sweep
  qwen2_5_math_1_5b_instruct_q4_k_s                        evidence under 2026-10-09-max-tokens-sweep
  qwen_qwen3_embedding_4b_gguf_q4_k_m                      evidence under embedding
  second_state_all_minilm_l6_v2_embedding_gguf_q5_k_m      evidence under embedding
  second_state_all_minilm_l6_v2_embedding_gguf_q5_k_s      evidence under embedding
  second_state_nomic_embed_text_v1_5_embedding_gguf_q4_0   evidence under embedding
  second_state_nomic_embed_text_v1_5_embedding_gguf_q4_k_m evidence under embedding
  second_state_nomic_embed_text_v1_5_embedding_gguf_q4_k_s evidence under embedding
```

## Not evaluated

The remainder. This is **"no data", not "no effect"** — nothing in the catalog or the evidence tree
mentions them, so nothing is known about how they behave on this harness. They are candidates
because their headers were read and their licenses checked, which is the bar for entering the
catalog; it is not a claim about quality.

The inventory does not rank them. Ordering is the campaign's job, and the founding principle is
smallest-first, so a 0.5 GB candidate is worth a box before a 7 B one.

## What this inventory is not

It does not say a NOT_EVALUATED model would fail, and it does not say a REJECTED one is a bad
model. A rejection here is a verdict on one artifact, at one quantization, against one corpus,
under one grounding policy, measured on one host. `huatuogpt_o1_7b_q4_k_m` is QUALIFIED on
`general` while its Q4_K_S sibling fails on `healthcare`; `eurollm-1.7b` failed on `general`
and qualified on `multilingual` with workload the only variable changed. The workload is not
predictable from the model's name in either direction, which is why the inventory records what was
measured rather than what should be expected.
