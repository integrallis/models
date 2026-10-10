# Candidate inventory: qualifications, failures and evidence still to review

Regenerate with:

```
python3 scripts/fleet/candidate-inventory.py --catalog scripts/fleet/fixtures/candidate-catalog --format md
python3 scripts/fleet/candidate-inventory.py --catalog scripts/fleet/fixtures/candidate-catalog \
    --check-doc docs/CANDIDATE-INVENTORY.md
```

CI uses the committed catalog fixture from ModelJars
`eb68debc29d30db24e9b7f99ecbdb5a3544f802f`. Its `source.json` records the source revision and file
hashes. Update the fixture and regenerate the counts together when adopting catalog changes.
Models never checks out or builds its downstream project. The fixture is required, and a missing
catalog fails the check rather than skipping it. Evidence is read from Git objects at Models `HEAD`, so
local downloads, staged reports and edits cannot change the result.

The matcher parses valid JSON and matches complete `modelId` fields (including nested verdict
records). Older reports without those fields use their exact filename stem. It normalizes only
underscore/hyphen spelling, never substrings or guessed display-name aliases. A matching artifact
is a pointer to evidence to inspect; it does not establish a successful run or qualification.

The catalog carries far more candidates than qualifications, and until this inventory existed the
difference was undocumented as a set. "Tried and failed" and "never looked at" are different facts
with different consequences, and publishing nulls is a house rule — a null nobody can enumerate is
not published.

| bucket | models | meaning |
| --- | --- | --- |
| **QUALIFIED** | 101 | a manifest row with `qualified` true |
| **REJECTED** | 1 | a manifest row with `qualified` false; the verdict is the finding |
| **EVALUATED_NOT_LANDED** | 19 | committed evidence names it, no manifest row claims it |
| **NOT_EVALUATED** | 336 | no manifest row or matching committed JSON artifact found |
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

Committed evidence names these, but no manifest row claims them. This bucket alone does not say
whether a run passed. The reports distinguish these findings:

- **Measured and failed.** The max-output-tokens sweep entries are published nulls: raising the cap
  from 256 to 768 cleared truncation entirely and did not change a single verdict, so those models
  genuinely do not contribute on their corpora. `benchmark-results/2026-10-09-max-tokens-sweep`
  holds the raw artifacts and `derive.py` re-derives every number. `fin_r1_7b_q4_k_m` is the
  exception and is recorded as **inconclusive**, not failed: it still truncated 56% of answers at
  the higher cap while every answer it finished was correct, so the cap was still the binding
  constraint and a third cap is owed.
- **Embedding oracle failures.** All nine embedding reports listed below record `qualified: false`:
  their minimum oracle cosine falls below the recorded `0.999` floor. They are failed measurements,
  not successful qualifications waiting only for a release. A new run must satisfy the gate before
  admission; changing the runtime version label cannot repair those results.

```
EVALUATED_NOT_LANDED (19):
  bartowski_mathstral_7b_v0_1_gguf_q4_k_s                  evidence: 2026-10-09-max-tokens-sweep/raw/mt256-math-shard624/bartowski_mathstral_7b_v0_1_gguf_q4_k_s.comparator.json, 2026-10-09-max-tokens-sweep/raw/mt256-math-shard624/bartowski_mathstral_7b_v0_1_gguf_q4_k_s.json
  bartowski_qwen2_5_math_7b_instruct_gguf_q4_k_m           evidence: 2026-10-09-max-tokens-sweep/raw/mt256-math-shard624/bartowski_qwen2_5_math_7b_instruct_gguf_q4_k_m.comparator.json, 2026-10-09-max-tokens-sweep/raw/mt256-math-shard624/bartowski_qwen2_5_math_7b_instruct_gguf_q4_k_m.json
  fin_r1_7b_q4_0                                           evidence: 2026-10-09-max-tokens-sweep/raw/mt256-finance-shard626/fin_r1_7b_q4_0.comparator.json, 2026-10-09-max-tokens-sweep/raw/mt256-finance-shard626/fin_r1_7b_q4_0.json
  fin_r1_7b_q4_k_m                                         evidence: 2026-10-09-max-tokens-sweep/raw/mt256-finance-shard626/fin_r1_7b_q4_k_m.comparator.json, 2026-10-09-max-tokens-sweep/raw/mt256-finance-shard626/fin_r1_7b_q4_k_m.json
  huatuogpt_o1_7b_q4_0                                     evidence: 2026-10-09-max-tokens-sweep/raw/mt256-healthcare-shard627/huatuogpt_o1_7b_q4_0.comparator.json, 2026-10-09-max-tokens-sweep/raw/mt256-healthcare-shard627/huatuogpt_o1_7b_q4_0.json
  huatuogpt_o1_7b_q4_k_s                                   evidence: 2026-10-09-max-tokens-sweep/raw/mt256-healthcare-shard627/huatuogpt_o1_7b_q4_k_s.comparator.json, 2026-10-09-max-tokens-sweep/raw/mt256-healthcare-shard627/huatuogpt_o1_7b_q4_k_s.json
  liquidai_lfm2_5_embedding_350m_gguf_q4_k_m               evidence: embedding/liquidai-lfm2-5-embedding-350m-gguf-q4-k-m.json
  liquidai_lfm2_5_embedding_350m_gguf_q5_k_m               evidence: embedding/liquidai-lfm2-5-embedding-350m-gguf-q5-k-m.json
  liquidai_lfm2_5_embedding_350m_gguf_q6_k                 evidence: embedding/liquidai-lfm2-5-embedding-350m-gguf-q6-k.json
  phi_4_mini_instruct_q4_0                                 evidence: 2026-10-09-max-tokens-sweep/raw/mt256-math-shard624/phi_4_mini_instruct_q4_0.comparator.json, 2026-10-09-max-tokens-sweep/raw/mt256-math-shard624/phi_4_mini_instruct_q4_0.json
  phi_4_mini_instruct_q4_k_m                               evidence: 2026-10-09-max-tokens-sweep/raw/mt256-math-shard624/phi_4_mini_instruct_q4_k_m.comparator.json, 2026-10-09-max-tokens-sweep/raw/mt256-math-shard624/phi_4_mini_instruct_q4_k_m.json
  qwen2_5_math_1_5b_instruct_q4_0                          evidence: 2026-10-09-max-tokens-sweep/raw/mt256-math-shard624/qwen2_5_math_1_5b_instruct_q4_0.comparator.json, 2026-10-09-max-tokens-sweep/raw/mt256-math-shard624/qwen2_5_math_1_5b_instruct_q4_0.json
  qwen2_5_math_1_5b_instruct_q4_k_s                        evidence: 2026-10-09-max-tokens-sweep/raw/mt256-math-shard624/qwen2_5_math_1_5b_instruct_q4_k_s.comparator.json, 2026-10-09-max-tokens-sweep/raw/mt256-math-shard624/qwen2_5_math_1_5b_instruct_q4_k_s.json
  qwen_qwen3_embedding_4b_gguf_q4_k_m                      evidence: embedding/qwen-qwen3-embedding-4b-gguf-q4-k-m.json
  second_state_all_minilm_l6_v2_embedding_gguf_q5_k_m      evidence: embedding/second-state-all-minilm-l6-v2-embedding-gguf-q5-k-m.json
  second_state_all_minilm_l6_v2_embedding_gguf_q5_k_s      evidence: embedding/second-state-all-minilm-l6-v2-embedding-gguf-q5-k-s.json
  second_state_nomic_embed_text_v1_5_embedding_gguf_q4_0   evidence: embedding/second-state-nomic-embed-text-v1-5-embedding-gguf-q4-0.json
  second_state_nomic_embed_text_v1_5_embedding_gguf_q4_k_m evidence: embedding/second-state-nomic-embed-text-v1-5-embedding-gguf-q4-k-m.json
  second_state_nomic_embed_text_v1_5_embedding_gguf_q4_k_s evidence: embedding/second-state-nomic-embed-text-v1-5-embedding-gguf-q4-k-s.json
```

## Not evaluated

The remainder. This is **"no matching evidence found", not "no effect"** — no manifest row or valid
committed JSON matches their complete ID under the rules above. Reports under other aliases,
non-JSON notes, uncommitted runs or external evidence are outside this inventory; absence here does
not prove a model was never tested. They are candidates
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
