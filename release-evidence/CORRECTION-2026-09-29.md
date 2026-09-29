# Correction: the 0.3.48 and 0.3.49 qualification claims overstate model quality

Filed 2026-09-29, the same day both releases shipped. **Nothing in the released artifacts is wrong.**
The code is unaffected; what was wrong is the evidence written beside it.

## What was claimed

`release-evidence/0.3.48/RESULTS.md`, `release-evidence/0.3.49/RESULTS.md` and the matching CHANGELOG
entries state that fourteen models were qualified, "every one at 1.000 on all six quality metrics over
27/27 attempts with zero failures".

## What was actually measured

`correctAnswerRate` scores the **pipeline**, not the model. Under the grounding policy these runs used
(`bounded-context-injection-screened-citation-safe-statement-grounding-v20`), a generated answer that
fails citation screening is replaced with text extracted from the retrieved document, and the
replacement is what gets scored. Across all 14 models and 378 attempts:

| grounding decision | attempts | share |
|---|---|---|
| `EXTRACTIVE_FALLBACK` (the harness answered) | 261 | **69%** |
| `MODEL_ANSWER` | 75 | 20% |
| `RETRIEVAL_ABSTENTION` | 42 | 11% |

Applying `RagProductionQualificationPolicy` -- this repository's own gate, requiring a model-answer
rate of at least 1/3 and at least 90% of those answers correct -- **five of the fourteen qualify, not
fourteen**:

| model | arch | model-answered | of those correct | extractive fallback | raw correct | verdict |
|---|---|---|---|---|---|---|
| `gemma_4_e4b_it_q4_0` | `gemma4` | 44.4% | 100.0% | 44.4% | 100.0% | **QUALIFIED** |
| `liquidai_lfm2_5_1_2b_instruct_gguf_q4_k_m` | `lfm2` | 44.4% | 100.0% | 44.4% | 66.7% | **QUALIFIED** |
| `bartowski_kwaipilot_kat_coder_v2_5_dev_gguf_q4_k_m` | `qwen35moe` | 55.6% | 100.0% | 33.3% | 100.0% | **QUALIFIED** |
| `unsloth_qwen3_30b_a3b_instruct_2507_gguf_q4_k_m` | `qwen3moe` | 77.8% | 100.0% | 11.1% | 100.0% | **QUALIFIED** |
| `unsloth_qwen3_coder_30b_a3b_instruct_gguf_q4_k_m` | `qwen3moe` | 55.6% | 100.0% | 33.3% | 100.0% | **QUALIFIED** |
| `bartowski_deepseek_coder_v2_lite_instruct_gguf_q4_k_m` | `deepseek2` | 0.0% | 0.0% | 88.9% | 0.0% | not qualified |
| `gemma_3n_e2b_it_q8_0` | `gemma3n` | 0.0% | 0.0% | 88.9% | 0.0% | not qualified |
| `lmstudio_community_gemma_4_e2b_it_gguf_q4_k_m` | `gemma4` | 0.0% | 0.0% | 88.9% | 0.0% | not qualified |
| `unsloth_gpt_oss_20b_gguf_q4_k_m` | `gpt-oss` | 0.0% | 0.0% | 88.9% | 44.4% | not qualified |
| `deepreinforce_ai_ornith_1_0_35b_gguf_q4_k_m` | `qwen35moe` | 0.0% | 0.0% | 88.9% | 0.0% | not qualified |
| `unsloth_qwen3_5_35b_a3b_gguf_q4_k_m` | `qwen35moe` | 0.0% | 0.0% | 88.9% | 0.0% | not qualified |
| `unsloth_qwen3_6_35b_a3b_gguf_ud_q4_k_m` | `qwen35moe` | 0.0% | 0.0% | 88.9% | 77.8% | not qualified |
| `unsloth_qwen3_30b_a3b_gguf_q4_k_m` | `qwen3moe` | 0.0% | 0.0% | 88.9% | 0.0% | not qualified |
| `unsloth_qwen3_coder_next_gguf_q4_k_m` | `qwen3next` | 0.0% | 0.0% | 88.9% | 11.1% | not qualified |

## Why nothing caught it

`RagBenchmarkSummary` published `correctAnswerRate` and did not publish the model-contribution rates.
Those existed only inside `runs[]`, so every consumer of a summary -- a report, a release note, a
person reading either -- saw the pipeline's score with nothing beside it to indicate that the model
had not produced the answer. The qualification policy was never wrong; it was never applied to these
runs, and the summary gave no hint that it needed to be.

Fixed in the same commit as this correction: `rawCorrectAnswerRate`, `modelAnswerRate`,
`modelAnswerCorrectRate` and `extractiveFallbackRate` are now part of the summary and of the CLI's
one-line output, and `RagStatisticsTest` pins the case where a model contributing nothing still
reports 1.000.

## What the raw output shows, which is the more serious finding

Reading the models' own text, before substitution, splits the nine failures three ways.

**Incoherent output -- a probable decoder defect.** Two of the architectures added in 0.3.48 emit text
that is not language:

- `deepseek2`: `###\n###\n###...` repeated for the whole completion.
- `gemma3n`: `You are asking a question, and Y NOTE:\n- **PLEASE NOTE: SEE ALLH1-XXH*C++*XHHN*mll*nnssC++`

Both have `rawCorrectAnswerRate` of 0.0. `qwen3next` degenerates differently -- *"The question asks for
the amount of the claim and what is the amount of the claim."* -- at 11.1%. These are not weak models;
this is what a broken decode looks like, and the extractive fallback hid it behind a perfect score.

**Correct content, screened out.** `unsloth_qwen3_6_35b_a3b` reaches 77.8% raw correct and 0% model
contribution; `gpt-oss` 44.4% and 0%. The model answered well and the answer was rejected, most often
for a missing citation.

**Reasoning traces not stripped.** `qwen3moe`, `qwen35moe` and `gemma4` E2B emit `<think>...` or
`Thinking Process:` before answering. `kat-coder` returns `<think></think>You have 30 calendar
days...` -- a correct answer failing screening on a leaked wrapper. This is a harness defect, not a
model one, and it is why several entries read 0% raw correct despite answering sensibly.

## Standing after this correction

- Qualified on published weights, on their own output: **`lfm2`, `qwen3moe`, `qwen35moe`**, and
  `gemma4` E4B. Five models.
- **Not qualified**: `qwen3next`, `deepseek2`, `gemma3n`, `gpt-oss`, `mistral3`, and the nine model
  entries above.
- The eight architectures in 0.3.48 are still implemented, still unit-verified, and still load real
  published weights end to end without failing a run. That part of both releases stands. What does not
  stand is any claim about the quality of what they generate.

No published qualification record is retracted, because none was ever contributed to a catalogue from
these runs. The claim existed only in these release notes, and this file is the correction.
