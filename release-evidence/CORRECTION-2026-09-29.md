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

**Incoherent output.** Two of the architectures added in 0.3.48 emit text that is not language:

- `deepseek2`: `###\n###\n###...` repeated for the whole completion.
- `gemma3n`: `You are asking a question, and Y NOTE:\n- **PLEASE NOTE: SEE ALLH1-XXH*C++*XHHN*mll*nnssC++`

Both have `rawCorrectAnswerRate` of 0.0. `qwen3next` degenerates differently -- *"The question asks for
the amount of the claim and what is the amount of the claim."* -- at 11.1%.

**`deepseek2` had a wrong prompt template AND a broken decoder. The template was not the cause.**

The harness did render the V1 `### Instruction:` / `### Response:` format while the published GGUF's own
`tokenizer.chat_template` for DeepSeek-Coder-V2-Lite-Instruct renders `{{ 'User: ' + content + '\n\n' }}`
and `'Assistant: '`. That was a real defect and is fixed, as a new `deepseek-v2` template rather than an
edit to `deepseek`, whose V1 coder models are already qualified against the old markers with their
greedy oracles pinned on them.

It was recorded here, briefly, as the diagnosis. It was not. Re-run on the corrected template with a
256-token cap (shard-32, `models@0.3.49+rerun12-1e0f2fed6152`), the model emits **pure newline tokens
for the entire completion** -- `truncatedAnswerRate` 1.000, `modelAnswerRate` 0.000, 24 of 27 attempts
falling back and 3 abstaining on retrieval. `###` under one prompt and `\n` under another are two
shapes of the same degeneracy. **The deepseek2 decoder is defective on real weights**, and the
template fix, while correct, changed nothing about that.

**`gemma3n` and `qwen3next` are not explained by prompting either.** The `gemma` template this run used matches
gemma3n's own turn markers (`<start_of_turn>user` / `<end_of_turn>` / `<start_of_turn>model`), read from
the published GGUF, and `chatml` matches Qwen's. Those two remain open, and a decoder defect is the
leading explanation rather than a confirmed one.

**Truncated at the output cap -- a harness setting, not a model.** The run used
`maxOutputTokens = 64`. Every failing attempt examined stopped at exactly 64 output tokens, mid-sentence
or mid-word:

- `gpt-oss` ended on *"...Windshield repair has a 75 dollar deductible"* -- correct, and cut off before
  it could emit the citation the grounding policy screens for.
- `qwen3-30b-a3b` ended on *"...must be reported through the Aurora portal within "*, having spent most
  of the budget on a `<think>` reasoning trace.

A thinking model cannot reach an answer in 64 tokens, and no model can emit prose followed by a
citation in what is left. `unsloth_qwen3_6_35b_a3b` reaches 77.8% raw correct with 0% contribution for
the same reason. This accounts for most of the nine failures and is a configuration defect in how the
campaign was run, not a property of the models.

`extractiveFallbackRate` and `truncatedAnswerRate` are now both part of the summary and the CLI's
output, so a run cut off at the cap is visible as truncation rather than indistinguishable from a model
with nothing to say.

**Instruction-following.** `ornith-1.0-35b` echoed the prompt's own instructions back
(*"...One short sentence. Copy source ID exactly from brackets."*) rather than answering. That one is a
model-quality observation and stands.

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
