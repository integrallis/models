# Models 0.3.50 release qualification

Release preparation on 2026-09-30. This patch is corrective: it fixes how qualification evidence is
reported, two decoder defects, and a crash in shared code. **No new model qualification is claimed, and
no existing one is retracted.**

## Why this release exists

0.3.48 and 0.3.49 shipped with release notes claiming fourteen models qualified at 1.000 on every
quality metric. That was a pipeline metric reported as model quality: 69% of those 378 attempts were
answered by `EXTRACTIVE_FALLBACK`, where the harness replaces a generated answer that fails citation
screening with text extracted from the retrieved document. Under this repository's own
`RagProductionQualificationPolicy`, five of the fourteen clear the model-contribution gate.

The full correction is `release-evidence/CORRECTION-2026-09-29.md`; the process analysis, including why
no control caught it, is `docs/findings/qualification-process-failure.md`.

## What is fixed

**Reporting.** `RagBenchmarkSummary` now publishes `rawCorrectAnswerRate`, `modelAnswerRate`,
`modelAnswerCorrectRate`, `extractiveFallbackRate` and `truncatedAnswerRate`, and the CLI prints the
contribution rates and the truncation rate beside `correct=`. Those numbers existed only inside
`runs[]`, so a summary could report a perfect score for a model that contributed nothing and no reader
had anything to contradict it. `RagStatisticsTest` pins that exact case, and reverting either the
raw-evaluation read or the fallback count fails it by name.

**The output cap defaults to 256, not 64.** The campaign never chose 64; it inherited the default. At
64, most attempts were truncated: gpt-oss was cut off mid-word on a correct answer before it could emit
the citation the grounding policy screens for, and a thinking model spent its whole budget on a
reasoning trace. A truncated answer cannot carry a citation, so it failed screening and was replaced.

**Grounding policy v21: a reasoning trace is no longer scored as the answer.** Screening a `<think>`
block fails on the block's own terms -- it reasons aloud, so it states things the documents do not
support, and the answer after it is never examined. A closed `<think>`, `<thinking>` or `<reasoning>`
block is removed before screening, with `rawText` still carrying the full generation. The policy id
becomes v21 because this changes which decision a completion receives; records written under v20 keep
their meaning.

**DeepSeek-V2 models were prompted with the V1 format.** The harness rendered
`### Instruction:` / `### Response:` while DeepSeek-Coder-V2-Lite-Instruct's own
`tokenizer.chat_template` uses `User: ` / `Assistant: `. A new `deepseek-v2` template fixes it;
`deepseek` is untouched, because the V1 coder models are qualified against those markers with their
greedy oracles pinned on them.

**GELU could read past the end of the tanh table.** `ArrayIndexOutOfBoundsException: Index 65537 out of
bounds for length 65537`, which killed a gemma3n run 456 seconds in. A value one float ulp below the
table's limit still scales onto the final slot, so the interpolation read one past it. Shared code: any
architecture whose feed-forward uses GELU could reach it. No existing measurement changes, because the
only affected inputs previously threw.

**gemma3n never applied its final logit softcap.** The reference's graph ends in SCALE, TANH, SCALE
around the output projection -- `30 * tanh(logit / 30)` -- and gemma3n's published header declares no
softcapping key, so the reference's default of 30 applies. Confirmed against a dump of that graph: raw
`-15.6992` leaves as `-14.4074`, and `30 * tanh(-15.6992 / 30)` is `-14.41`. The formula now lives in
`TensorOps.softcap`, with gemma4 delegating to it. **This is not why gemma3n produces corrupted text**:
the transform is monotonic, so it cannot change a greedy argmax, and the generated text is identical
before and after.

## Defects that remain open, measured against an external oracle

llama.cpp was used as a test oracle only -- nothing of it is linked or copied. On the published
gemma-3n-E2B-it Q8_0 GGUF, sha256-verified identical to the artifact the fleet ran, the oracle answers
`" Paris.\n\nThis is a true statement."` and this decoder answers `" Paris, France d'Aquiisiturii-"`.
Prompt tokenisation matches exactly and the first generated token matches; the raw logits do not,
differing by roughly a factor of two with sign flips.

Checked against the reference and found correct, so excluded as causes: the prompt template and
tokenisation, the `sqrt(n_embd)` embedding scale, feed-forward order, activation sparsity's presence and
its `relu(x - (mean + k*std))` with the `n - 1` denominator, the stream merge, the L2 magnitude and
rescale, and the KV ring capacity.

- `deepseek2` emits only newline tokens for an entire completion once prompted with its own template.
- `gemma3n` emits corrupted text.
- `qwen3next` degenerates into repetition.

Locating these needs a per-layer tensor comparison against the oracle's graph dump. Not done.

## Test and build gates

`./gradlew build` green, with strict Javadoc, SpotBugs, dependency locks, staged publications, SBOMs and
the published-module coverage floor in the gate.
