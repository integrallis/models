# Generation safety: repetition-loop detector on real greedy outputs

Date: 2026-09-16. Branch `feat/generation-safety`. Every number below was **measured here** unless
marked otherwise. The full per-run table, with the configuration header, is in
[`raw-report.md`](raw-report.md). It was written by the harness itself; nothing in it was edited by hand.

## What was run

```
./gradlew :backend-java:generationSafetyLoopExperiment
# optional: -PgenerationSafety.modelDirectory=... -PgenerationSafety.models=a.gguf,b.gguf
#           -PgenerationSafety.maxTokens=200 -PgenerationSafety.contextLength=512
#           -PgenerationSafety.report=path/to/report.md
```

Harness: `backend-java/src/test/java/com/integrallis/models/backend/purejava/GenerationSafetyLoopExperiment.java`.
For each (model, prompt) pair it runs `GenerationLoop` (sequential path, pure-Java backend) once with detection
**off** (the baseline) and once for each detector configuration. It records the stop reason, completion
tokens, `GenerationLoop.repetitionLoopStops()`, whether the detector-on text is an exact prefix of the
baseline text, and the last 60 characters at the stop.

Configuration (the same values are also written at the top of `raw-report.md`):

| setting | value |
|---|---|
| host | macOS, `os.arch` x86_64, JDK 25.0.3, shared developer machine (not a controlled benchmark host) |
| sampling | temperature 0 (greedy), repetitionPenalty 1.0, minP 0, no stop sequences |
| maxTokens / context | 200 / 512 |
| detector `off` | `RepetitionLoopDetection.disabled()` |
| detector `A` | maxSpan 32, minRepeats 4, minLoopTokens 16 |
| detector `B` | maxSpan 64, minRepeats 3, minLoopTokens 32 |
| detector `C` | maxSpan 16, minRepeats 10, minLoopTokens 64 |

Models: the locally cached pinned GGUF fixtures. The sha256 of every file is recorded in `raw-report.md` and matches
`backend-java/src/test/resources/model-fixtures.properties`: Qwen3-0.6B Q4_0, SmolLM2-360M-Instruct Q8_0,
Qwen2.5-Coder-0.5B-Instruct Q4_0, TinyLlama-1.1B-Chat v1.0 Q4_0. Prompts are raw text with no chat template.

**Data provenance caveat:** the six prompts were written for this check. They are not a published dataset. P1 and P4
*instruct* the model to repeat. P2 invites a legitimate list. The others are open-ended. This is a smoke-level
check of mechanism and behaviour on real outputs, **not** a benchmark of loop prevalence or detector
precision/recall. Those need published data and a labelled notion of "degenerate", neither of which exists here.

## Results (measured)

96 generations = 4 models x 6 prompts x 4 configurations. Run time was 54 min 40 s on the host above. Wall time
was not a measured quantity and no throughput claim is made.

1. **Detection never perturbed the text it did not cut.** 96/96 runs had detector-on output that was an
   exact prefix of the detector-off output for the same model and prompt.
2. **The switch is observable.** Every `REPETITION_LOOP` stop coincided with `repetitionLoopStops() == 1` on a
   fresh loop. Every run without the stop reported 0. With detection off, no run stopped with `REPETITION_LOOP`.
3. Triggers by configuration, out of 24 (model, prompt) pairs each: A 12, B 12, C 10. The baseline stopped with MAX_TOKENS in 18 pairs and EOS in 6. No EOS-terminated baseline was cut short by any configuration.

| model | prompt | off (baseline) | A | B | C | reading of the tail at the stop |
|---|---|---|---|---|---|---|
| Qwen3 0.6B | P0 fox | MAX_TOKENS 200 | LOOP @107 | LOOP @83 | MAX 200 | degenerate "The dog is a dog. The fox is a fox." cycle |
| all 4 | P1 instructed `_1_2_3_4` | MAX_TOKENS 200 | LOOP @32 | LOOP @32 | LOOP @80 | instructed repetition. The detector cannot tell it from a loop |
| all 4 | P4 "I am a robot." | MAX_TOKENS 200 | LOOP @16-20 | LOOP @32 | LOOP @64 | instructed repetition, as above |
| Qwen2.5-Coder 0.5B | P0 fox | MAX_TOKENS 200 | LOOP @112 | LOOP @92 | MAX 200 | degenerate "jumps over the lazy dog once more" cycle |
| Qwen2.5-Coder 0.5B | P3 poem | MAX_TOKENS 200 | LOOP @107 | LOOP @92 | LOOP @197 | degenerate "peaceful and peaceful place, with no worries or worries" |
| TinyLlama 1.1B | P2 list 1..50 | MAX_TOKENS 200 | LOOP @40 | LOOP @32 | LOOP @100 | degenerate `1..5` cycle instead of counting to 50 |
| TinyLlama 1.1B | P0 fox | MAX_TOKENS 200 | MAX 200 | MAX 200 | MAX 200 | numbered repeats ("13. The quick brown fox ..."): ordinals change, so correctly **not** a loop |
| SmolLM2 360M | P2 list 1..50 | EOS 148 | EOS 148 | EOS 148 | EOS 148 | legitimate list to 50: not triggered |

Reading of the tails is **believed** (a human judgement of 60-character tails), not measured.

## What this does and does not show

- It shows that on the sequential path the detector stops only by truncation (prefix property), that it reports
  and counts every stop, and that on these outputs it fired on the obvious degenerate cycles. It did not fire on
  either legitimate numbered list (TinyLlama P0, SmolLM2 P2).
- It **cannot** show precision or recall. Instructed repetition (P1, P4) is exact periodicity and is stopped by
  design. That limitation is documented on `RepetitionLoopDetection` and in the unit tests. Whether any
  threshold is a good default is **no data**, which is why detection stays off by default.
- It did not exercise the speculative or continuous-batching paths on a real model. Those paths are covered by
  unit tests on synthetic backends only.
- Per-token cost is established by the unit test
  `RepetitionLoopDetectorTest.workPerTokenIsBoundedByTheWindowIndependentOfGenerationLength` (exactly
  `min(maxSpan, tokensSoFar)` comparisons per token over 200,000 tokens), not by a timing measurement.
- Incidental (measured, not investigated): the resolved end-of-generation set for both Qwen GGUFs includes id
  128247 besides 151643/151645. That comes from the pre-existing GGUF vocabulary-text heuristics, which this
  change did not modify.
