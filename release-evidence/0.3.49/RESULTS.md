# Models 0.3.49 release qualification

> **CORRECTED 2026-09-29 — see [CORRECTION-2026-09-29.md](../CORRECTION-2026-09-29.md).**
> The qualification claims below overstate model quality. `correctAnswerRate` scores the pipeline, not
> the model: 69% of these attempts were answered by `EXTRACTIVE_FALLBACK`, where the harness replaced
> the generated answer with text extracted from the retrieved document. Applying this repository's own
> `RagProductionQualificationPolicy`, **five of the fourteen models qualify, not fourteen**. The
> architectures load and run real published weights; the quality numbers below do not say what they
> appear to say.


Release preparation on 2026-09-29, hours after 0.3.48. One fix: MXFP4 projections use every core
instead of one.

**What is measured here.** A local A/B of the two code paths on the matrix shape gpt-oss actually
uses. **No new model qualification is claimed**, and no qualification from 0.3.48 is retracted.

## The defect

Every quantization except MXFP4 hands its whole matrix to vectors-core's `gguf*BatchDotProduct`,
which parallelises above a one-mebibyte threshold. MXFP4 looped rows on the calling thread. Found on a
16-vCPU qualification worker running gpt-oss from a GGUF: CloudWatch showed a flat **8% CPU across
eight consecutive five-minute periods -- about one core of sixteen** -- while K-quant models on the same
instance type pegged it. The code confirmed it before the fix was written; the CPU figure is what
prompted reading the code.

## Measured effect

One gpt-oss expert matrix, 2880 x 2880, 4.2 MiB of MXFP4 weights, 30 timed projections after warmup,
on a 12-core host. Both arms go through the same public entry point and the same kernel on the same
data; only the arena differs, because a confined arena is owner-thread-only and therefore selects the
serial path:

| path | ms per projection |
|---|---|
| serial (confined arena) | 11.60 |
| threaded (shared arena) | 2.25 |
| **speedup** | **5.15x** |

Not 12x on 12 cores: the kernel dequantizes as it reads, so it is bandwidth-bound rather than
compute-bound. The 16-vCPU fleet instance is a different machine and its own figure has to be measured
there, not extrapolated from this.

## Why this cannot change an answer

Rows are split, never reductions. Each output element is an independent dot product over its own row,
so no two threads contribute to one sum and each row is folded in the order it always was. The test
asserts the threaded result is **bit-identical** to the serial one, not close -- in a backend that pins
float reduction order precisely so that one model gives one answer on every host, an approximate
assertion here would concede the thing being claimed.

## What the tests caught

The first version of the fix crashed rather than ran slowly. A `MemorySegment` from a confined arena is
readable only by its allocating thread, so a pool thread reading its rows throws
`WrongThreadException` from inside the dot product. Parallelism is now gated on the segment being
reachable from another thread, which is the same property the execution planner already gates thread
sharing on, and a confined projection stays on the calling thread at any size. Three mutations were
run against the new tests: removing the accessibility gate and dropping the calling thread's own chunk
each fail a named test. A third -- using a floor rather than a ceiling for the chunk size -- survives,
correctly: it still covers every row, one extra band at a time, so it is an equivalent implementation
and not a missed defect.

## Scope

Two catalogue models read MXFP4: gpt-oss from a GGUF, where every expert is MXFP4 and the effect is the
whole model, and Qwen3-Next, where only the shared-expert gate and up are. `Qwen3-Coder-Next`'s
published 241 ms per token was measured on exactly the code 0.3.48 shipped and stands as a measurement
of that release. Re-measuring it here would be a new measurement epoch, and is not claimed.

The scalar float activation is untouched and still deliberate. The K-quant kernels quantize the
activation to Q8 and reduce in integer arithmetic; an MXFP4 kernel doing the same must first be proven
identical to this one. This release fixes the thread factor alone.

## Test and build gates

`./gradlew build` green, with strict Javadoc, SpotBugs, dependency locks, staged publications, SBOMs
and the published-module coverage floor in the gate.
