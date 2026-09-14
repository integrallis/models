# V18 cold activated-inference determinism finding

Status: **fixed in the Java execution kernel; production-Q4 calibration must be repeated on the
final released dependency and Models revision**.

The first live-screen launcher stopped before scoring because one frozen BFCL case contained a
legitimate prior `assistant` message. After the loader fix, the exposed calibration was repeated at
Models revision `9f1e5c60fda30ee1ec036d5d1decb9d76cd76527`. It passed the declared gate, but an
independent comparison with the first calibration found one numerical difference:

| Case | First process | Second process | Difference |
| --- | ---: | ---: | ---: |
| `irrelevance_126` | 1.9912996184850087 | 2.031763056986989 | 0.04046343850198042 |

The other 74 scores were bit-identical. The threshold and classification were unchanged. The
second report and raw log have SHA-256 values
`93359d5ca1ca0fd08351c90f41afe78d79e496d2fb6961b4dfc0d63f44a12a1c` and
`73c9bbd1ee088915b08a377ac2b40ec569751dd489b84e549ca7ed8640b48b9e`, respectively.
Those files are diagnostic artifacts, not qualification evidence, because the cold discrepancy
invalidated advancement to the live screen.

## Targeted experiments

The exact Qwen3 1.7B Q4_K_M model, V9 adapter, V16 applicability head, and exposed prompt were then
run in-process on Temurin Java 25.0.3, an Intel Core i7-9750H, and the Java backend. No external
inference engine was used.

| Arm | First score | Second score | Third score | Finding |
| --- | ---: | ---: | ---: | --- |
| Shared activated prefix | 1.5396118858220120 | 1.2482542202453004 | 1.2482542202453004 | Cold activated result differed; later results were exact. |
| Shared, parallel GGUF disabled | 1.5446572603973294 | 1.2482542202453004 | not run | Parallel scheduling was not the cause. |
| Recomputed prefix | 1.5705720856937830 | 1.2482542202453004 | not run | Physical KV sharing was not the cause. |
| Base model without adapter | 20.955594939049377 | 20.955594939049377 | not run | Base Q4 inference, reset, and the head were stable. |

The first/second real-weight hidden-state regression test failed before the fix. A test-first
intermediate intervention prewarmed the seven mapped-F32 LoRA projection shapes at adapter load.
That made the hidden states exact and a complete 75-case diagnostic again passed at calls `47/50`,
no-calls `24/25`, balanced accuracy `0.95`, and physical sharing `75/75`. Its report SHA-256 is
`314d5bf277feafdbc5ed0a7fb1e843713b9c88c090acda5a89067ef0b5bc8817`. The result proved the
localization, but the prewarm is diagnostic—not the retained implementation and not qualification
evidence.

A standalone Vectors fresh-JVM probe then reproduced the cold/hot mismatch without model state,
cache reuse, or parallel execution. The mapped F32 kernel ended each vector accumulator with
`FloatVector.reduceLanes(ADD)`. Java does not specify the lane-addition order for that reduction;
the interpreted and compiled paths used different legal arithmetic trees.

Vectors 0.1.21 provides an owned `F32ExecutionMatrix` with an explicit shuffle/add tree, a
little-endian `MemorySegment` factory, and four-row single-input execution. Models copies each
validated LoRA A/B tensor once into this immutable Java layout and removes the prewarm. Fresh-JVM
tests compare raw output bits before and after 256 calls at 64-, 128-, and 256-bit vector ceilings,
including the real `32x2048` and `2048x32` adapter shapes. The Models real-weight first/second
hidden-state regression remains the end-to-end gate.

In a controlled three-fork JMH run, the combined rank-up/rank-down pair changed from `15.651 us`
to `11.662 us`, 25.49% faster on the measured Intel host. This is a host-specific performance
result, not an Apple Silicon claim. The durable fix is entirely in-process Java; no external
inference runtime or Rust kernel is used.

## Admission consequence

The BFCL-live window remained unscored. V18 returns to Phase 1 and must repeat all 75 exposed cases
under the final Models revision resolving Vectors 0.1.21 from Maven Central. That report and hash
must be committed before a new live-screen attempt. No pre-fix or prewarm-dependent score or
threshold is carried forward.
