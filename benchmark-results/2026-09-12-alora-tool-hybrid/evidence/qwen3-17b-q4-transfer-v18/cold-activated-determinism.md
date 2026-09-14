# V18 cold activated-inference determinism finding

Status: **fixed in Java; production-Q4 calibration must be repeated on the fixed revision**.

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

The first/second real-weight hidden-state regression test failed before the fix. Models now
prewarms the seven mapped-F32 LoRA projection shapes at adapter load with neutral Java arrays. The
test then passed with bit-identical first and second activated hidden states. Unit coverage also
proves prewarming does not change a subsequent projection. The fix is revision
`4ebbc471553d5956970091d72f08fd79ad835a4a`.

The evidence localizes the behavior to the cold Activated-LoRA F32 projection path. Its dependence
on whether the same Java process has already executed those projection shapes is consistent with a
compiled-code warmup transition; a future JVM-focused write-up should retain this result while
distinguishing that causal interpretation from the directly observed facts above.

## Admission consequence

The BFCL-live window remained unscored. V18 returns to Phase 1 and must repeat all 75 exposed cases
under revision `4ebbc471553d5956970091d72f08fd79ad835a4a`. That report and hash must be committed
before a new live-screen attempt. No pre-fix score or threshold is carried forward.
