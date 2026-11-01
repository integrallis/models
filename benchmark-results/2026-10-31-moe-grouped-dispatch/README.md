# Grouped expert dispatch on the Llama path — 2026-10-31

The Llama path's routed feed-forward issued `3 * expert_used_count` batch-1 kernel dispatches per
routed layer per token — **24 at `used = 8`** — where ggml issues one `mul_mat_id` over the selected
rows. `flint` already groups them and is qualified; the Llama path did not.

**Result: mellum's decode ratio 0.3771 to 0.5052 (+34%). maple unchanged, because TQ1_0 is not
grouped-eligible in the kernel.** Neither reaches the 0.80 floor.

## Measured

Same host (RunPod `cpu3c`, 16 vCPU AMD EPYC 7713, AVX2, no AVX-512), same comparator
(`llama-server` at `3f03b1772`), `sameWorkload()` matched on all eight compared fields.

| model | experts | route | ours p50 | comparator p50 | `decodeThroughputRatio` |
|---|---|---|---:|---:|---:|
| **mellum** Q4_K | 64 / 8 | per-expert | 11.34 | 30.06 | 0.3771 |
| **mellum** Q4_K | 64 / 8 | **grouped** | **15.11** | 29.91 | **0.5052** |
| **maple** TQ1_0 | 256 / 8 | per-expert | 13.02 | 62.43 | 0.2085 |
| **maple** TQ1_0 | 256 / 8 | **grouped** | 12.14 | 58.21 | 0.2086 |

The comparator held at ~30 tok/s across mellum's two runs, so the 34% is attributable to our arm
rather than to host drift.

**Accuracy is unchanged on both models** — 7 of 9 answered, 7 of 7 correct, matching the comparator
case for case. The grouped route is numerically equivalent, which is the point: the activation
policy is extracted into one `activateExpert` shared by both routes rather than copied, so a model
cannot be qualified on arithmetic the other route does not reproduce.

## Why maple gained nothing

Probed directly rather than inferred:

```
TQ1_0   grouped=false  independent=false
Q4_K    grouped=true   independent=true
Q4_0    grouped=true   independent=true
Q8_0    grouped=true   independent=true
```

`isGroupedEligible` declines TQ1_0, so maple falls through to the per-expert loop and the 0.2086 is
the old number re-measured. Adding TQ1_0 to the grouped kernel is Rust work and would unblock maple
and any future ternary MoE.

## What this does not establish

- **Neither model qualifies.** mellum needs **1.58×** more decode throughput, maple **3.8×**.
- **No end-to-end claim.** mellum's `endToEndLatencyRatio` read 7.63 then 0.51, but the comparator's
  own e2e p95 swung 10 630 ms to 48 135 ms between those runs. That is host variance on a shared
  64-core machine, not a result, and it is not reported as a gate change.
- **No profile.** This removed dispatch overhead that was visible from the code; it does not say
  what the remaining 1.58× is. #322 still wants a fresh profile.
- **One host class**, 9 cases per arm, one iteration, zero warmups.
