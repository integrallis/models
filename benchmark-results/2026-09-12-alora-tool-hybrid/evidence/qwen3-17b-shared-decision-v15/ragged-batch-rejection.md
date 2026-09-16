# V15 ragged-batch execution rejection

## Purpose

The first sequential V15 case required roughly 220 seconds on the local Intel qualification host.
Commit `2cc4dfc5d15476fb3a180bf284d848c43cba9ded` therefore added a bounded four-session path over the
existing Java ragged-prefill implementation. This was an execution-only experiment; the frozen
partition, token IDs, threshold selection, and quality gates did not change.

## Fixed execution gate

Before the batched run, the preflight required the first sequential sentinel and its batched form
to produce bit-identical call and no-call logits. Every branch also had to prove identity of its
physical immutable KV-prefix storage.

## Observed result

On 2026-09-13, the sequential `irrelevance_126` sentinel had previously reported a rounded margin
of `0.5556`. The first four-case ragged batch produced:

| Case | Expected | Margin | Physical prefix | Batch wall time |
| --- | --- | ---: | --- | ---: |
| `irrelevance_126` | no call | `0.5042` | yes | 265,587 ms |
| `irrelevance_165` | no call | `9.2994` | yes | 265,587 ms |
| `irrelevance_169` | no call | `-8.3690` | yes | 265,587 ms |
| `irrelevance_172` | no call | `3.9466` | yes | 265,587 ms |

The sentinel was not bit-identical, so the run was stopped immediately after the first batch. No
threshold was calculated and no report was written. The likely cause is different floating-point
accumulation in the batched quantized matrix kernels, but that diagnosis is not treated as proof.

## Decision

The batched qualification path is rejected and was removed by commit `8ac3889`. V15 returns to the
exact sequential Java path. Ragged batching may be researched separately, but these four margins
cannot be used to change the frozen V15 policy or gates.
