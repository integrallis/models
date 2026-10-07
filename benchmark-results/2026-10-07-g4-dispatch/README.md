# G4: the dispatch ceiling, measured

**Measured 2026-10-07 on an RTX 4090 (compute capability 8.9, driver 12080), Runpod, one pod.**
Models revision `1955b55ba16f9a2cd90655be3ac7323f3efa452a`. Artifacts: `rust-ptx-decode.json`,
`worker.log`, `runner.sh`.

Model: Granite 4.1 3B Q4_K_M, 2,099,501,664 bytes, sha256
`662b0626cd58f443baea23559b469df6576a81d349649c59413b36a9fb32eb29`, recomputed by the gate from the
file it actually opened. 20 prompts, 64 tokens each, 16 warmup tokens, `greedy-argmax-lowest-index-v1`,
`--arm both` so the denominator is the same process on the same host.

## The numbers

| | measured |
| --- | --- |
| **launches per decode step** | **241.0** |
| **transfers per decode step** | **402.0** |
| **activation bytes per decode step** | **12,660,712 (12.66 MB)** |
| kernel launches, total | 312,496 |
| host-to-device transfers | 208,416 |
| device-to-host transfers | 312,496 |
| decode steps | 1,260 |

Arms, same host, one after the other:

| arm | prefill tok/s | decode tok/s |
| --- | --- | --- |
| accelerated (Rust PTX) | 12.19 | **9.02** |
| control (Vector API) | 7.29 | **5.05** |

Gates:

- `routingObservability` (G2) **passed** — 2,301,496 accelerated operations, and **`refusals` is
  empty**: nothing silently fell back, so the counters describe work that really ran on the device.
- `startupHonesty` (G5) **passed** — 352 ms readiness against a 120,000 ms ceiling.
- `decodeSpeedup` (G4) **FAILED** — **1.788x against a 3.00x threshold**.
- `qualified: false`.

## What this settles

**The ceiling is dispatch, not arithmetic.** Every decode step issues 241 launches and 402 transfers
moving 12.66 MB. The projections are already bit-exact at model scale and nothing was refused, so
the kernels are not the problem: making the arithmetic free would still leave 402 host round-trips
per token. 1.788x is what that costs.

The cause is in `CudaGgufBatchedMatrixKernel.project`: it issues one launch and one `copyToHost`
per projection. `multiplyTriple` and `multiplyDual` share a single activation upload -- which is why
host-to-device (208,416) is lower than device-to-host (312,496) -- but each projection still returns
its result to a Java `float[]`, because the Java decoder owns the norm, rope, residual and SiLU
between projections.

Launch grouping has already been won and is not the remaining lever: the comment in `project`
records 54,306 launches for 890 projections before the batch index moved into `blockIdx.y`.

## Two corrections this run forced

**1. These exact figures already existed and were never committed.** A session note recorded "241
launches, 402 host transfers and 12.7 MB of activations per decode step" from a 2026-09-26 A40 run,
and a search of main and every worktree found them in no artifact. They are reproduced here to
within the rounding of the note, on different hardware. The lesson is not that the note was wrong --
it was right -- but that an uncommitted measurement is indistinguishable from a remembered one.

**2. A prediction derived from source was wrong, and is recorded as wrong.** Reading
`LlamaForwardPass`, the decode path issues 7 projections per layer (wq, wk, wv, output, ffnGate,
ffnUp, ffnDown) plus an output head, which for 40 blocks predicts ~281 launches and ~442 transfers.
Measured: 241 and 402. `decodeProjections` is 256,296 over 1,260 steps, about 203 projections per
step, so roughly 5.1 projections per layer reach the device rather than 7, with `refusals` empty.
**Why is not established here**, and it is not guessed at: the static read of the dispatch sites did
not predict the routed count, and closing that gap is its own question.

## What it does not say

Nothing about G1. The attention argmax question is untouched by this run, and the efficient path
cannot qualify on speed alone.

Nothing about a fix. That device-resident activations would collapse 402 transfers to about 2 is
arithmetic, not a measurement; it needs the elementwise kernels (rmsnorm, rope, residual-add,
silu-glu) and a seam that hands off a layer instead of a matmul, and then its own two-arm run.

One deviation from `benchmark-results/2026-09-18-gpu-large-model/REPRODUCE-RUST-PTX.md`, stated as
one: that recipe's step 4 uses Gemma 4 26B-A4B Q4_K_M (16.8 GB). This used Granite 4.1 3B (2.1 GB),
because it is the model the dispatch question was asked about and the download is paid for in GPU
time. The G4 threshold arithmetic is unchanged; the absolute tok/s are not comparable to the 26B run.
