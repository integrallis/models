# Is the CUDA kernel inert? No. Measured 2026-09-22.

Retracting a bad diagnostic first: I previously reported "GPU utilisation 0%" as a finding.
`nvidia-smi` samples instantaneously and says nothing about whether our kernel was asked to do
work. It was not evidence of anything and should not have been reported.

`CudaRoutingCounters` answers the question properly. Run: 5 JevBench items, Qwen3.5-4B Q4_K_M,
RunPod A40 (cc 86, ptx sm_80), `-Ddecisions.kernel=cuda -Ddecisions.requireKernel=cuda`,
prefillBatchSize=512, maxContextLength=4096, letter-logit mode.

```
cuda inert            false
cuda accelerated ops  890
cuda by tensor type   {Q6_K/PREFILL_PROJECTION=105, Q6_K/DECODE_PROJECTION=115,
                       Q4_K/PREFILL_PROJECTION=335, Q4_K/DECODE_PROJECTION=335}
cuda refusals         {}
cuda launches         54306
cuda h2d              730 copies, 274.9 MiB
cuda d2h              890 copies, 674.7 MiB
cuda weight uploads   89 tensors, 1290.8 MiB
wall 103.7 s for 5 tasks
```

**Zero refusals.** Every projection, prefill and decode, Q4_K and Q6_K, ran on the device. The
kernel is not inert and the earlier "GPU is 3.2x slower than CPU at 4B" measurement stands as a
measurement of a real device run.

## Why it is slower — two structural causes, both ours

1. **54,306 launches for 890 projections** (61 per projection). `CudaGgufBatchedMatrixKernel.project`
   loops `for (int batch = 0; batch < batchSize; batch++)` issuing one launch per batch row,
   because `CudaDriver.launch` hardcodes `gridDim.y = 1` and `gridDim.z = 1` and the Rust entry
   point takes `batch` as a scalar parameter rather than reading `blockIdx.y`. The kernel is
   decode-shaped (grid = rows, block = 32 threads, one warp per output row) and prefill reuses it
   per token. Each of those launches re-reads the whole weight matrix from HBM; there is no reuse
   of a dequantised weight tile across the batch, which is exactly the reuse the CPU batched
   Vector API path gets for free.

2. **1,620 host round trips with a synchronize at every projection.** Only projections are on
   device. Every RMSNorm, RoPE, attention, SwiGLU and residual is back on the CPU, so each of the
   890 projections pays a device->host copy and a `cuCtxSynchronize`. Aggregate transfer is only
   ~950 MiB over 103 s (~9 MiB/s), so PCIe bandwidth is not the limit — the round-trip count is.

Weight uploads are correct: 89 tensors, 1290.8 MiB, uploaded once. The resident-weight cache works.

## Fix, if this is pursued

Give `launch` a grid-Y dimension and have `models_q4k_decode_projection` / `models_q6k_decode_projection`
read the batch index from `blockIdx.y`, so one launch covers the whole prefill batch and a
dequantised weight tile is reused across batch rows. That collapses 54,306 launches to ~890 and
turns a per-token weight read into a per-tile one. It does not address cause 2.

Not attempted yet. Recorded as "not done", not as "cannot be done".

## Update: the grid-Y fix landed, and the verdict is negative

`CudaDriver.launch` gained a grid-Y dimension and the projection kernels now read the batch row
from `blockIdx.y`. Launches for a five-item run went **54,306 -> 890**, exactly one per
projection. ABI bumped 1 -> 2 so a stale PTX module fails closed. Verified bit-identical to the
old kernel on the two items that were warm in both runs (all ten probabilities, last digit).

Warm head-to-head, same A40 host, same five items, `-Ddecisions.warmupItems=3`:

| arm | wall, 5 items |
| --- | ---: |
| CUDA, grid-Y kernel | 98.7 s |
| CPU, rust-ffm Vector API | 21.0 s |

**The device path is 4.7x slower than 8-core CPU SIMD.** The earlier "3.2x slower" figure
flattered the GPU: both arms were cold, and the CPU path gains more from warmup.

Collapsing the launch count bought about 5%. That is the measurement that matters here, because
it falsifies the diagnosis it was built on: launch overhead was not the term. The term is the
1,620 synchronising host round trips per five items — one device-to-host copy and one
`cuCtxSynchronize` per projection, because only projections are on the device and every RMSNorm,
RoPE, attention, SwiGLU and residual runs back on the host.

**A design that offloads only the matmuls cannot win, at any kernel quality.** Beating the CPU
requires activations resident on the device across the whole layer stack, which means putting the
elementwise and attention stages on the device too. That is a substantially larger piece of work
than a kernel tweak, and it is the only version of this worth attempting. Not started.
