# Live reference Q4_K dispatch

Identified 2026-11-08 after independent ordinary Q4_K and Q5_K dot fixtures passed.
The layer traces still differed in Q/K projections. The pinned reference's actual
load log says `CPU_REPACK`: x86 Q4_K matrices with rows divisible by eight use
`ggml_gemv_q4_K_8x8_q8_K` / `ggml_gemm_q4_K_8x8_q8_K`, not the ordinary
`ggml_vec_dot_q4_K_q8_K` fixture. Q5_K and Q6_K stay on ordinary x86 kernels.

The repacked GEMV accumulates one integer dot/minimum per 256-element block into
two independent float accumulators, then subtracts them. GEMM accumulates each
64-element subblock into two independent float accumulators. Four-token groups
use GEMM; remaining tokens use GEMV. This is a graph-shape contract, not a
prompt-specific choice. A correct implementation must preserve that contract
across internal worker partitions and output-row tails.

An external native spike replaces the ordinary Q4_K path temporarily, alongside
the independently validated Q5_K correction and 16x16 cache tiling. This spike
is not production code: production must preserve existing ordinary/ordered
contracts and select the reference contract explicitly for the image runtime.
An independent synthetic graph allocates actual `CPU_REPACK` weight buffers and
records that its tensor traits were installed. It checks columns
256/768/4096/11008 and batches 1/3/4/8/17. The first comparison exposed a spike
limitation: small worker row partitions fell back to the ordinary batched
kernel. With one complete eight-row partition, all twenty shapes are bit-exact.
Both attempts are retained; production tests must cover arbitrary partitions.

Next gates: six known answer regressions, A/B/A first-case timing, then full
frozen qualification on both models. No existing evidence is replaced and no
threshold changes. Existing owned machines, spend cap and hard deletion deadline
remain unchanged. The ordinary-kernel test suite is intentionally not claimed
for the external spike; a shipping implementation must pass it unchanged.
