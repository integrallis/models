# Kernel head-to-head: the cause, and the fix

Box: i7-9750H (AVX2, no AVX-512), 6 cores / 12 threads, 12 MiB L3, macOS. llama.cpp `785322d17`,
built `-DGGML_METAL=OFF -DGGML_BLAS=OFF` with the printed backend asserted as CPU.
**Single-threaded throughout**, so thread scaling and bandwidth saturation cannot contribute.
ggml runs in the same process as the in-run control. Both kernels' outputs are compared before any
timing is believed.

## The cause is kernel-internal, and it is size-independent

Baseline, our kernel against ggml's at three working-set sizes:

| working set | ggml | ours | ours/ggml |
|---|---|---|---|
| 1.1 MiB (L2-resident) | 4.87 GB/s | 2.33 GB/s | 0.477 |
| 9 MiB (L3-ish) | 4.15 GB/s | 1.89 GB/s | 0.455 |
| 36 MiB (beyond L3) | 3.48 GB/s | 1.80 GB/s | 0.518 |

A flat ~2.0-2.2x across a 33x change in working set **rules out cache and memory effects**, and
single-threading rules out scheduling. It matches the 2.17x measured end-to-end on the pod, so the
whole decode gap lives inside the single-row kernel.

## What the difference was

Per 256-element block, with the *inner* dot already identical (4 weight loads, 8
`maddubs`, 8 `madd_epi16`):

| | ggml | ours (before) |
|---|---|---|
| six-bit scale/min unpack | **once** per block: 3 u32 words, 5 scalar ops, one `cvtepu8_epi16` | **16x** per block: branchy `qk_scale`/`qk_min` per group, from memory |
| scale broadcast | `shuffle_epi8` from a register | `set1_epi16` from a GPR, 8x |
| mins term | one `loadu` of `bsums` + `hadd_epi16` + `madd_epi16` + `fmadd` | **scalar 8-iteration loop**, 16 i16 loads, 8 muls |
| float conversion | one `cvtepi32_ps` + `fmadd` per block | same |

So the defect was per-block overhead wrapped around a correct inner loop -- the same class the
earlier tile port already priced at 40% (`repack-tile-is-the-prefill-win`: "the dominant cost was
redundant work per batch group, not instruction selection"). The GEMV never got that fix.

## The fix, and what it bought

One shared helper `qk_unpack_scales_and_mins` does the field surgery once per block and returns the
scales as eight i16 lanes plus `[scales | mins]` as sixteen; the mins term becomes
`hadd_epi16` + `madd_epi16`. Ported from `ggml_vec_dot_q4_K_q8_K` (MIT, attribution retained in the
source comment).

Before/after in one session on one box, ggml as the in-run control, 4096 rows:

| type | before | after | our speedup |
|---|---|---|---|
| **Q4_K** | 1.86 GB/s (0.454x ggml) | **3.66 GB/s (0.858x)** | **1.97x** |
| Q5_K | 2.84 GB/s (0.700x) | 3.15 GB/s (0.758x) | 1.11x |

Q5_K gains little because its mins term was already partly vectorised; Q4_K's was not.

**Numerically exact, not merely close.** The harness's agreement figures are *identical* before and
after -- mean relative difference 1.471e-08 at this size, unchanged to all printed digits. That is
expected and was predicted: the only reordering is of integer sums, integer addition is
associative, and no lane overflows (pair sums of 16 q8 values are at most 4064, products at most
~256k). The crate's 43 tests pass, including the bit-for-bit ordered-reduction ones.

## Reproduce

    KERNELS=<path>/model-kernels/target/release ./build-and-run.sh


## Re-measured against CURRENT main after rebasing -- these supersede the numbers above

The figures above were taken against a base that was **124 commits behind `origin/main`**, on a host
that was 11-15 GB into swap. After rebasing onto `33700300` (both commits applied with no
conflicts, 48 crate tests passing) the same harness, same box, now unloaded, gives:

| type | main | fixed | gain | ours/ggml |
|---|---:|---:|---:|---|
| **Q4_K** | 5.64 GB/s | **9.49 GB/s** | **1.68x** | 0.536 -> **0.905** |
| Q5_K | 8.17 GB/s | 8.45 GB/s | **1.03x -- null** | 0.822 -> 0.854 |

**Take 1.68x as the Q4_K figure, and treat Q5_K as a null**: main had already improved the Q5_K row
dot independently, so there is little left for this change to take there.

Note ggml itself measured **10.5 GB/s** here against **4.15 GB/s** in the earlier session on the
same machine -- a 2.5x swing from machine load alone. That is the whole reason ggml runs in-process
as the control and why only `ours/ggml` is comparable across sessions: 0.455 -> 0.991 earlier,
0.536 -> 0.905 now, the same finding at different absolute scales.
