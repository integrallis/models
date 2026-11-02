# Extracting the Q4_K batched tile: bit-exact, and free

`compute_q4_k_batched_row_range_avx2` carried the lane-layout decision for the quantised prefill
GEMM inline, inside a doubly-nested loop in a 466-line function. This moves those sixteen lines into
`qk_batched_tile_lanes` with no arithmetic change -- same order, same intrinsics, same operands --
so that the layout can be changed in one place instead of being edited in situ.

**Measured here**, on the local box. Nothing in this directory is estimated or extrapolated.

## Host

- host: Intel(R) Core(TM) i7-9750H CPU @ 2.60GHz / 6 physical, 12 logical, MacBookPro16,1
- vector: AVX2 FMA F16C, no AVX-512 (machdep.cpu.leaf7_features)
- date: 2026-11-02T21:25:04Z
- harness: ab.c -> jmodels_quantized_f32_batched_matmul_with_context, format 2 (Q4K)
- best_ms = min over 20 reps after 3 warmups; hash = FNV-1a over the whole output buffer
- passes alternate which arm runs first, to keep run order from favouring either

Note that this is **not** the `i9-14900HX` recorded as the dev box in the repository-tree
`CLAUDE.md`; this machine is a 6-core Coffee Lake. AVX2-only holds either way, which is what the
kernel selection depends on, but the core count does.

## Bit-exactness

The harness hashes the entire output buffer with FNV-1a and compares arms. One hash per shape,
shared by both arms, is the assertion that the extraction changed no arithmetic:

```
output hash, rows=7168 cols=2304 batch=32 threads=1: IDENTICAL across both arms (50c4c25fb419ff1f)
output hash, rows=2304 cols=7168 batch=16 threads=1: IDENTICAL across both arms (4a7e70205a9bc2ab)
output hash, rows=7168 cols=2304 batch=32 threads=8: IDENTICAL across both arms (50c4c25fb419ff1f)
```

The crate's own guards for this function also pass: `cargo test --release`, 51 passed, 0 failed,
1 ignored -- including `q4_k_ordered_avx2_matches_scalar_block_fold_bit_for_bit` and
`q4_k_avx2_preserves_single_row_rounding_across_batch_sizes`.

## Throughput

Ten passes per shape, alternating which arm runs first so run order cannot favour either. `best_ms`
is the minimum of 20 reps after 3 warmups.

| shape | extracted best_ms (median) | parent best_ms (median) | ratio | passes extracted faster |
|---|---:|---:|---:|---:|
| rows=7168 cols=2304 batch=32 threads=1 | 18.825 (n=10, 18.527-19.197) | 19.399 (n=10, 19.135-19.530) | 0.9704 | 10/10 |
| rows=2304 cols=7168 batch=16 threads=1 | 9.963 (n=10, 9.703-10.522) | 10.169 (n=10, 10.009-10.408) | 0.9796 | 7/10 |
| rows=7168 cols=2304 batch=32 threads=8 | 3.879 (n=10, 3.568-4.316) | 3.825 (n=10, 3.704-4.232) | 1.0143 | 5/10 |

**Read this as "no regression", not as an optimisation.** The extraction cannot make the arithmetic
cheaper: it issues the same instructions in the same order. The 3% single-thread difference on the
first shape is consistent (10 of 10 passes) but it is a code-placement effect -- the same
instructions landing at different addresses -- and placement is not something this change controls
or can be relied on to keep. It is not banked as a gain anywhere, and the multi-threaded shape,
where the kernel is bandwidth-bound rather than issue-bound, shows nothing (ratio 1.0143, 5 of 10).

The honest summary is the one the change was made for: **factoring the tile out costs nothing.**

## Formatting

`cargo fmt --check` reports 3 diffs on the parent commit, all inside the Q4_K kernel and all
introduced by `dcd8b172`. The extraction adds none -- the count is 3 before and 3 after -- and the
second commit on this branch clears them.

## Reproducing

`./run.sh` rebuilds both arms from git and re-runs every shape; `./summarize.py` regenerates the
tables above from `ab-raw.txt`. `ab.c` is the harness: it drives
`jmodels_quantized_f32_batched_matmul_with_context` with format 2, which is the route the Java
prefill takes, rather than a reimplementation of it.

## What this does not do

It does not make prefill faster. The interleaved `block_q4_Kx8` layout that this factoring was meant
to enable was built, validated byte-for-byte against ggml's own `ggml_gemv_q4_K_8x8_q8_K`, and
measured: it does not pay on this caller, because the batch loop already hoists the nibble decode
out, so both layouts issue the same 2048 `maddubs` per block. That null and its evidence are in
`benchmark-results/2026-11-02-mellum-q4k-kernel-verification`. The layout code is kept on the
branch `spike/q4k-interleave-layout` rather than merged, because nothing calls it.
