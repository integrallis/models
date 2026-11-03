# The batched Q4_K kernel never got dcd8b172's unpack. Giving it that is +12% on prefill.

**Measured here.** Nothing in this directory is projected.

## What was wrong

`dcd8b172` replaced sixteen branchy six-bit field extractions per block with a single vectorised
unpack — **in the single-row kernel only**. `compute_q4_k_batched_row_range_avx2`, which is the
prefill route for every batch size above one, still called `qk_scale` and `qk_min` eight times each
per block:

```rust
for group in 0..8 {
    group_scales[group] = qk_scale(scales, group);   // branchy, from memory
    group_mins[group] = qk_min(scales, group);
}
```

Prefill has been the weak side of this port for weeks (0.53x of the comparator) and this sat in it
the whole time. It was missed because the row kernel's fix was measured, reported as "+30.1%
prefill", and the batched kernel was never read afterwards.

## Result: batched Q4_K matmul, mellum FFN shape (7168 x 2304, batch 32)

| | best_ms | GFLOP/s |
|---|---:|---:|
| baseline | 14.859 mean, 14.521-15.491 | 71.4 |
| unified | **13.210 mean, 12.958-13.478** | **80.0** |

**+12.0%**, faster in 8 of 8 alternating passes, with **non-overlapping ranges**. Raw output in
`raw-batched-ab.txt`; `ab.c` drives the real FFI entry point rather than a reimplementation.

## The two kernels now share their minimum correction

`qk_block_minimum_products` is the four-lane minimum correction, called by both the single-row and
the batched kernel, which then accumulate it identically.
`q4_k_avx2_preserves_single_row_rounding_across_batch_sizes` asserts the two are bit-for-bit equal
at batch sizes 1, 3, 8, 32 and 33; it now holds **by construction** rather than because two
implementations happened to agree. Before this change the batched kernel reduced the correction in
eight lanes (`mullo_epi32` over `setr_epi32`) and the row kernel in four
(`madd_epi16` over hadd'd sums), which is why vectorising only one of them broke that test.

## Numerics changed, and that matters for release

The output hash moves (`50c4c25fb419ff1f` -> `48f5a9b0edbf0ebd`). The minimum correction now
accumulates across blocks in four f32 lanes and is reduced once per row, where it was a horizontal
reduction and a scalar FMA per block. Integer arithmetic is unchanged and exact; the float grouping
is not.

**mellum's qualification was measured on the old numerics.** This is a different library, so the
decode and contribution figures in `../2026-11-03-mellum-zen3-two-arm` do not transfer to it, and a
release carrying this should re-run the two-arm rather than inherit them.

## What is not claimed

- **No decode number.** The single-row kernel's own change is not resolved on this host: its
  harness ratios moved inside the laptop's noise, and that machine's ggml spread is ~10% against
  0.7% on a dedicated box. The +12% above is the batched path only.
- **Not bit-identical to ggml.** Earlier standalone variants read `0.000e+00` agreement on a
  dedicated Milan box; this build reads `1.471e-08` and `7.813e-09` here, the same as baseline. The
  discrepancy is unexplained, so the bit-exactness claim is withdrawn rather than repeated. Both
  builds agree with ggml to fp32 rounding, which is all the harness assertion requires.

## Gates

`cargo fmt --check` 0 diffs, `cargo clippy --release --all-targets` clean, `cargo test --release`
55 passed 0 failed, `./gradlew :backend-native:check` BUILD SUCCESSFUL.
