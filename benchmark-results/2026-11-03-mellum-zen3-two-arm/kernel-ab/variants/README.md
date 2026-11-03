# The three measured Q4_K row-dot variants, as patches

Reproduction inputs for the kernel table in `../../README.md`. **None of these is shipped.** They
are kept because each was measured on reference hardware and re-deriving them costs more than
storing them.

Each patch applies to `backend-native/src/main/rust/model-kernels/src/lib.rs` at the `base` the
measurement used (models `3fef5ad0`, which is `main` at the time of the run):

    patch -p1 --directory backend-native/src/main/rust/model-kernels < mins.patch

| patch | change | 36 MiB | 9 MiB | output vs ggml |
|---|---|---:|---:|---|
| `mins.patch` | minimum correction accumulates in four f32 lanes, reduced once per row, instead of a per-block horizontal reduction plus scalar FMA | +1.29% | +1.63% | **0.000e+00** |
| `combo.patch` | `mins` plus a `vpshufb` scale broadcast out of the unpacked register and `#[inline]` on two hot helpers | **+1.54%** | **+2.33%** | **0.000e+00** |
| `chain.patch` | `combo` plus folding each pair before touching the accumulator | +1.37% | +2.11% | null against `combo` |

Percentages are ratio gain against the shipped baseline, n=8 per arm per size, arms alternated,
ggml in-process as the drift control, on an EPYC-Milan box at load 0.02. `run-ab.sh` rebuilds every
variant and re-runs the comparison.

## Why none of them ships

`combo` and `chain` break `q4_k_avx2_preserves_single_row_rounding_across_batch_sizes`. That test
asserts `compute_q4_k_batched_row_range_avx2` is bit-for-bit equal to `dot_q4_k_q8_k_row_avx2` at
every batch size, and the two kernels reduce the minimum correction in different lane counts -- the
batched one in eight (`mullo_epi32` over `setr_epi32`), the row one in four (`madd_epi16` over
hadd'd sums). Vectorising only the row kernel gives the two different float groupings. Shipping any
of these requires unifying that lane layout across both kernels first.

Their end-to-end effect is also unresolved: `combo` measured 0.8744 (sd 0.0105) against the
baseline's 0.8683 (sd 0.0030) over three paired runs -- overlapping ranges. The isolated gain is
clean at n=8; the end-to-end gain is not, and a single first pass reading 0.8816 would have
supported a "+1.7%" claim the other two passes do not.
