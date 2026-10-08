# The tile port: correct, and a null so far

Written 2026-10-08, after `RESULTS.md` established that the layout is worth 1.49x-1.87x inside
llama.cpp. This records what porting it into our Rust shim actually produced.

**Outcome: the port is correct and measures at parity with the shipping kernel. It does not
reproduce the speedup.** Published as a null, as it stands.

## What was built, and then removed

The module measured at parity, so it was never a fast path, and keeping an
`#[allow(dead_code)]` module in the shipping crate is the exact habit this repo is trying to break.
**It was deleted.** Everything needed to redo it is here: the layout, the arithmetic, the
validation strategy, and the measured progression showing where the time went. Re-creating it is a
morning's work; carrying it as dead weight is a permanent tax.

It was `backend-native/src/main/rust/model-kernels/src/q4k_tile.rs`, following llama.cpp at
`a5822222909b785f23ddc74ce3c8f85bd0e38562`:

- `repack_weights` — row-major Q4_K to 8-row-interleaved `block_q4_Kx8`, after `make_block_q4_Kx8`
  (`repack.cpp:2836`). **Size-preserving**: `8 * 144 == 1152`, so a caller owning the buffer can
  transform in place. Rejects untileable shapes rather than producing a wrong answer.
- `interleave_activations` — our row-major Q8_K to 4-batch-row-interleaved `block_q8_Kx4`.
- `gemm_scalar` — a direct port of `ggml_gemm_q4_K_8x8_q8_K_generic` (`repack.cpp:1905`).
- `gemm_avx2` — our own AVX2 formulation over those layouts.

The module is registered but **wired into no dispatch path**. It cannot affect any shipped result.

## Correctness

Validated three ways, against `dot_q4_k_q8_k_row_scalar`, which shares no code with the repack,
the interleave or either GEMM:

- the repacked six-bit scales and mins round-trip against our own independent `qk_scale`/`qk_min`
  decoders, per row and per sub-block
- `gemm_scalar` agrees with the row-major reference on 4x8x256, 4x16x512, 8x24x768 and 12x32x512
- `gemm_avx2` agrees with **both** the scalar tile and the row-major reference on the same shapes

Tolerance, not equality: the two sum in different orders. The timed benchmark additionally asserts
that the timed tile arm agrees with the shipping kernel, so a wrong arm cannot be reported as fast.

## The measurement, and where the time actually went

Single-threaded, Q4_K 12288x2560 batch 64 — the shape with a committed 392 GFLOP/s datum — on an
Intel i7-9750H (AVX2, no AVX-512). Best-of-N per arm. Ratios are against
`compute_q4_k_batched_row_range_avx2`, the shipping kernel.

| stage | GFLOP/s | ratio to shipping | what changed |
|---|---|---|---|
| scalar tile | 2.0 | 0.67x *vs scalar row-major* | layout alone, no intrinsics |
| AVX2 tile, first form | 40.2 | **0.51x** | `hadd` + `permutevar` per k step per lane |
| constants pre-permuted into hadd order | 51.4 | **0.64x** | data permutes removed from the inner loop |
| eight accumulators instead of sixteen | 48.1 | 0.62x | **no effect** — register pressure was not the cost |
| scale unpack hoisted out of the batch loop | 80.4 | **1.03x** | it had been running 16x over, once per batch group |
| scale vectors cached per block | 83.2 | 1.02x | no further effect |
| nibble unpack cached per row group | 80.4 | 1.12x | within host noise |

Four repeated invocations of the final form gave ratios of 0.97x, 1.05x, 0.94x and 1.06x, with the
*shipping* arm itself ranging 47.3-56.2 ms. **Median ~1.01x, and the host's noise is larger than
the effect.** The honest reading is parity.

Two findings worth keeping from that table:

1. **The scalar layout change alone is a regression (0.67x).** The interleaving buys nothing
   without vectorisation; it only adds index arithmetic. There is no partial credit on this work.
2. **The dominant cost was redundant work per batch group, not instruction selection.** Unpacking
   the six-bit scales once per batch group rather than once per weight block cost 40% of the
   kernel. Register pressure, which looked like the obvious culprit, measured as nothing.

Repack cost: 7.6 ms for a 12288x2560 tensor, 2.2 GiB/s — a one-off at load, which would be roughly
two seconds for a 7B Q4_K_M.

## Why this is a null and not a failure of the idea

`RESULTS.md` measured the layout at 1.49x-1.87x by toggling one cmake flag in llama.cpp's own code
on two microarchitectures with tight spreads. That result stands. What has not been reproduced is
that speedup *in our kernel*, and the difference is our inner-loop sequence: llama.cpp's AVX2 Q4_K
GEMM is some 700 lines of intrinsics using `blend`/`permute` patterns we have not matched, where
ours reduces eight elements with `maddubs` -> `madd` -> `hadd`.

So the remaining gap is specifically in `gemm_avx2`, and **not** in the repack or the interleave,
both of which are now validated and reusable by any future kernel.

## What must not be claimed

- that the tile does not work — it works in llama.cpp, measured
- that our kernel is as fast as llama.cpp's — unmeasured; `llama-bench` compares llama.cpp to
  llama.cpp, and no direct kernel-to-kernel ratio against ours exists
- any post-port TTFT improvement — nothing is wired in, so there is nothing to feel

## Next, in order

1. Re-run this A/B on a quiet box. The laptop's shipping arm varied 19% between invocations, which
   is wider than anything being measured; the AVX-512 pod showed 0.2-2.2% spreads.
2. Port llama.cpp's AVX2 inner sequence faithfully rather than our own formulation, keeping the
   validated repack and interleave and the three-way correctness harness.
3. Only then consider the AVX-512 tier, which we have no equivalent of at all, and the Java-side
   plumbing — which carries a memory cost: repacked weights need their own buffer unless the
   mapping is made private, roughly the size of the Q4_K tensors.
