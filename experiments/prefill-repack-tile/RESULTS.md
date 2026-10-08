# Results: the 8x8 repacked tile is worth 1.5-1.9x on Q4_K prefill. Port it.

Collected 2026-10-08, after the protocol in README.md was frozen. Every number below is printed by
`summarize.py` from the raw artifacts in `results/` — 10 `llama-bench` JSON files from the local
host and the verbatim pod log from the AVX-512 host. Nothing is hand-typed.

Reproduce: `python3 summarize.py`


### Intel i7-9750H, 6 cores / 12 threads, 6 threads used  (AVX2, no AVX-512 — thermally limited laptop, iteration only)

| model | prompt | repack ON (tok/s) | repack OFF (tok/s) | speedup | ON spread | OFF spread | verdict |
|---|---|---|---|---|---|---|---|
| HuatuoGPT-o1-7B-Q4_K_M | 1024 | 26.29 | 14.07 | **1.87x** | 1.7% | 1.7% | effect > spread |
| deepseek-coder-1.3b-instruct.Q4_K_M | 1024 | 110.31 | 72.00 | **1.53x** | 12.2% | 1.2% | effect > spread |

### AMD Ryzen Threadripper 7960X 24-Cores  (AVX-512 (avx512_bf16 avx512_bitalg avx512_vbmi2 avx512_vnni avx512_vpopcntdq avx512bw avx512cd avx512dq avx512f avx512ifma avx512vbmi avx512vl))

| model | prompt | repack ON (tok/s) | repack OFF (tok/s) | speedup | ON spread | OFF spread | verdict |
|---|---|---|---|---|---|---|---|
| Llama-3.2-1B-Instruct-Q4_K_M | 1024 | 623.60 | 361.31 | **1.73x** | 0.2% | 0.4% | effect > spread |
| Llama-3.2-1B-Instruct-Q4_K_M | 4096 | 459.40 | 307.37 | **1.49x** | 2.2% | 0.4% | effect > spread |

## Verdict

The registered rule was: `S >= 1.30` on AVX2 → port the tile. Measured speedups are **1.53x** and
**1.87x** on AVX2 and **1.73x / 1.49x** on AVX-512, every arm clearing the threshold and every arm
clearing its own kill criterion (effect larger than within-arm spread). **Decision: port the
interleaved tile into the Rust shim.**

The effect is larger on the bigger model (1.87x on a 7B against 1.53x on a 1.3B), which is the
direction the catalogue is growing, and it narrows at p4096 (1.49x) where the matmul is large
enough that cache behaviour starts to bound both arms.

## What was actually varied

One cmake flag, `GGML_CPU_REPACK`, at llama.cpp `a5822222909b785f23ddc74ce3c8f85bd0e38562`, with
BLAS and Metal compiled out of both arms so neither could silently take a different path. ON
interleaves 8 weight rows into `block_q4_Kx8` at load and runs `ggml_gemm_q4_K_8x8_q8_K`
(8 output rows x 4 batch rows per pass). OFF falls back to a per-output-row `vec_dot` loop over
row-major Q4_K — structurally what `compute_q4_k_batched_row_range_avx2` does today.

A trap worth recording: the first local build pair had Metal enabled by default and reported
`backend = MTL` at 6.44 tok/s, running on the laptop's integrated GPU. A CPU kernel flag measured
through a GPU backend would have produced a clean-looking null. Both arms were rebuilt with
`-DGGML_METAL=OFF -DGGML_BLAS=OFF` and verified to report `backend = CPU` before any data here was
collected.

## What this does and does not establish

Establishes: the tile shape is worth 1.5-1.9x inside llama.cpp's own code, on two different
microarchitectures, at two prompt lengths, with tight spreads.

Does **not** establish: our kernel's ratio against either arm. `llama-bench` measures llama.cpp
against llama.cpp. For reference and not as a comparison, our Q4_K batched matmul measures 155.1
GFLOP/s on the local i7-9750H and 392 GFLOP/s on 8 vCPU EPYC-Milan (`cargo test --release --
--ignored prefill_budget`, best-of-20, 12288x2560 batch 64) — a kernel figure in different units
from an engine tok/s, so no ratio is claimed from it.

Also does not establish anything about short-prompt TTFT, which is where our measured 1.65x
user-visible gap lives. These runs use 1024 and 4096 token prompts; the certified RAG workload
prefills 176 tokens with 63% of them served from our prefix cache.

## An additional gap this surfaced

Our Rust kernels gate on `avx2 && fma && f16c` and have no AVX-512 path. llama.cpp's Q4_K GEMM has
an `AVX512BW && AVX512DQ` tier above its AVX2 one (`arch/x86/repack.cpp:2077`). On the Threadripper
7960X — full AVX-512 including `avx512_vnni` — their kernel used that tier while ours would have
run the AVX2 path. The size of that second gap is unmeasured and is not claimed here.

## Hosts

- Intel i7-9750H, 6 cores / 12 threads, 6 threads used. AVX2, no AVX-512. Thermally limited
  laptop; the 12.2% ON spread on the 1.3B model is that, and is why this host only iterates.
- AMD Ryzen Threadripper 7960X (Zen 4), 16 of 48 vCPU on a shared RunPod `cpu5c`, AVX-512 with
  `avx512bw/dq/vnni/bf16`. Sharing the host is a noise source; the interleaved design and the
  0.2-2.2% observed spreads bound it. Pod terminated after the run; total spend across both pods
  was about $0.12.
