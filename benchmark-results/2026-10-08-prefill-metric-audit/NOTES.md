# The prefill numbers were measuring three different things. Two of them were not prefill.

Audited 2026-10-08. Every figure below is printed by `audit.py` in this directory from the
committed `benchmark-results/certified-*/rag/*/` reports, or read directly out of named source
files at a pinned commit. Nothing here is hand-typed or remembered.

Reproduce: `python3 benchmark-results/2026-10-08-prefill-metric-audit/audit.py`
Table: `prefill-metric-audit.csv` (31 paired models, 16 vCPU AMD EPYC 7R13, AVX2, no AVX-512).

## The confusion

Two measurements of "our prefill versus theirs" were both live and disagreed by 4x:

| comparison | figure | where |
|---|---|---|
| ours vs Ollama, 31 models, certified fleet | **0.24x** (we are 4x slower) | derived from `certified-*` reports |
| ours vs llama.cpp, 1 model, Hetzner CCX33 | **0.91x** (we are at parity) | `2026-09-23-prefill-budget/NOTES.md` |

Neither number is a prefill-throughput measurement. They fail for different reasons.

## 1. The 0.24x Ollama ratio is retracted: the denominator is physically impossible

`OllamaGenerationParser.java:49-53` derives the comparator's prefill rate from Ollama's own
self-reported counters, `prompt_eval_count / prompt_eval_duration`. Checked against what the
machine can do — prefill costs about `2 * params * tokens` FLOP, parameter counts estimated
*conservatively* from GGUF artifact size, against a ceiling of 2.0 TFLOP/s that is already
rounded up from the box's ~1.84 TFLOP/s fp32 FMA peak:

- **14 of 31** Ollama prefill rates imply a throughput above the machine's peak FLOP rate.
- The worst implies **21.1 TFLOP/s, 10.6x the box's peak.**
- Our own rates, by the identical arithmetic, land at **0.28-0.60 TFLOP/s, 14-30% of peak** —
  0 of 31 impossible, and near-constant from 0.9B to 12B, which is what a compute-bound kernel
  looks like.

A rate that exceeds the hardware is not a slow measurement or a noisy one. It is not a
measurement. Dividing our honest rate by it yields 4.18x, and that number means nothing.

The mechanism, confirmed in llama.cpp's source rather than guessed (see §3): the counter
reports the *whole* prompt length while the duration covers only the tokens actually evaluated
after prefix-cache reuse. Our reports carry `cacheReadInputTokens` and `cacheWriteInputTokens`
separately and our rate uses `cacheWriteInputTokens` only — `InProcessGenerationClient.java:157`.
Ollama's reports carry **0 cache tokens across all 31 reports** because the field does not exist
on that side. So the two rates were never the same quantity.

**This invalidates the "4x prefill deficit" that was recorded as established fact.** It does not
invalidate the priority. See §4.

## 2. The 0.91x llama.cpp ratio is qualified: the prompt was too short to measure prefill

That comparison ran a **60-token prompt** (`2026-09-23-prefill-budget/NOTES.md`). Both arms
reported ~41-45 tok/s prefill with p95 TTFT of 1443-1639 ms. 60 tokens over ~1.45 s is ~41 tok/s
— so the figure both arms produced is approximately `prompt_length / fixed_per_request_overhead`,
and the ratio between them is a ratio of overheads, not of prefill throughput.

Corroborating: that same box measured **392 GFLOP/s** on our Q4_K batched matmul directly
(12288x2560, batch 64, 20 repeats, best-of). A 0.6B model prefilling at 44.9 tok/s implies
54 GFLOP/s — 7x below what the same box demonstrably sustains on the same kernel. The engine
figure cannot be the kernel's speed.

That note's own retraction of the 4.9x matmul claim stands and was correct. The replacement
0.91x is simply measured on a prompt that cannot exercise the thing being compared.

## 3. What llama.cpp actually does, read from source at `a58222229`

Not inferred. Read:

- `ggml/src/ggml-cpu/llamafile/sgemm.cpp` — tinyBLAS, the fast tiled GEMM, supports **only**
  F32, BF16, F16, Q8_0, Q4_0, Q5_0, IQ4_NL. **Q4_K/Q5_K/Q6_K appear nowhere in the file.**
  Our catalog is overwhelmingly Q4_K_M, so this path never runs for our models — or theirs.
- `ggml/src/ggml-cpu/repack.cpp:4599-4607` — instead, Q4_K goes through the *repack* extra
  buffer type: with AVX2 and output rows divisible by 8, weights are repacked at load into
  `block_q4_Kx8` (8 rows interleaved) and matmul dispatches to `ggml_gemm_q4_K_8x8_q8_K`.
- `ggml/CMakeLists.txt:152` — `option(GGML_CPU_REPACK ... ON)`. **On by default**, so a stock
  build on our fleet boxes uses it.
- `ggml/src/ggml-cpu/arch/x86/repack.cpp:2042` — that GEMM has a real **AVX2** implementation
  (`#if defined(__AVX2__) || defined(__AVX512F__)`), with an extra AVX-512BW/DQ tier above it.
- `ggml/src/ggml-cpu/server-context.cpp:503` — llama.cpp's own server sets
  `timings.prompt_n = n_prompt_tokens_processed`, i.e. it reports *processed* tokens and
  correctly excludes cache hits. llama.cpp's counter is honest; Ollama's is the one that is not.

### The structural difference between their Q4_K kernel and ours

Both are integer paths: Q8_K activations, `maddubs`/`madd` accumulation. Ours is **not** missing
SIMD and **not** missing weight-block reuse — two things previously claimed in this repo and
both wrong. `compute_q4_k_batched_row_range_avx2` (`backend-native/.../lib.rs:4198`) already
decodes each weight block once and reuses it across the batch.

The difference is the tile shape:

| | ours | llama.cpp |
|---|---|---|
| weight layout | row-major Q4_K | `block_q4_Kx8`, 8 rows interleaved |
| activation layout | row-major Q8_K | `block_q8_Kx4`, 4 rows interleaved |
| computed per inner pass | 1 output row x 1 batch row | 8 output rows x 4 batch rows |
| horizontal reduction | one per (output row, batch row) | **none** — the 8 lanes *are* 8 output rows |

Ours pays, per (row, block, batch): 8 `set1_epi16` broadcasts of a batch-invariant group scale,
a `mullo_epi32` (10-cycle latency on Zen3) on batch-invariant minimums, and a
`horizontal_sum_f32_avx2` per output element. Roughly 8 of ~39 vector ops per inner pass do the
multiply-accumulate; the rest is scale plumbing and reduction. Interleaving the weights makes
the lane dimension the output dimension, which deletes the reduction and amortizes the nibble
unpack across 4 batch rows and the activation load across 8 output rows.

This is the tried solution, shipping in llama.cpp and used by Ollama, with an AVX2 kernel we can
read and a `_generic` scalar reference (`repack.cpp:1905`) to cross-validate a port against.

## 4. The priority was right. The number attached to it was wrong.

TTFT remains a standing priority and the user was right to insist on it. What changes is the
magnitude and the cause, not the direction.

The one metric here that is definition-independent — wall clock to first token, taken by **our**
harness clock on both arms, same host, same prompts, identical input-token counts:

| | median | worst | best |
|---|---|---|---|
| TTFT p50, ours / Ollama | **1.65x** | 20.36x | 0.57x |
| TTFT p95, ours / Ollama | **1.66x** | 10.40x | 0.63x |

Absolute: our TTFT p50 median **1521 ms** against Ollama's **720 ms**. We are faster on 4 of 31
models (all Gemma; Ollama is poor there).

So: we are slower to first token, by about 1.65x and not 4x, and the tail is far worse than the
median — one model at 20x. A user feels the 1521 ms and the tail, not the ratio.

## 5. What this workload cannot test

The certified RAG workload prefills a median of **176 tokens per attempt**, with 63% of those
served from our prefix cache. That is far too short for prefill throughput to dominate TTFT, so
these reports **cannot** rank prefill kernels, in either direction. They are evidence about
short-prompt TTFT and nothing more. A long-prompt arm does not exist yet; its absence is "no
data", not "no gap".

## 6. What is now established, and what is not

Measured here, with artifacts:
- our Q4_K batched matmul sustains 392 GFLOP/s on 8 vCPU EPYC-Milan (2026-09-23, 20 repeats)
- our certified prefill rates imply 0.28-0.60 TFLOP/s, 14-30% of fleet-box peak, 31 models
- our TTFT is 1.65x Ollama's at p50 on a 176-token prompt, 20.4x worst case
- Ollama's reported prefill rate exceeds machine peak on 14 of 31 models

Read in source, not measured (llama.cpp `a58222229`):
- Q4_K prefill goes through repacked `q4_K_8x8_q8_K` with an AVX2 kernel, enabled by default
- tinyBLAS does not implement any K-quant
- llama.cpp's server prompt counter excludes cache hits; Ollama's does not

**Not established, and must not be asserted until measured:**
- any ratio between our Q4_K kernel and llama.cpp's on the same shapes and box — the direct
  kernel A/B has never been run
- whether the 8x4 tile wins on our shapes, and by how much
- long-prompt (>=1024 token) TTFT against any comparator
- Ollama's actual prefill throughput, by any honest measurement

## 7. Decided next, in order

1. ~~**Kernel and engine A/B against llama.cpp's own tools**~~ — **DONE 2026-10-08**, see
   `experiments/prefill-repack-tile/RESULTS.md`. `GGML_CPU_REPACK` ON vs OFF, BLAS and Metal
   compiled out of both arms, interleaved: **1.53x** (deepseek-coder-1.3B) and **1.87x**
   (HuatuoGPT-o1-7B) on AVX2, **1.73x** at p1024 and **1.49x** at p4096 on an AVX-512 Threadripper
   7960X. Every arm clears the registered 1.30x port threshold and its own kill criterion.
   **Decision: port the interleaved tile.** The run also surfaced a second, unmeasured gap: our
   kernels gate on `avx2 && fma && f16c` and have no AVX-512 path, while theirs has an
   AVX512BW/DQ tier. Original plan text follows.

   Kernel and engine A/B against llama.cpp's own tools, not a new microbenchmark:
   `test-backend-ops perf -o MUL_MAT` for Q4_K mul_mat GFLOP/s (comparable to our committed
   392 GFLOP/s datum) and `llama-bench -p 1024/4096 -n 0` for long-prompt prefill throughput.
   Fast iteration locally on an i7-9750H (AVX2, no AVX-512); the authoritative run goes on a
   VPS, and **must include an AVX-512 box**, because `arch/x86/repack.cpp` gives their Q4_K GEMM
   an AVX512BW/DQ tier for which we have no equivalent at all. A result measured only on AVX2
   would understate their ceiling and ours is not measured there either.
2. **Fix the harness so this cannot recur**: stop ranking on vendor self-reported prefill
   counters, and make a reported rate that exceeds the host's peak FLOP rate fail the report
   instead of appearing in a comparison table.
3. **Add a long-prompt TTFT arm** (>=1024 tokens, cache cold) so prefill is actually exercised;
   the existing 176-token workload cannot see it.
4. Port the 8x4 interleaved tile — (1) says it wins — measuring the port against the `_generic`
   reference in `repack.cpp:1905` for agreement before any speed claim.

## 8. What the port would be worth to a user, as a projection

`audit.py` prints this and labels it a projection, because that is what it is. Our implied prefill
time is ~95% of our measured TTFT even at this workload's 176-token prompts, so a prefill-kernel
win passes through to TTFT nearly one-for-one. Taking the **lowest** speedup any A/B arm measured
(1.49x):

| | measured today | projected after the port |
|---|---|---|
| our TTFT p50, median | 1521 ms | ~1046 ms |
| TTFT ratio vs Ollama, median | 1.65x | ~1.14x |
| models where we beat Ollama on TTFT | 4/31 | 8/31 |

That is the user-visible case for doing this work, and it is the only reason to do it. It is also
not a result: only a measured post-port TTFT may be reported as one.
