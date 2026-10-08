# Is the 8x8 repacked-interleaved tile worth porting for Q4_K prefill?

Registered 2026-10-08, before any data in this directory was collected.

## Background, and what this replaces

Two earlier diagnoses of our prefill speed were wrong, and a third number was invalid:

- "the kernel lacks SIMD" — false, AVX2 K-quant kernels exist
- "the AVX2 kernel re-decodes the weight block per batch row" — false, it already blocks
- "we are 4x behind Ollama on prefill" — invalid; the ratio's denominator is Ollama's
  self-reported counter, which implies above-peak FLOP/s on 14 of 31 certified models
  (`benchmark-results/2026-10-08-prefill-metric-audit/`)

What *is* established, by reading llama.cpp at `a58222229` rather than guessing: Q4_K prefill
there does not use tinyBLAS (which implements no K-quant). It uses the repack extra buffer type,
which interleaves 8 weight rows into `block_q4_Kx8` at load and runs `ggml_gemm_q4_K_8x8_q8_K` —
8 output rows x 4 batch rows per pass, with the lane dimension being the output dimension, so
there is no horizontal reduction. Ours computes 1x1 per pass and horizontally reduces per output
element. `GGML_CPU_REPACK` defaults ON.

## Hypothesis

The interleaved 8x4 tile is the material difference between our Q4_K prefill kernel and
llama.cpp's, and is worth porting into the Rust shim.

## Design

`GGML_CPU_REPACK=ON` vs `OFF`, same llama.cpp commit, same model, same box, BLAS compiled out of
both. OFF falls back to a per-output-row `vec_dot` loop over row-major Q4_K — structurally our
kernel. So the ON/OFF delta estimates the prize **in their code, before we write any Rust.**

`llama-bench -p N -n 0` is llama.cpp's own tool; no new benchmark code is introduced, per the
framework-first rule. Arms are interleaved (ON,OFF,ON,OFF,ON,OFF) because the local box is a
thermally limited laptop.

Prompt length 1024 and 4096, not the 176 tokens the certified RAG workload uses: at 176 tokens
prefill does not dominate TTFT, which is why that workload cannot rank prefill kernels at all.

Hosts:
- local i7-9750H, 6 cores, AVX2, **no AVX-512** — iteration only
- VPS, authoritative, and **must include an AVX-512 box**: `arch/x86/repack.cpp` gives their
  kernel an AVX512BW/DQ tier we have no equivalent for, so an AVX2-only result understates the
  ceiling we would be porting toward

## Decision rule

Decided on the **VPS** runs, not the laptop. Let `S` be the median ON/OFF speedup at p1024.

- `S >= 1.30` on AVX2 → port the tile. The ceiling is worth the work.
- `1.10 <= S < 1.30` → port only if the AVX-512 box shows `S >= 1.50`, since the catalogue's
  larger hosts are where it would pay.
- `S < 1.10` → **do not port.** The tile is not the gap, and the published null stands as the
  result. Look instead at where TTFT actually goes at 176 tokens, where fixed per-request cost,
  not kernel throughput, plausibly dominates.

## Kill criterion

If within-arm run-to-run spread on a host exceeds the ON/OFF delta on that host, that host's run
is inconclusive and is reported as such — not averaged into a verdict. Observed laptop spread was
already ±8% on a single configuration, so the laptop may well be killed by this rule; that is
expected and is why the VPS decides.

## What this experiment cannot answer

- our own kernel's speed relative to either arm. llama-bench measures llama.cpp against
  llama.cpp. A direct ratio against our Rust kernel needs our harness on the same box at the same
  prompt length, which is a separate step and is not claimed from this one.
- anything about TTFT at short prompts, which is the regime the certified catalogue actually
  measures and where our 1.65x user-visible gap lives.

## Status

Protocol frozen, then executed. **Result: 1.53x and 1.87x on AVX2, 1.73x and 1.49x on AVX-512 —
the rule's port threshold is met on every arm.** See `RESULTS.md`; raw artifacts in `results/`.

The port that decision authorised is written and correct but measures at **parity** with the
shipping kernel, so it is a null as it stands and is wired into no dispatch path. See
`PORT-NOTES.md`, which records where the time actually went.
