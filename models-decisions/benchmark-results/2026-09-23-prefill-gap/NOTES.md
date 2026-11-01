# Our prefill is 7.3x slower than llama.cpp, measured against it on one host

2026-09-23. Found while trying to qualify Qwen3.5-4B Q4_K_M so Harriet could carry a ModelJars
marker. The qualification failed, and the reason is not the model or the hardware.

## The measurement

RunPod `cpu5c`, 32 vCPU AMD Ryzen Threadripper 7960X, 64 GB. Weights
`Qwen3.5-4B-Q4_K_M.gguf`, sha256 `00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4`,
verified byte-identical to the copy used for every JevBench run. `models` clean at `v0.3.43`.
Harness `scripts/run-controlled-rag-qualification.sh`, workload `general`, template `chatml`,
3 iterations, 9 cases. Reports in this directory.

| engine | p95 TTFT | p50 decode | p95 end-to-end | tier |
| --- | ---: | ---: | ---: | --- |
| **models rust-ffm (ours)** | **7,139.0 ms** | **25.83 tok/s** | 10,696.0 ms | OFFLINE |
| Ollama | 954.1 ms | 24.15 tok/s | 5,215.9 ms | USABLE |
| llama.cpp | 981.9 ms | 24.28 tok/s | 4,961.6 ms | USABLE |

All three: 27/27 cases, 100% correct, abstention accuracy 1.0. **Quality is not in question.**

**Decode is ours to win: 25.83 tok/s is the fastest of the three.** Prefill is where we lose, by
7.3x against llama.cpp on the same host, same weights, same prompts.

## What this corrects

An earlier reading of this run quoted 954 ms as our TTFT. That was **Ollama's** line. The arms are
labelled `plain-java/<engine>/<model>`, and the engine field is the baseline, not the candidate.
Ours is the `rust-ffm` line.

Before that, an 8-vCPU run failed the same gate at 4,165.9 ms and was attributed to "a 4B model on
8 cores". That number was also llama.cpp's; ours was worse. The host was never the cause, and
moving from 8 to 32 cores only lifted the references into USABLE while we stayed OFFLINE.

## Why it matters beyond this gate

This is the same bottleneck three other measurements have pointed at, now with a reference number:

* JevBench: the hard tier is 111 of 231 items and **96% of all compute**, mean 11.0 s against
  0.37 s for standard. Long prompts, prefill-bound.
* `prefillBatchSize=512` helps only long prompts: 1.19x on hard, 0.95-0.99x on easy and standard.
* The CUDA path lost to CPU SIMD because only projections were on the device and every projection
  paid a host round trip -- again prefill shape, not arithmetic.

A 7.3x gap against a mature C++ implementation is the size of the opportunity, and it is now
measured rather than asserted.

## What it does not say

It does not localise the gap. Candidates not yet separated: the attention over the prompt window,
the K-quant matmul at prefill batch shapes, thread scaling across 32 cores, and per-token overhead
in the graph. The next step is a profile of our prefill against llama.cpp's on one host, not
another guess.

It also says nothing about Harriet's quality. Harriet's published JevBench numbers were measured
on our engine and stand; this gate is about latency, and the engine's decode is competitive.

## Consequence

`unsloth_qwen3_5_4b_gguf_q4_k_m` cannot be qualified on our engine today, so
`org.modeljars.composite:harriet` cannot be published: the catalog refuses a composition whose
members are not independently qualified, and it is right to.

`com.integrallis:models-decisions:0.3.43` is published and installable regardless.
