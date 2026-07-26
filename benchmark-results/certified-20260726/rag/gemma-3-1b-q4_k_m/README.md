# Gemma 3 1B Instruct Q4_K_M general RAG qualification

This directory contains controlled, same-host qualification evidence for the
exact Gemma 3 1B Instruct Q4_K_M artifact:

- SHA-256:
  `12bf0fff8815d5f73a3c9b586bd8fee8e7b248c935de70dec367679873d0f29d`
- size: 806,058,496 bytes
- host: 8-vCPU AMD EPYC-Milan, 32 GiB RAM, Linux amd64
- JVM: GraalVM Community Java 25.0.3
- Models commit: `840ade15b3ac82852194f5fa4d63ad9c0012ef60`
- ModelJars profile commit: `c2f567c5de81d353cb3ff96eefaf5433a5fcb39d`
- Vectors commit: `c298a0b73970468794c1ba403022e2adc517e57e`
- native controls: Ollama 0.32.0 and llama.cpp b10012 (`c71854292`)

Every report uses the same nine general-knowledge cases, corpus SHA-256
`4b27eba8f166c84ef19c53de825445a6d0097f9bd8efa20b2d7013f34621f83c`,
top-1 retrieval, the `gemma` prompt template, a 2,048-token context, a
64-token output limit, one complete warmup, and three measured iterations.
Each backend ran sequentially in an isolated process on the same host.

## Results

| Backend | Tier | p95 TTFT (ms) | p50 decode (tok/s) | p95 E2E (ms) | Correct |
|---|---|---:|---:|---:|---:|
| Models Rust/FFM, Java decode | PRODUCTION_READY | 976.7 | 14.63 | 3,134.9 | 100% |
| Models Rust/FFM, ModelJar profile | PRODUCTION_READY; qualified | 954.3 | 41.06 | 1,619.7 | 100% |
| Ollama 0.32.0 | USABLE | 1,130.8 | 29.90 | 2,101.0 | 100% |
| llama.cpp b10012 | PRODUCTION_READY | 940.3 | 48.85 | 1,524.0 | 100% |

The marker-driven Models run reaches 137.3% of Ollama and 84.0% of llama.cpp
decode throughput. Its p95 end-to-end latency is 0.771x Ollama and 1.063x
llama.cpp, within the production policy limits of 1.5x and 2.0x. These ratios
are measurements for this exact host and workload, not cross-hardware claims.

The retained ModelJar profile selects eight native kernel workers and native
quantized decode. The marker-only benchmark supplied no performance system
properties, and diagnostics prove the exact profile was `ENABLED`. Native
decode improves median throughput by 2.806x, p95 end-to-end latency by 48.3%,
and peak RSS by 32.4% over Java decode. All 27 raw generations, grounding
decisions, evaluations, and final answers are identical.

The retained `models-rust-ffm-rejected-batch64.json` control proves that the
generic 64-token prefill and batched-attention profile was slower for this
artifact. Those settings are intentionally absent from the Gemma profile.

All 27 grounded answers were correct. Nine answers retained model text, fifteen
used validated extractive fallback, and three correctly abstained.
