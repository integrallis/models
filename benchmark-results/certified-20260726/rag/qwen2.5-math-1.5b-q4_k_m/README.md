# Qwen2.5-Math 1.5B Q4_K_M math RAG qualification

This directory contains controlled, same-host qualification evidence for the
exact Qwen2.5-Math 1.5B Instruct Q4_K_M artifact:

- source: `bartowski/Qwen2.5-Math-1.5B-Instruct-GGUF`
- source revision: `951ed2aea09c43e331c612e74d83e4a23ca98e3b`
- ModelJar coordinate:
  `org.modeljars.huggingface:bartowski.qwen2.5-math-1.5b-instruct-gguf.q4_k_m:2.5.0-q4_k_m.1`
- SHA-256:
  `9614a50f03c897028920ca0dc4365da570bf587f9ee7768261216fe370b37e8e`
- size: 986,048,832 bytes
- host: dedicated 8-vCPU AMD EPYC-Milan, 32 GiB RAM, Linux amd64
- Models commit: `49b848f3e594d07322cbe2ea8418e9312609dd33`
- ModelJars profile commit: `dd75d510ceebf71606e69aa80d6d7624173755e9`
- Vectors commit: `c298a0b73970468794c1ba403022e2adc517e57e`
- Models JVM: GraalVM Community Java 25.0.3
- native controls: Ollama 0.32.0 and llama.cpp b10012 (`c71854292`)

Every report uses the same artifact bytes, nine math RAG cases, corpus SHA-256
`4da610e627e7481d54d1595ef1191dff454455d9bb04fc814b5ea12fd6180315`,
top-1 retrieval, the `chatml-direct` prompt template, a 2,048-token context,
a 64-token output limit, stop sequence `. `, one complete warmup, and three
measured iterations. Models, Ollama, and llama.cpp ran sequentially in
separate isolated processes on the same dedicated host with eight CPU threads.
GraalVM applies only to the Models rows; Ollama and llama.cpp used their native
runtimes.

## Results

| Backend | Tier | p95 TTFT (ms) | p50 decode (tok/s) | p95 E2E (ms) | Correct |
|---|---|---:|---:|---:|---:|
| Models Rust/FFM v10, explicit settings | PRODUCTION_READY | 781.2 | 27.02 | 2,093.7 | 100% |
| Models Rust/FFM v10, ModelJar profile | PRODUCTION_READY; qualified | 754.8 | 26.99 | 2,079.1 | 100% |
| Ollama 0.32.0 | PRODUCTION_READY | 606.3 | 26.67 | 3,219.7 | 100% |
| llama.cpp b10012 | USABLE | 1,084.4 | 44.92 | 2,415.2 | 100% |

The marker-driven Models run reaches 101.19% of Ollama and 60.08% of llama.cpp
decode throughput. Its p95 end-to-end latency is 0.646x Ollama and 0.861x
llama.cpp. These ratios compare only the sequential process-isolated runs on
the same hardware described above; they are not cross-hardware performance
claims.

All 27 grounded answers were correct. Nine answers retained model text and all
nine were correct, fifteen used validated extractive fallback, and three
correctly abstained. The reports pin grounding policy
`trusted-title-provenance-statement-anchors-safe-discourse-explicit-abstention-v13`.

The marker-only run supplied no decode, worker-count, batch-size, or attention
system properties. Diagnostics record the exact
`qwen2_5_math_1_5b_q4_k_m_epyc_milan_jdk25_rust_ffm` profile as `ENABLED`,
with no selector mismatch or missing launch argument. The profile selects
eight native workers, native quantized decode, 32-token batched prefill, and
disabled batched attention. All 27 prompt hashes, raw generations, grounding
decisions, evaluations, and raw evaluations match the explicit control
exactly.

Both Models reports record `rust-ffm-v10`, the AVX2 Q4_K batched kernel, and
grouped Q4_K/Q6_K projections. `CertifiedRagEvidenceTest` recomputes the
qualification decision from the retained marker and comparator reports.
