# Indian-Legal-Qwen2.5-3B Q4_K_M legal RAG qualification

This directory contains controlled, same-host qualification evidence for the
exact Indian-Legal-Qwen2.5-3B Q4_K_M artifact:

- source: `GSMS-B/Indian-Legal-Qwen2.5-3B-GGUF`
- source revision: `b1be624893865ae28ac424d85b410b808315be76`
- ModelJar coordinate:
  `org.modeljars.huggingface:gsms-b.indian-legal-qwen2.5-3b-gguf.q4_k_m:2.5.0-q4_k_m.1`
- SHA-256:
  `20e09a60606859d9a5401f4d261d02c1a1c57b75ee322a10b034cdbf2506fcb5`
- size: 1,929,902,560 bytes
- host: dedicated 8-vCPU AMD EPYC-Milan, 32 GiB RAM, Linux amd64
- Models commit: `8f98d8fd9d838b4386d8de47faaf9a4a3e71fa58`
- ModelJars profile commit: `789f292368728da1a356e926216753d3a11fac1b`
- Vectors commit: `c298a0b73970468794c1ba403022e2adc517e57e`
- Models JVM: GraalVM Community Java 25.0.3
- native controls: Ollama 0.32.0 and llama.cpp b10012 (`c71854292`)

Every report uses the same artifact bytes, nine legal cases, corpus SHA-256
`a3499a08589094e7cb36299785bf128d5dad4a781fee96651878c267d61d0c2b`,
top-1 retrieval, the `chatml` prompt template, a 2,048-token context, a
64-token output limit, one complete warmup, and three measured iterations.
Models, Ollama, and llama.cpp ran sequentially in isolated processes on the
same host with eight CPU threads. GraalVM qualifies only the Models row;
Ollama and llama.cpp used their native runtimes.

## Results

| Backend | Tier | p95 TTFT (ms) | p50 decode (tok/s) | p95 E2E (ms) | Correct |
|---|---|---:|---:|---:|---:|
| Models Rust/FFM v10, explicit settings | USABLE | 1,322.3 | 15.65 | 3,565.1 | 100% |
| Models Rust/FFM v10, ModelJar profile | USABLE; qualified | 1,251.2 | 15.58 | 3,487.1 | 100% |
| Ollama 0.32.0 | PRODUCTION_READY | 986.5 | 18.49 | 2,984.8 | 100% |
| llama.cpp b10012 | OFFLINE | 2,109.5 | 24.24 | 3,574.7 | 100% |

The marker-driven Models run reaches 84.26% of Ollama and 64.26% of llama.cpp
decode throughput. Its p95 end-to-end latency is 1.168x Ollama and 0.975x
llama.cpp, within the production policy limits of 1.5x and 2.0x. These ratios
qualify this exact artifact, host class, JVM, backend, and workload; they are
not cross-hardware performance guarantees.

All 27 grounded answers were correct. Twelve answers retained model text and
all twelve were correct, twelve used validated extractive fallback, and three
correctly abstained. The reports pin grounding policy
`trusted-title-provenance-statement-anchors-safe-discourse-explicit-abstention-v11`.

The marker-only run supplied no decode, worker-count, batch-size, or attention
system properties. Diagnostics record the exact
`indian_legal_qwen2_5_3b_q4_k_m_epyc_milan_jdk25_rust_ffm` profile as
`ENABLED`, with no selector mismatch or missing launch argument. The profile
selects eight native workers, native quantized decode, 32-token batched
prefill, and disabled batched attention. All 27 prompt hashes, raw generations,
grounding decisions, evaluations, and raw evaluations match the explicit
control exactly.

Both Models reports record `rust-ffm-v10`, the AVX2 Q4_K batch vector
accumulation kernel, and grouped Q4_K/Q6_K projections. The qualification
decision is recomputed from the marker report and immutable controls by
`CertifiedRagEvidenceTest`.
