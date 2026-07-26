# DeepSeek-R1-Distill-Qwen-1.5B Q4_K_M general RAG qualification

This directory contains controlled, same-host qualification evidence for the
exact DeepSeek-R1-Distill-Qwen-1.5B Q4_K_M artifact:

- source: `bartowski/DeepSeek-R1-Distill-Qwen-1.5B-GGUF`
- source revision: `9cc28b17e86fa2415fcb070f8ee5ec27c965aa61`
- ModelJar coordinate:
  `org.modeljars.huggingface:bartowski.deepseek-r1-distill-qwen-1.5b-gguf.q4_k_m:1.0.0-q4_k_m.1`
- SHA-256:
  `1741e5b2d062b07acf048bf0d2c514dadf2a48f94e2b4aa0cfe069af3838ee2f`
- size: 1,117,320,800 bytes
- host: dedicated 8-vCPU AMD EPYC-Milan, 32 GiB RAM, Linux amd64
- Models commit: `2627e41958c634f12a7f22ec7667aff8909ae6f7`
- ModelJars profile commit: `c5cdae86eac8242a6e3308cc60cfb39ced9279e8`
- Vectors commit: `c298a0b73970468794c1ba403022e2adc517e57e`
- Models JVM: GraalVM Community Java 25.0.3
- native controls: Ollama 0.32.0 and llama.cpp b10012 (`c71854292`)

Every report uses the same artifact bytes, nine general RAG cases, corpus
SHA-256 `4b27eba8f166c84ef19c53de825445a6d0097f9bd8efa20b2d7013f34621f83c`,
top-1 retrieval, the `chatml-no-think` prompt template, a 2,048-token context,
a 64-token output limit, one complete warmup, and three measured iterations.
Models, Ollama, and llama.cpp ran sequentially in isolated processes on the
same host with eight CPU threads. GraalVM qualifies only the Models rows;
Ollama and llama.cpp used their native runtimes.

## Results

| Backend | Tier | p95 TTFT (ms) | p50 decode (tok/s) | p95 E2E (ms) | Correct |
|---|---|---:|---:|---:|---:|
| Models Rust/FFM v10, explicit settings | USABLE | 1,110.5 | 28.77 | 2,895.6 | 100% |
| Models Rust/FFM v10, ModelJar profile | USABLE; qualified | 1,048.9 | 29.04 | 2,820.3 | 100% |
| Ollama 0.32.0 | PRODUCTION_READY | 675.4 | 29.35 | 2,549.8 | 100% |
| llama.cpp b10012 | USABLE | 1,304.7 | 45.21 | 2,215.2 | 100% |

The marker-driven Models run reaches 98.95% of Ollama and 64.23% of llama.cpp
decode throughput. Its p95 end-to-end latency is 1.106x Ollama and 1.273x
llama.cpp, within the production policy limits of 1.5x and 2.0x. These ratios
qualify this exact artifact, host class, JVM, backend, and workload; they are
not cross-hardware performance guarantees.

All 27 grounded answers were correct. Nine answers retained model text and all
nine were correct, fifteen used validated extractive fallback, and three
correctly abstained. The reports pin grounding policy
`trusted-title-provenance-statement-anchors-safe-discourse-explicit-abstention-v12`.

The marker-only run supplied no decode, worker-count, batch-size, or attention
system properties. Diagnostics record the exact
`deepseek_r1_distill_qwen_1_5b_q4_k_m_epyc_milan_jdk25_rust_ffm` profile as
`ENABLED`, with no selector mismatch or missing launch argument. The profile
selects eight native workers, native quantized decode, 32-token batched
prefill, and disabled batched attention. All 27 prompt hashes, raw generations,
grounding decisions, evaluations, and raw evaluations match the explicit
control exactly.

Both Models reports record `rust-ffm-v10`, the AVX2 Q4_K batch vector
accumulation kernel, and grouped Q4_K/Q6_K projections. The qualification
decision is recomputed from the marker report and immutable controls by
`CertifiedRagEvidenceTest`.
