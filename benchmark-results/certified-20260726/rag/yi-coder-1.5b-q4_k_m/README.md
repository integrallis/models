# Yi-Coder 1.5B Q4_K_M general RAG qualification

This directory contains controlled, same-host qualification evidence for the
exact Yi-Coder 1.5B Chat Q4_K_M artifact:

- source: `bartowski/Yi-Coder-1.5B-Chat-GGUF`
- source revision: `442545c4f577cfdab9a70b38fec719c47d393b8b`
- original model: `01-ai/Yi-Coder-1.5B-Chat`
- ModelJar coordinate:
  `org.modeljars.huggingface:bartowski.yi-coder-1.5b-chat-gguf.q4_k_m:1.5.0-q4_k_m.1`
- SHA-256:
  `61c72ab3dd56a15b8a3aee30f55e180675daa87d04219e8afc32e8852c175f32`
- size: 963,674,304 bytes
- license: Apache-2.0
- host: dedicated 8-vCPU AMD EPYC-Milan, 32 GiB RAM, Linux amd64
- Models commit: `03238c90419d0883a15b6868ea4308d8167d6379`
- ModelJars profile commit used by the selected run:
  `5733d4ee8b70464c28bf916272fceba999e45aa8`
- Vectors commit: `c298a0b73970468794c1ba403022e2adc517e57e`
- Models JVM: Eclipse Adoptium Java 25.0.3
- native controls: Ollama 0.32.0 and llama.cpp b10012 (`c71854292`)

Every report uses the same artifact bytes, nine general RAG cases, corpus
SHA-256
`4b27eba8f166c84ef19c53de825445a6d0097f9bd8efa20b2d7013f34621f83c`,
top-1 retrieval, the `chatml-answer` template, a 2,048-token context, a
64-token output limit, one complete warmup, and three measured iterations.
Models, Ollama, and llama.cpp ran sequentially in separate isolated processes
on the same dedicated host with eight CPU threads. Java 25 applies only to the
Models rows; Ollama and llama.cpp used their native runtimes.

## Results

| Backend | Tier | p95 TTFT (ms) | p50 decode (tok/s) | p95 E2E (ms) | Correct |
|---|---|---:|---:|---:|---:|
| Models Rust/FFM v10, explicit profile controls | USABLE | 1,122.3 | 29.92 | 3,222.9 | 100% |
| Models Rust/FFM v10, ModelJar profile | USABLE; qualified | 1,129.1 | 29.74 | 3,245.0 | 100% |
| Ollama 0.32.0 | PRODUCTION_READY | 559.2 | 28.99 | 2,924.5 | 100% |
| llama.cpp b10012 | USABLE | 1,227.8 | 48.98 | 2,420.5 | 100% |

The marker-driven Models run reaches 102.58% of Ollama and 60.71% of llama.cpp
decode throughput. Its p95 end-to-end latency is 1.110x Ollama and 1.341x
llama.cpp. These ratios compare only the sequential process-isolated runs on
the same hardware described above; they are not cross-hardware claims.

All 27 grounded answers were correct. Fifteen answers retained model text and
all fifteen were correct, nine used validated extractive fallback, and three
correctly abstained. The reports pin grounding policy
`trusted-title-provenance-statement-anchors-safe-discourse-explicit-abstention-v18`.

The marker-only run supplied no decode, worker-count, batch-size, or attention
system properties. Diagnostics record the exact
`bartowski_yi_coder_1_5b_chat_gguf_q4_k_m_epyc_milan_jdk25_rust_ffm` profile
as `ENABLED`, with no selector mismatch or missing launch argument. All 27
prompt hashes, raw generations, grounding decisions, evaluations, and raw
evaluations match the explicit control exactly.

`qualification.json` binds the final marker report to both comparator reports.
`CertifiedRagEvidenceTest` recomputes qualification, determinism, and automatic
profile-selection assertions from these retained files.
