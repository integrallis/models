# DeepSeek-Coder 1.3B Q4_K_M coding RAG qualification

This directory contains controlled, same-host qualification evidence for the
exact DeepSeek-Coder 1.3B Instruct Q4_K_M artifact:

- source: `TheBloke/deepseek-coder-1.3b-instruct-GGUF`
- source revision: `4595af8c3dff738094bd6c86054dfb5a90d5c41e`
- ModelJar coordinate:
  `org.modeljars.huggingface:thebloke.deepseek-coder-1.3b-instruct-gguf.q4_k_m:1.3.0-q4_k_m.1`
- SHA-256:
  `04cebb6fafa40ae628cf6bfeb76032ec792852f54020c559ad0a56b9f2839118`
- size: 873,582,624 bytes
- license: DeepSeek Model License
- host: dedicated 8-vCPU AMD EPYC-Milan, 32 GiB RAM, Linux amd64
- Models commit: `9c08ebd866d2c35b5e4695992f6ff62e4b0bf8d8`
- ModelJars profile commit used by the selected run: `0b50df8`
- Vectors commit: `c298a0b73970468794c1ba403022e2adc517e57e`
- Models JVM: GraalVM Community Java 25.0.3
- native controls: Ollama 0.32.0 and llama.cpp b10012 (`c71854292`)

Every report uses the same artifact bytes, nine coding RAG cases, corpus
SHA-256
`6841c286837b4c45c06fe8d103b2e044b61a1bfe75a61b64fa04c7ca31b20e45`,
top-1 retrieval, the corrected `deepseek` instruction template, a 2,048-token
context, a 64-token output limit, one complete warmup, and three measured
iterations. Models, Ollama, and llama.cpp ran sequentially in separate isolated
processes on the same dedicated host with eight CPU threads. GraalVM applies
only to the Models rows; Ollama and llama.cpp used their native runtimes.

## Results

| Backend | Tier | p95 TTFT (ms) | p50 decode (tok/s) | p95 E2E (ms) | Correct |
|---|---|---:|---:|---:|---:|
| Models Rust/FFM v10, explicit profile controls | USABLE | 1,049.6 | 30.07 | 3,006.1 | 100% |
| Models Rust/FFM v10, ModelJar profile | USABLE; qualified | 1,010.2 | 29.95 | 2,972.0 | 100% |
| Ollama 0.32.0 | PRODUCTION_READY | 596.4 | 32.68 | 2,272.1 | 100% |
| llama.cpp b10012 | USABLE | 1,218.5 | 50.46 | 2,452.8 | 100% |

The marker-driven Models run reaches 91.65% of Ollama and 59.35% of llama.cpp
decode throughput. Its p95 end-to-end latency is 1.308x Ollama and 1.212x
llama.cpp. These ratios compare only the sequential process-isolated runs on
the same hardware described above; they are not cross-hardware performance
claims.

All 27 grounded answers were correct. Nine answers retained model text and all
nine were correct, fifteen used validated extractive fallback, and three
correctly abstained. The reports pin grounding policy
`trusted-title-provenance-statement-anchors-safe-discourse-explicit-abstention-v14`.

The marker-only run supplied no decode, worker-count, batch-size, or attention
system properties. Diagnostics record the exact
`deepseek_coder_1_3b_q4_k_m_epyc_milan_jdk25_rust_ffm` profile as `ENABLED`,
with no selector mismatch or missing launch argument. The profile selects
eight native workers, native quantized decode, 32-token batched prefill, and
disabled batched attention. All 27 prompt hashes, raw generations, grounding
decisions, evaluations, and raw evaluations match the explicit control
exactly.

`qualification.json` binds the final marker report to both comparator reports.
`CertifiedRagEvidenceTest` recomputes the qualification decision and the
determinism and profile-selection assertions from these retained files.
