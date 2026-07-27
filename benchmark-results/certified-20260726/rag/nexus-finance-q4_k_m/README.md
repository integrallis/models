# Nexus Finance 1.5B Q4_K_M finance RAG qualification

This directory contains controlled, same-host qualification evidence for the
exact Nexus Finance Q4_K_M artifact:

- source: `King3Djbl/nexus-finance-GGUF`
- source revision: `c3388d8f6126505a7f998ca1b5354f286c9ca891`
- ModelJar coordinate:
  `org.modeljars.huggingface:king3djbl.nexus-finance-gguf.q4_k_m:1.0.0-q4_k_m.1`
- SHA-256:
  `a849ca49f91889b614518e9242ceec6e24034289de0f3f8621784f3e1b77bb60`
- size: 986,045,632 bytes
- license: Apache-2.0
- host: dedicated 8-vCPU AMD EPYC-Milan, 32 GiB RAM, Linux amd64
- Models runtime commit: `9b1ae378865b4475f14f86b1ebb3aa09bc589127`
- ModelJars profile commit: `f5e972b0b087a5b03df1f13ecc067ee2b25c1d9c`
- Vectors commit: `c298a0b73970468794c1ba403022e2adc517e57e`
- Models JVM: Eclipse Adoptium Java 25.0.3, HotSpot C2
- native controls: Ollama 0.32.0 and llama.cpp b10012 (`c71854292`)

Every report uses the same artifact bytes, nine finance RAG cases, corpus
SHA-256
`923a8686ef45df6c09c08c0bd80e08ab40a96ad0b4d37fb356729dbd6738befc`,
top-1 retrieval, the `chatml-no-think` template, a newline stop sequence, a
2,048-token context, a 64-token output limit, one complete warmup, and three
measured iterations. Models, Ollama, and llama.cpp ran sequentially in
separate isolated processes on the same dedicated host with eight CPU threads.
The JDK applies only to Models; Ollama and llama.cpp used their native
runtimes.

## Results

| Backend | Tier | p95 TTFT (ms) | p50 decode (tok/s) | p95 E2E (ms) | Correct |
|---|---|---:|---:|---:|---:|
| Models Rust/FFM v10, explicit profile controls | PRODUCTION_READY | 791.9 | 30.46 | 2,258.3 | 100% |
| Models Rust/FFM v10, ModelJar profile | PRODUCTION_READY; qualified | 814.6 | 30.17 | 2,293.1 | 100% |
| Ollama 0.32.0 | PRODUCTION_READY | 615.7 | 28.89 | 2,942.3 | 100% |
| llama.cpp b10012 | USABLE | 1,061.2 | 45.76 | 2,465.1 | 100% |

The marker-driven Models run reaches 104.44% of Ollama and 65.93% of
llama.cpp decode throughput. Its p95 end-to-end latency is 0.779x Ollama and
0.930x llama.cpp. These ratios compare only the sequential process-isolated
runs on the same hardware described above; they are not cross-hardware
performance claims.

All 27 grounded answers were correct. Fifteen answers retained model text and
all fifteen were correct, nine used validated extractive fallback, and three
correctly abstained. The reports pin grounding policy
`trusted-title-provenance-statement-anchors-safe-discourse-explicit-abstention-v17`.

The marker-only run supplied no decode, worker-count, batch-size, or attention
system properties. Diagnostics record the exact
`king3djbl_nexus_finance_gguf_q4_k_m_epyc_milan_jdk25_rust_ffm` profile as
`ENABLED`, with no selector mismatch or missing launch argument. All 27 prompt
hashes, raw generations, grounding decisions, evaluations, and raw evaluations
match the explicit control exactly.

`qualification-marker.json` binds the final marker report to both comparator
reports. `CertifiedRagEvidenceTest` recomputes the qualification decision,
determinism assertions, and profile selection from these retained files.
