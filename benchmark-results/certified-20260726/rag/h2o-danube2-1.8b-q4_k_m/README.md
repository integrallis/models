# H2O Danube2 1.8B Q4_K_M general RAG qualification

This directory contains controlled, same-host qualification evidence for the
exact H2O Danube2 1.8B Chat Q4_K_M artifact:

- source: `h2oai/h2o-danube2-1.8b-chat-GGUF`
- source revision: `8683c1aff099497d64785336532113f61ddf161d`
- ModelJar coordinate:
  `org.modeljars.huggingface:h2oai.h2o-danube2-1.8b-chat-gguf.q4_k_m:2.0.0-q4_k_m.1`
- SHA-256:
  `6a303ee6b94a961aa43e48eb11629e933c4438fae5e6db336318a5d33fe57d79`
- size: 1,112,145,056 bytes
- license: Apache-2.0
- host: dedicated 8-vCPU AMD EPYC-Milan, 32 GiB RAM, Linux amd64
- Models commit: `fc7f9754fb9e2dbb258ae19a87616689c2a0cddc`
- ModelJars profile commit used by the selected run:
  `05cc4fa5b740c905b137dee637a110db1fcf2eb3`
- Vectors commit: `c298a0b73970468794c1ba403022e2adc517e57e`
- Models JVM: Eclipse Adoptium Java 25.0.3
- native controls: Ollama 0.32.0 and llama.cpp b10012 (`c71854292`)

Every report uses the same artifact bytes, nine general RAG cases, corpus
SHA-256
`4b27eba8f166c84ef19c53de825445a6d0097f9bd8efa20b2d7013f34621f83c`,
top-1 retrieval, the `h2o-direct` template, a 2,048-token context, a 64-token
output limit, one complete warmup, and three measured iterations. Models,
Ollama, and llama.cpp ran sequentially in separate isolated processes on the
same dedicated host with eight CPU threads. Java 25 applies only to the Models
rows; Ollama and llama.cpp used their native runtimes.

## Results

| Backend | Tier | p95 TTFT (ms) | p50 decode (tok/s) | p95 E2E (ms) | Correct |
|---|---|---:|---:|---:|---:|
| Models Rust/FFM v10, explicit profile controls | USABLE | 1,411.7 | 25.92 | 3,881.9 | 100% |
| Models Rust/FFM v10, ModelJar profile | USABLE; qualified | 1,481.6 | 26.09 | 3,928.1 | 100% |
| Ollama 0.32.0 | PRODUCTION_READY | 377.0 | 24.37 | 3,111.8 | 100% |
| llama.cpp b10012 | USABLE | 1,704.1 | 37.24 | 3,460.2 | 100% |

The marker-driven Models run reaches 107.07% of Ollama and 70.07% of llama.cpp
decode throughput. Its p95 end-to-end latency is 1.262x Ollama and 1.135x
llama.cpp. These ratios compare only the sequential process-isolated runs on
the same hardware described above; they are not cross-hardware claims.

All 27 grounded answers were correct. Nine answers retained model text and all
nine were correct, fifteen used validated extractive fallback, and three
correctly abstained. The reports pin grounding policy
`trusted-title-provenance-statement-anchors-safe-discourse-explicit-abstention-v18`.

The marker-only run supplied no decode, worker-count, batch-size, or attention
system properties. Diagnostics record the exact
`h2oai_h2o_danube2_1_8b_chat_gguf_q4_k_m_epyc_milan_jdk25_rust_ffm` profile
as `ENABLED`, with no selector mismatch or missing launch argument. All 27
prompt hashes, raw generations, grounding decisions, evaluations, and raw
evaluations match the explicit control exactly.

`qualification.json` binds the final marker report to both comparator reports.
`CertifiedRagEvidenceTest` recomputes qualification, determinism, and automatic
profile-selection assertions from these retained files.
