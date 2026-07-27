# SmolLM2 1.7B Q4_K_M general RAG qualification

This directory contains controlled, same-host qualification evidence for the
exact SmolLM2 1.7B Instruct Q4_K_M artifact:

- source: `HuggingFaceTB/SmolLM2-1.7B-Instruct-GGUF`
- source revision: `2d4a76a30b4af41ecd395c35725ac11688d4cfe4`
- ModelJar coordinate:
  `org.modeljars.huggingface:huggingfacetb.smollm2-1.7b-instruct-gguf.q4_k_m:2.0.0-q4_k_m.1`
- SHA-256:
  `decd2598bc2c8ed08c19adc3c8fdd461ee19ed5708679d1c54ef54a5a30d4f33`
- size: 1,055,609,536 bytes
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
top-1 retrieval, the `chatml` template, a 2,048-token context, a 64-token
output limit, one complete warmup, and three measured iterations. Models,
Ollama, and llama.cpp ran sequentially in separate isolated processes on the
same dedicated host with eight CPU threads. Java 25 applies only to the Models
rows; Ollama and llama.cpp used their native runtimes.

## Results

| Backend | Tier | p95 TTFT (ms) | p50 decode (tok/s) | p95 E2E (ms) | Correct |
|---|---|---:|---:|---:|---:|
| Models Rust/FFM v10, explicit profile controls | USABLE | 1,323.1 | 22.91 | 2,376.4 | 100% |
| Models Rust/FFM v10, ModelJar profile | USABLE; qualified | 1,371.0 | 22.83 | 2,420.0 | 100% |
| Ollama 0.32.0 | PRODUCTION_READY | 841.5 | 26.91 | 1,895.5 | 100% |
| llama.cpp b10012 | USABLE | 1,576.5 | 42.20 | 2,183.3 | 100% |

The marker-driven Models run reaches 84.84% of Ollama and 54.10% of llama.cpp
decode throughput. Its p95 end-to-end latency is 1.277x Ollama and 1.108x
llama.cpp. These ratios compare only the sequential process-isolated runs on
the same hardware described above; they are not cross-hardware claims.

All 27 grounded answers were correct. Fifteen answers retained model text and
all fifteen were correct, nine used validated extractive fallback, and three
correctly abstained. The reports pin grounding policy
`trusted-title-provenance-statement-anchors-safe-discourse-explicit-abstention-v18`.

The marker-only run supplied no decode, worker-count, batch-size, or attention
system properties. Diagnostics record the exact
`huggingfacetb_smollm2_1_7b_instruct_gguf_q4_k_m_epyc_milan_jdk25_rust_ffm`
profile as `ENABLED`, with no selector mismatch or missing launch argument.
All 27 prompt hashes, raw generations, grounding decisions, evaluations, and
raw evaluations match the explicit control exactly.

`qualification.json` binds the final marker report to both comparator reports.
`CertifiedRagEvidenceTest` recomputes qualification, determinism, and automatic
profile-selection assertions from these retained files.
