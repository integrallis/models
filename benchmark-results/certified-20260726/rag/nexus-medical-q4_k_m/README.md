# Nexus Medical 1.5B Q4_K_M healthcare RAG qualification

This directory contains controlled, same-host qualification evidence for the
exact Nexus Medical Q4_K_M artifact:

- source: `King3Djbl/nexus-medical-GGUF`
- source revision: `ee4785b0b4c5bf646801f5806248244b661da5e6`
- ModelJar coordinate:
  `org.modeljars.huggingface:king3djbl.nexus-medical-gguf.q4_k_m:1.0.0-q4_k_m.1`
- SHA-256:
  `f26138b8e049e91a3e006f215a3618e3da0fef28755f73fe83bea19f73c700ea`
- size: 986,045,632 bytes
- license: Apache-2.0
- host: dedicated 8-vCPU AMD EPYC-Milan, 32 GiB RAM, Linux amd64
- Models runtime commit: `b21567803a5832b5aa54f75f030cdb6dc77df3eb`
- ModelJars profile commit: `55209cec9532b8d629b4eb3a5c37fd798163f028`
- Vectors commit: `c298a0b73970468794c1ba403022e2adc517e57e`
- Models JVM: Eclipse Adoptium Java 25.0.3, HotSpot C2
- native controls: Ollama 0.32.0 and llama.cpp b10012 (`c71854292`)

Every report uses the same artifact bytes, nine healthcare RAG cases, corpus
SHA-256
`50b20fdae13db749bdf1bc029c681d53fff8895fb11aa3a177264ab3e6fd62f0`,
top-1 retrieval, the `chatml-direct` template, the exact leading-space stop
sequence ` So the answer is`, a 2,048-token context, a 64-token output limit,
one complete warmup, and three measured iterations. Models, Ollama, and
llama.cpp ran sequentially in separate isolated processes on the same
dedicated host with eight CPU threads. The JDK applies only to Models; Ollama
and llama.cpp used their native runtimes.

## Results

| Backend | Tier | p95 TTFT (ms) | p50 decode (tok/s) | p95 E2E (ms) | Correct |
|---|---|---:|---:|---:|---:|
| Models Rust/FFM v10, explicit profile controls | PRODUCTION_READY | 706.9 | 29.26 | 2,821.1 | 100% |
| Models Rust/FFM v10, ModelJar profile | PRODUCTION_READY; qualified | 713.6 | 29.38 | 2,810.7 | 100% |
| Ollama 0.32.0 | PRODUCTION_READY | 354.6 | 26.93 | 3,926.3 | 100% |
| llama.cpp b10012 | USABLE | 1,080.2 | 45.41 | 2,552.8 | 100% |

The marker-driven Models run reaches 109.08% of Ollama and 64.69% of
llama.cpp decode throughput. Its p95 end-to-end latency is 0.716x Ollama and
1.101x llama.cpp. These ratios compare only the sequential process-isolated
runs on the same hardware described above; they are not cross-hardware
performance claims.

All 27 grounded answers were correct. Twelve answers retained model text and
all twelve were correct, twelve used validated extractive fallback, and three
correctly abstained. The reports pin grounding policy
`trusted-title-provenance-statement-anchors-safe-discourse-explicit-abstention-v18`.

The marker-only run supplied no decode, worker-count, batch-size, or attention
system properties. Diagnostics record the exact
`king3djbl_nexus_medical_gguf_q4_k_m_epyc_milan_jdk25_rust_ffm` profile as
`ENABLED`, with no selector mismatch or missing launch argument. All 27 prompt
hashes, raw generations, grounding decisions, evaluations, and raw evaluations
match the explicit control exactly.

`qualification-marker.json` binds the final marker report to both comparator
reports. `CertifiedRagEvidenceTest` recomputes the qualification decision,
determinism assertions, and profile selection from these retained files.
