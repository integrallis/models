# Nexus Legal 1.5B Q4_K_M legal RAG qualification

This directory contains controlled, same-host qualification evidence for the
exact Nexus Legal Q4_K_M artifact:

- source: `King3Djbl/nexus-legal-GGUF`
- source revision: `e55697a1bcbce71db82e7551c98cb6bd279620f6`
- ModelJar coordinate:
  `org.modeljars.huggingface:king3djbl.nexus-legal-gguf.q4_k_m:1.0.0-q4_k_m.1`
- SHA-256:
  `309437dc5aa980fd7025fc29b3a25740564244e124a0d806967ee05ed01b6d93`
- size: 986,045,472 bytes
- license: Apache-2.0
- host: dedicated 8-vCPU AMD EPYC-Milan, 32 GiB RAM, Linux amd64
- Models runtime commit: `9b1ae378865b4475f14f86b1ebb3aa09bc589127`
- ModelJars profile commit: `f5e972b0b087a5b03df1f13ecc067ee2b25c1d9c`
- Vectors commit: `c298a0b73970468794c1ba403022e2adc517e57e`
- Models JVM: Eclipse Adoptium Java 25.0.3, HotSpot C2
- native controls: Ollama 0.32.0 and llama.cpp b10012 (`c71854292`)

Every report uses the same artifact bytes, nine legal RAG cases, corpus
SHA-256
`a3499a08589094e7cb36299785bf128d5dad4a781fee96651878c267d61d0c2b`,
top-1 retrieval, the `chatml-direct` template, the exact leading-space stop
sequence ` Therefore`, a 2,048-token context, a 64-token output limit, one
complete warmup, and three measured iterations. Models, Ollama, and llama.cpp
ran sequentially in separate isolated processes on the same dedicated host
with eight CPU threads. The JDK applies only to Models; Ollama and llama.cpp
used their native runtimes.

## Results

| Backend | Tier | p95 TTFT (ms) | p50 decode (tok/s) | p95 E2E (ms) | Correct |
|---|---|---:|---:|---:|---:|
| Models Rust/FFM v10, explicit profile controls | PRODUCTION_READY | 750.7 | 27.70 | 2,767.6 | 100% |
| Models Rust/FFM v10, ModelJar profile | PRODUCTION_READY; qualified | 712.6 | 27.76 | 2,724.1 | 100% |
| Ollama 0.32.0 | PRODUCTION_READY | 582.9 | 31.47 | 2,957.7 | 100% |
| llama.cpp b10012 | USABLE | 1,050.2 | 45.36 | 2,409.5 | 100% |

The marker-driven Models run reaches 88.20% of Ollama and 61.18% of llama.cpp
decode throughput. Its p95 end-to-end latency is 0.921x Ollama and 1.131x
llama.cpp. These ratios compare only the sequential process-isolated runs on
the same hardware described above; they are not cross-hardware performance
claims.

All 27 grounded answers were correct. Twelve answers retained model text and
all twelve were correct, twelve used validated extractive fallback, and three
correctly abstained. The reports pin grounding policy
`trusted-title-provenance-statement-anchors-safe-discourse-explicit-abstention-v17`.

The marker-only run supplied no decode, worker-count, batch-size, or attention
system properties. Diagnostics record the exact
`king3djbl_nexus_legal_gguf_q4_k_m_epyc_milan_jdk25_rust_ffm` profile as
`ENABLED`, with no selector mismatch or missing launch argument. All 27 prompt
hashes, raw generations, grounding decisions, evaluations, and raw evaluations
match the explicit control exactly.

`qualification-marker.json` binds the final marker report to both comparator
reports. `CertifiedRagEvidenceTest` recomputes the qualification decision,
determinism assertions, and profile selection from these retained files.
