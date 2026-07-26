# SmolLM3 3B Q4_K_M general RAG qualification

This directory contains controlled, same-host qualification evidence for the
exact SmolLM3 3B Q4_K_M artifact:

- source: `ggml-org/SmolLM3-3B-GGUF`
- source revision: `4965cb60b150737b68a0408c36aeefb65078f894`
- ModelJar coordinate:
  `org.modeljars.huggingface:ggml-org.smollm3-3b-gguf.q4_k_m:3.0.0-q4_k_m.1`
- SHA-256:
  `8334b850b7bd46238c16b0c550df2138f0889bf433809008cc17a8b05761863e`
- size: 1,915,305,312 bytes
- license: Apache-2.0
- host: dedicated 8-vCPU AMD EPYC-Milan, 32 GiB RAM, Linux amd64
- Models commit: `20733a0cd7cf9207e5641d7f749458a3021fc245`
- ModelJars profile commit used by the selected run: `372defe`
- Vectors commit: `c298a0b73970468794c1ba403022e2adc517e57e`
- Models JVM: GraalVM Community Java 25.0.3
- native controls: Ollama 0.32.0 and llama.cpp b10012 (`c71854292`)

Every report uses the same artifact bytes, nine general RAG cases, corpus
SHA-256
`4b27eba8f166c84ef19c53de825445a6d0097f9bd8efa20b2d7013f34621f83c`,
top-1 retrieval, the `chatml-no-think` template, a 2,048-token context, a
64-token output limit, one complete warmup, and three measured iterations.
Models, Ollama, and llama.cpp ran sequentially in separate isolated processes
on the same dedicated host with eight CPU threads. GraalVM applies only to the
Models rows; Ollama and llama.cpp used their native runtimes.

## Results

| Backend | Tier | p95 TTFT (ms) | p50 decode (tok/s) | p95 E2E (ms) | Correct |
|---|---|---:|---:|---:|---:|
| Models Rust/FFM v10, explicit profile controls | USABLE | 1,745.2 | 15.88 | 5,343.1 | 100% |
| Models Rust/FFM v10, ModelJar profile | USABLE; qualified | 1,727.9 | 15.93 | 5,318.9 | 100% |
| Ollama 0.32.0 | USABLE | 1,125.9 | 17.43 | 3,875.3 | 100% |
| llama.cpp b10012 | OFFLINE | 2,434.1 | 24.32 | 4,823.9 | 100% |

The marker-driven Models run reaches 91.35% of Ollama and 65.48% of llama.cpp
decode throughput. Its p95 end-to-end latency is 1.373x Ollama and 1.103x
llama.cpp. These ratios compare only the sequential process-isolated runs on
the same hardware described above; they are not cross-hardware performance
claims. The llama.cpp row remains `OFFLINE` because its p95 TTFT misses the
absolute 2-second `USABLE` gate, even though it remains a valid relative
comparator.

All 27 grounded answers were correct. Twelve answers retained model text and
all twelve were correct, twelve used validated extractive fallback, and three
correctly abstained. The reports pin grounding policy
`trusted-title-provenance-statement-anchors-safe-discourse-explicit-abstention-v14`.

The marker-only run supplied no decode, worker-count, batch-size, or attention
system properties. Diagnostics record the exact
`smollm3_3b_q4_k_m_epyc_milan_jdk25_rust_ffm` profile as `ENABLED`, with no
selector mismatch or missing launch argument. The profile selects eight native
workers, native quantized decode, 32-token batched prefill, and disabled batched
attention. All 27 prompt hashes, raw generations, grounding decisions,
evaluations, and raw evaluations match the explicit control exactly.

`qualification.json` binds the final marker report to both comparator reports.
`CertifiedRagEvidenceTest` recomputes the qualification decision and the
determinism and profile-selection assertions from these retained files.
