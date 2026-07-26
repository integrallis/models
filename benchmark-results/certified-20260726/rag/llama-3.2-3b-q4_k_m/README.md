# Llama 3.2 3B Q4_K_M general RAG qualification

This directory contains controlled, same-host qualification evidence for the
exact Llama 3.2 3B Instruct Q4_K_M artifact:

- source: `bartowski/Llama-3.2-3B-Instruct-GGUF`
- source revision: `5ab33fa94d1d04e903623ae72c95d1696f09f9e8`
- ModelJar coordinate:
  `org.modeljars.huggingface:bartowski.llama-3.2-3b-instruct-gguf.q4_k_m:3.2.0-q4_k_m.1`
- SHA-256:
  `6c1a2b41161032677be168d354123594c0e6e67d2b9227c84f296ad037c728ff`
- size: 2,019,377,696 bytes
- license: Llama 3.2 Community License
- host: dedicated 8-vCPU AMD EPYC-Milan, 32 GiB RAM, Linux amd64
- Models commit: `49b848f3e594d07322cbe2ea8418e9312609dd33`
- ModelJars profile commit used by the selected run:
  `ad3d2d4d68403fe8fea544877d421097e0fc65f7`
- Vectors commit: `c298a0b73970468794c1ba403022e2adc517e57e`
- Models JVM: GraalVM Community Java 25.0.3
- native controls: Ollama 0.32.0 and llama.cpp b10012 (`c71854292`)

Every report uses the same artifact bytes, nine general RAG cases, corpus
SHA-256
`4b27eba8f166c84ef19c53de825445a6d0097f9bd8efa20b2d7013f34621f83c`,
top-1 retrieval, the `llama3` prompt template, a 2,048-token context, a
64-token output limit, one complete warmup, and three measured iterations.
Models, Ollama, and llama.cpp ran sequentially in separate isolated processes
on the same dedicated host with eight CPU threads. GraalVM applies only to the
Models rows; Ollama and llama.cpp used their native runtimes.

## Results

| Backend | Tier | p95 TTFT (ms) | p50 decode (tok/s) | p95 E2E (ms) | Correct |
|---|---|---:|---:|---:|---:|
| Models Rust/FFM v10, explicit attention-off | USABLE | 1,591.3 | 15.65 | 3,765.6 | 100% |
| Models Rust/FFM v10, rejected attention-on | USABLE | 1,646.7 | 16.10 | 3,775.6 | 100% |
| Models Rust/FFM v10, ModelJar profile | USABLE; qualified | 1,621.7 | 15.70 | 3,789.3 | 100% |
| Ollama 0.32.0 | USABLE | 1,226.7 | 17.57 | 3,273.9 | 100% |
| llama.cpp b10012 | OFFLINE | 2,355.2 | 23.34 | 3,846.9 | 100% |

The marker-driven Models run reaches 89.33% of Ollama and 67.24% of llama.cpp
decode throughput. Its p95 end-to-end latency is 1.157x Ollama and 0.985x
llama.cpp. These ratios compare only the sequential process-isolated runs on
the same hardware described above; they are not cross-hardware performance
claims.

All 27 grounded answers were correct. Eighteen answers retained model text and
all eighteen were correct, six used validated extractive fallback, and three
correctly abstained. The reports pin grounding policy
`trusted-title-provenance-statement-anchors-safe-discourse-explicit-abstention-v13`.

The marker-only run supplied no decode, worker-count, batch-size, or attention
system properties. Diagnostics record the exact
`llama_3_2_3b_q4_k_m_epyc_milan_jdk25_rust_ffm` profile as `ENABLED`, with no
selector mismatch or missing launch argument. The profile selects eight native
workers, native quantized decode, 32-token batched prefill, and disabled
batched attention. All 27 prompt hashes, raw generations, grounding decisions,
evaluations, and raw evaluations match the explicit control exactly.

The complete attention-on run improved decode by about 2.6%, but regressed p95
TTFT, prefill throughput, and CPU use. Its p95 E2E result differed from the two
attention-off runs by less than 0.4% and changed direction between repetitions,
so no stable end-to-end gain was established. The experiment is retained as
rejected evidence and is not selected automatically.

`qualification.json` binds the final marker report to both comparator reports.
`CertifiedRagEvidenceTest` recomputes the qualification decision and the
determinism and profile-selection assertions from these retained files.
