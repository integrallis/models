# H2O Danube3 500M Q4_K_M general RAG qualification

This directory retains controlled, same-host qualification evidence for the
exact H2O Danube3 500M Chat Q4_K_M artifact:

- source: `h2oai/h2o-danube3-500m-chat-GGUF`
- source revision: `73d306884d4f4ed09fa6d146fe3260e3b739dc74`
- ModelJar coordinate:
  `org.modeljars.huggingface:h2oai.h2o-danube3-500m-chat-gguf.q4_k_m:3.0.0-q4_k_m.1`
- SHA-256:
  `021f78849c5670ecb2aa4cd7c5972eee0a3c9e41e33e5902c408a2ab989f0b43`
- size: 317,877,408 bytes
- license: Apache-2.0
- host: dedicated 8-vCPU AMD EPYC-Milan, 32 GiB RAM, Linux amd64
- Models commit: `3d312534241dd80feee84d043c8727a874da9719`
- Vectors version: `0.1.5`
- Models JVM: Eclipse Adoptium Java 25.0.3
- native controls: Ollama 0.32.0 and llama.cpp b10012 (`c71854292`)

Every tuned report uses the same artifact bytes, nine general RAG cases,
corpus SHA-256
`4b27eba8f166c84ef19c53de825445a6d0097f9bd8efa20b2d7013f34621f83c`,
top-1 retrieval, the canonical `h2o` template, a 2,048-token context, a
64-token output limit, one complete warmup, and three measured iterations.
Models, Ollama, and llama.cpp ran sequentially in separate processes on the
same dedicated host with eight CPU threads.

## Results

| Backend | Tier | p95 TTFT (ms) | p50 decode (tok/s) | p95 E2E (ms) | Correct |
|---|---|---:|---:|---:|---:|
| Models Rust/FFM v12 | PRODUCTION_READY; qualified | 415.6 | 82.09 | 1,131.7 | 100% |
| Ollama 0.32.0 | PRODUCTION_READY | 234.8 | 40.65 | 2,662.7 | 100% |
| llama.cpp b10012 | PRODUCTION_READY | 437.6 | 134.00 | 913.6 | 100% |

Models reaches 201.9% of Ollama and 61.3% of llama.cpp median decode
throughput. Its p95 end-to-end latency is 0.425x Ollama and 1.239x llama.cpp.
These ratios compare only sequential, process-isolated runs on the same host;
they are not cross-hardware claims.

All 27 grounded answers were correct. Nine retained model text and all nine
were correct, fifteen used validated extractive fallback, and three correctly
abstained. The canonical `h2o` template is material: the exploratory
`h2o-direct` run retained only six model answers and correctly failed the
one-third model-contribution gate, so it is not the published configuration.

The tuned Models run enabled native quantized decode, eight native kernel
workers, a 32-token prefill batch, and single-row attention score/value paths.
`models-rust-ffm-baseline.json` repeats that exact profile for 27 requests; all
prompt hashes, raw model outputs, grounding decisions, and evaluations match
the qualified candidate byte for byte. This deterministic pair makes the
artifact-bound profile safe for automatic ModelJars selection.
`default-correctness/` separately proves the same artifact and template pass
all nine cases with library defaults and native quantized decode disabled.
`qualification.json` binds the tuned report to both comparator reports, and
`CertifiedRagEvidenceTest` recomputes the qualification from the retained
files.
