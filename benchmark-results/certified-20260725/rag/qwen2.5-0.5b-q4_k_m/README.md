# Qwen2.5 0.5B Q4_K_M general RAG qualification

This directory contains the controlled, same-host qualification evidence for
the exact Qwen2.5 0.5B Instruct Q4_K_M artifact:

- SHA-256:
  `74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db`
- size: 491,400,032 bytes
- host: 8-vCPU AMD EPYC-Milan, 32 GiB RAM, Linux amd64
- JVM: GraalVM Community Java 25.0.3
- Models commit: `0e3b8a7`
- ModelJars commit: `9e6e10a`
- Vectors commit: `c298a0b`
- native controls: Ollama 0.32.0 and llama.cpp b10012 (`c71854292`)

Every report uses the same nine general cases, corpus SHA-256
`4b27eba8f166c84ef19c53de825445a6d0097f9bd8efa20b2d7013f34621f83c`,
top-1 retrieval, ChatML prompts, a 2,048-token context, a 64-token output
limit, one complete warmup, and three measured iterations. All 27 candidate
attempts completed and all grounded answers were correct.

## Results

| Backend | Tier | p95 TTFT (ms) | p50 decode (tok/s) | p95 E2E (ms) | Correct |
|---|---|---:|---:|---:|---:|
| Models Rust/FFM, native decode disabled | PRODUCTION_READY | 520.1 | 22.80 | 1,820.2 | 100% |
| Models Rust/FFM, ModelJar profile | PRODUCTION_READY; qualified | 519.7 | 67.53 | 933.6 | 100% |
| Ollama 0.32.0 | PRODUCTION_READY | 375.3 | 46.05 | 1,088.0 | 100% |
| llama.cpp b10012 | PRODUCTION_READY | 488.8 | 98.05 | 783.4 | 100% |

The marker-driven Models run reaches 146.6% of Ollama and 68.9% of llama.cpp
decode throughput. Its p95 end-to-end latency is 0.858x Ollama and 1.192x
llama.cpp, within the unchanged production policy limits of 1.5x and 2.0x.
The qualification report records `QUALIFIED` against both comparators.

The retained ModelJar profile selects eight native kernel workers and native
quantized decode. The benchmark command supplied no performance system
properties; backend diagnostics prove the exact EPYC-Milan/Java-25 profile was
`ENABLED`. All 27 raw generations and grounded answers match the disabled
baseline exactly.

## Quality Scope

The production claim applies to the committed general guarded-RAG workload.
The grounding policy accepted 12 model answers, used validated extractive
fallback for 12, and correctly abstained three times. All 12 accepted model
answers were correct. This does not qualify Qwen2.5 for arbitrary ungrounded
generation.
