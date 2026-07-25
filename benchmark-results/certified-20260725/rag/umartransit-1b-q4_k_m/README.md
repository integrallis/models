# UmarTransit-1B Q4_K_M transportation RAG qualification

This directory contains the controlled, same-host qualification evidence for
the exact UmarTransit-1B Q4_K_M artifact:

- SHA-256:
  `db1a4489626110145274f508b3fa30439516a47b4e721fe02d67df4679db5b9a`
- size: 986,048,096 bytes
- host: 8-vCPU AMD EPYC-Milan, 32 GiB RAM, Linux amd64
- JVM: GraalVM Community Java 25.0.3
- Models commit: `1456100`
- ModelJars profile commit: `3e96e18`
- Vectors commit: `c298a0b`
- native controls: Ollama 0.32.0 and llama.cpp b10012 (`c71854292`)

Every report uses the same nine transportation cases, corpus SHA-256
`367a1e35ea2449ba2abeb29c1fbbf911bb5130f14a3955c649b1f23afe35ceb2`,
top-1 retrieval, ChatML prompts, a 2,048-token context, a 64-token output
limit, one complete warmup, and three measured iterations.

## Results

| Backend | Tier | p95 TTFT (ms) | p50 decode (tok/s) | p95 E2E (ms) | Correct |
|---|---|---:|---:|---:|---:|
| Models Rust/FFM, default prefill | USABLE | 1,437.7 | 28.95 | 2,278.8 | 100% |
| Models Rust/FFM, ModelJar profile | USABLE; qualified | 1,402.7 | 30.93 | 2,172.4 | 100% |
| Ollama 0.32.0 | PRODUCTION_READY | 557.4 | 26.84 | 1,550.0 | 100% |
| llama.cpp b10012 | USABLE | 1,008.7 | 44.97 | 1,569.7 | 100% |

The marker-driven Models run reaches 115.2% of Ollama and 68.8% of llama.cpp
decode throughput. Its p95 end-to-end latency is 1.402x Ollama and 1.384x
llama.cpp, within the unchanged production policy limits of 1.5x and 2.0x.

The retained ModelJar profile selects a 64-token prefill batch, batched
attention score and value paths, eight native kernel workers, and native
quantized decode. The marker-only benchmark supplied no performance system
properties, and diagnostics prove the exact profile was `ENABLED`. Its raw
generations, grounding decisions, evaluations, and final answers are identical
to the default-prefill baseline.

## Quality Scope

All 27 grounded answers were correct. Fifteen answers retained model text with
derived trusted citations, nine used validated extractive fallback, and three
correctly abstained. This qualification applies to the exact artifact,
workload, and controlled host profile.
