# MiniCPM5 1B Q4_K_M coding RAG qualification

This directory contains controlled, same-host qualification evidence for the
exact MiniCPM5 1B Q4_K_M artifact:

- SHA-256:
  `81b64d05a23b17b34c475f42b3e72fbde62d4b92cc34541f7a8031d0752deafa`
- size: 688,065,920 bytes
- host: 8-vCPU AMD EPYC-Milan, 32 GiB RAM, Linux amd64
- JVM: GraalVM Community Java 25.0.3
- Models commit: `ec2529729726958f5757a9b789354d503cca6158`
- ModelJars profile commit: `4b539b7f09a0fcccf0d93ccb230d3c6fe8b1b017`
- Vectors commit: `c298a0b73970468794c1ba403022e2adc517e57e`
- native controls: Ollama 0.32.0 and llama.cpp b10012 (`c71854292`)

Every report uses the same nine coding cases, corpus SHA-256
`6841c286837b4c45c06fe8d103b2e044b61a1bfe75a61b64fa04c7ca31b20e45`,
top-1 retrieval, the `minicpm5-no-think` prompt template, a 2,048-token context,
a 64-token output limit, one complete warmup, and three measured iterations.
Each backend ran sequentially in an isolated process on the same host.

## Results

| Backend | Tier | p95 TTFT (ms) | p50 decode (tok/s) | p95 E2E (ms) | Correct |
|---|---|---:|---:|---:|---:|
| Models Rust/FFM, default prefill | USABLE | 1,094.6 | 44.21 | 2,349.3 | 100% |
| Models Rust/FFM, ModelJar profile | USABLE; qualified | 1,042.4 | 49.01 | 2,152.1 | 100% |
| Ollama 0.32.0 | USABLE | 419.9 | 37.35 | 2,421.0 | 100% |
| llama.cpp b10012 | USABLE | 637.2 | 72.52 | 1,499.0 | 100% |

The marker-driven Models run reaches 131.2% of Ollama and 67.6% of llama.cpp
decode throughput. Its p95 end-to-end latency is 0.889x Ollama and 1.436x
llama.cpp, within the production policy limits of 1.5x and 2.0x.

The retained ModelJar profile selects a 64-token prefill batch, batched
attention score and value paths, eight native kernel workers, and native
quantized decode. The marker-only benchmark supplied no performance system
properties, and diagnostics prove the exact profile was `ENABLED`. Its raw
generations, grounding decisions, evaluations, and final answers are identical
to the default-prefill baseline.

## Quality Scope

All 27 grounded answers were correct. Twelve answers retained model text,
twelve used validated extractive fallback, and three correctly abstained. This
qualification applies to the exact artifact, workload, and controlled host
profile.
