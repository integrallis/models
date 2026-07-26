# Llama 3.2 1B Instruct Q4_K_M general RAG qualification

This directory contains controlled, same-host qualification evidence for the
exact Llama 3.2 1B Instruct Q4_K_M artifact:

- SHA-256:
  `6f85a640a97cf2bf5b8e764087b1e83da0fdb51d7c9fab7d0fece9385611df83`
- size: 807,694,464 bytes
- host: 8-vCPU AMD EPYC-Milan, 32 GiB RAM, Linux amd64
- JVM: GraalVM Community Java 25.0.3
- Models commit: `dd3150f414c8f1c741963c7953224d09e744f94c`
- ModelJars profile commit: `00215d3542a93c2f1c6ca41251f7244cdf5f8246`
- Vectors commit: `c298a0b73970468794c1ba403022e2adc517e57e`
- native controls: Ollama 0.32.0 and llama.cpp b10012 (`c71854292`)

Every report uses the same nine general-knowledge cases, corpus SHA-256
`4b27eba8f166c84ef19c53de825445a6d0097f9bd8efa20b2d7013f34621f83c`,
top-1 retrieval, the `llama3` prompt template, a 2,048-token context, a
64-token output limit, one complete warmup, and three measured iterations.
Each backend ran sequentially in an isolated process on the same host.

## Results

| Backend | Tier | p95 TTFT (ms) | p50 decode (tok/s) | p95 E2E (ms) | Correct |
|---|---|---:|---:|---:|---:|
| Models Rust/FFM, default prefill | USABLE | 1,348.4 | 37.71 | 2,023.1 | 100% |
| Models Rust/FFM, ModelJar profile | USABLE; qualified | 1,312.6 | 38.73 | 1,962.3 | 100% |
| Ollama 0.32.0 | USABLE | 543.3 | 34.75 | 1,640.1 | 100% |
| llama.cpp b10012 | USABLE | 799.4 | 57.06 | 1,412.5 | 100% |

The marker-driven Models run reaches 111.4% of Ollama and 67.9% of llama.cpp
decode throughput. Its p95 end-to-end latency is 1.196x Ollama and 1.389x
llama.cpp, within the production policy limits of 1.5x and 2.0x. These ratios
are measurements for this exact host and workload, not cross-hardware claims.

The retained ModelJar profile selects a 64-token prefill batch, batched
attention score and value paths, eight native kernel workers, and native
quantized decode. The marker-only benchmark supplied no performance system
properties, and diagnostics prove the exact profile was `ENABLED`. Its raw
generations, grounding decisions, evaluations, and final answers are identical
to the default-prefill baseline.

## Quality Scope

All 27 grounded answers were correct. Fifteen answers retained model text,
nine used validated extractive fallback, and three correctly abstained. This
qualification applies only to the exact artifact, workload, software versions,
and controlled host profile recorded above.
