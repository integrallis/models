# Qwen2.5 1.5B Q4_K_M general RAG qualification

This directory contains the controlled, same-host qualification evidence for
the exact Qwen2.5 1.5B Instruct Q4_K_M artifact:

- SHA-256:
  `6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e`
- size: 1,117,320,736 bytes
- host: 8-vCPU AMD EPYC-Milan, 32 GiB RAM, Linux amd64
- JVM: GraalVM Community Java 25.0.3
- Models commit: `0e3b8a7`
- ModelJars commit: `3e2e291`
- Vectors commit: `c298a0b`
- native controls: Ollama 0.32.0 and llama.cpp b10012 (`c71854292`)

Every report uses the same nine general cases, corpus SHA-256
`4b27eba8f166c84ef19c53de825445a6d0097f9bd8efa20b2d7013f34621f83c`,
top-1 retrieval, ChatML prompts, a 2,048-token context, a 64-token output
limit, one complete warmup, and three measured iterations.

## Results

| Backend | Tier | p95 TTFT (ms) | p50 decode (tok/s) | p95 E2E (ms) | Correct |
|---|---|---:|---:|---:|---:|
| Models Rust/FFM, default prefill | USABLE; failed Ollama latency gate | 1,923.9 | 28.69 | 2,905.2 | 100% |
| Models Rust/FFM, ModelJar profile | USABLE; qualified | 1,874.1 | 31.25 | 2,771.0 | 100% |
| Ollama 0.32.0 | PRODUCTION_READY | 677.4 | 28.76 | 1,864.3 | 100% |
| llama.cpp b10012 | USABLE | 1,138.2 | 44.93 | 1,806.2 | 100% |

The marker-driven Models run reaches 108.7% of Ollama and 69.6% of llama.cpp
decode throughput. Its p95 end-to-end latency is 1.486x Ollama and 1.534x
llama.cpp, within the unchanged production policy limits of 1.5x and 2.0x.

The retained ModelJar profile selects a 64-token prefill batch, batched
attention score and value paths, eight native kernel workers, and native
quantized decode. Each prefill control improved a one-iteration screen
independently. The marker-only benchmark supplied no performance system
properties, and diagnostics prove the exact profile was `ENABLED`.

## Quality Scope

All 27 grounded answers were correct. The grounding policy accepted nine model
answers, used validated extractive fallback for 15, and correctly abstained
three times. All nine accepted model answers were correct. This qualification
applies to the exact artifact, workload, and controlled host profile.
