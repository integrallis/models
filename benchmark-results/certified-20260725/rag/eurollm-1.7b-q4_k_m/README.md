# EuroLLM 1.7B Q4_K_M multilingual RAG qualification

This directory contains the controlled, same-host qualification evidence for
the exact EuroLLM 1.7B Instruct Q4_K_M artifact:

- SHA-256:
  `1cade17f491ea46a686dbee51fbd52442e0f001f102380c3b9d66b4a77f84093`
- size: 1,045,157,088 bytes
- host: 8-vCPU AMD EPYC-Milan, 32 GiB RAM, Linux amd64
- JVM: GraalVM Community Java 25.0.3
- Models commit: `180fae9`
- ModelJars commit: `06da6f6`
- Vectors commit: `c298a0b`
- native controls: Ollama 0.32.0 and llama.cpp b10012 (`c71854292`)

Every report uses the same nine multilingual cases, corpus SHA-256
`d1d0889113c2d2f3e6c6fe3e551bdeecfb2d0324454d75d7ac55462619c88178`,
top-1 retrieval, ChatML prompts, a 2,048-token context, a 64-token output
limit, one complete warmup, and three measured iterations. All 27 candidate
attempts completed and all grounded answers were correct.

## Results

| Backend | Tier | p95 TTFT (ms) | p50 decode (tok/s) | p95 E2E (ms) | Correct |
|---|---|---:|---:|---:|---:|
| Models Rust/FFM, unprofiled | USABLE | 1,639.1 | 30.42 | 2,723.8 | 100% |
| Models Rust/FFM, ModelJar profile | USABLE; qualified | 1,514.3 | 31.86 | 2,548.4 | 100% |
| Ollama 0.32.0 | PRODUCTION_READY | 537.9 | 29.02 | 1,709.6 | 100% |
| llama.cpp b10012 | USABLE | 1,104.3 | 49.06 | 1,804.7 | 100% |

The marker-driven Models run reaches 109.8% of Ollama and 64.9% of llama.cpp
decode throughput. Its p95 end-to-end latency is 1.491x Ollama and 1.412x
llama.cpp, within the unchanged production policy limits of 1.5x and 2.0x.
The qualification report records `QUALIFIED` against both comparators.

The retained ModelJar profile selected a 64-token prefill batch, batched
attention score and value paths, final-layer prompt pruning, final-layer
KV-only prefill, and native quantized decode. The benchmark command supplied no
performance system properties; backend diagnostics prove the exact
EPYC-Milan/Java-25 profile was `ENABLED`.

## Quality scope

The production claim applies to the committed multilingual guarded-RAG
workload. The grounding policy used the model answer for 21 of 27 attempts,
validated extractive fallback for three, and correct abstention for three. All
21 accepted model answers were correct. This does not qualify EuroLLM for
arbitrary ungrounded generation.
