# SmolLM2 360M Q8_0 native RAG qualification

This directory contains the current-schema, same-host qualification matrix for
the exact SmolLM2 360M Instruct Q8_0 artifact:

- SHA-256:
  `48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201`
- size: 386,404,992 bytes
- host: 8-vCPU AMD EPYC-Milan, 32 GiB RAM, Linux amd64
- JVM: GraalVM Community Java 25.0.3, fixed 1 GiB heap
- Models uncached commit: `1ca2f8bdbe3c7d6dbd9ce8822e4339b6f6d5f944`
- Models prefix-cache commit: `2729c3a`
- Vectors commit: `fde9858901624d1661a1cf51195d2c59737bcf87`
- ModelJars commit: `936034cd55174fccd42121aa1dc7309c49d55d6c`
- native controls: Ollama 0.32.0 and llama.cpp b10012 (`c71854292`)

Every report uses the same nine cases, corpus SHA-256
`4b27eba8f166c84ef19c53de825445a6d0097f9bd8efa20b2d7013f34621f83c`,
top-1 retrieval, ChatML prompt bytes, 2,048-token context, 64-token output
limit, one complete warmup, and three measured iterations. The 27 prompt hashes
match across all reports.

## Results

| Java path | Backend | Prefix cache | Tier | p50 TTFT (ms) | p95 TTFT (ms) | p50 decode (tok/s) | p50 E2E (ms) | p95 E2E (ms) | Correct |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|
| plain Java | pure Java | no | USABLE | 1452.1 | 1621.9 | 44.12 | 2447.4 | 3007.3 | 100% |
| plain Java | Rust/FFM | no | USABLE | 958.6 | 1082.3 | 44.15 | 1958.1 | 2466.9 | 100% |
| plain Java | pure Java | yes | PRODUCTION_READY | 598.7 | 756.6 | 43.69 | 1552.4 | 2145.5 | 100% |
| plain Java | Rust/FFM | yes | PRODUCTION_READY | 432.2 | 552.8 | 43.92 | 1367.4 | 1935.0 | 100% |
| LangChain4j | Rust/FFM | yes | PRODUCTION_READY | 427.8 | 552.2 | 43.93 | 1380.3 | 1943.9 | 100% |
| Spring AI | Rust/FFM | yes | PRODUCTION_READY | 430.8 | 550.9 | 43.96 | 1385.7 | 1944.2 | 100% |
| plain Java | Ollama | engine-managed | PRODUCTION_READY | 75.8 | 197.2 | 43.73 | 1249.5 | 2485.7 | 100% |
| plain Java | llama.cpp | disabled | PRODUCTION_READY | 338.6 | 365.5 | 102.23 | 773.2 | 1004.4 | 100% |

The Models prefix cache retained 3,153 of 5,064 prompt tokens (62.3%). On the
Rust/FFM path it reduced p50 TTFT by 54.9%, p95 TTFT by 48.9%, p50 end-to-end
latency by 30.2%, p95 end-to-end latency by 21.6%, and total CPU time by 40.7%.
All 27 cached raw generations and grounded answers are byte-identical to the
uncached Rust/FFM control.

For the measured RAG workload, cached Rust/FFM reaches 91.4% of Ollama's p50
end-to-end performance and is faster at p95 because Ollama's generated response
lengths vary. It reaches 78.3% of llama.cpp's p50 TTFT performance and 51.9% of
its p95 end-to-end performance. Decode is 100.4% of Ollama and 43.0% of
llama.cpp.

## Quality scope

The production tier applies the committed trusted-retrieval grounding policy.
For Models, 9 of 27 trials used the model answer, 15 used the validated
extractive fallback, and 3 correctly abstained before generation. Raw model
answers were independently correct in 18 of 27 trials; Ollama and llama.cpp
were raw-correct in 21 of 27. This qualifies SmolLM2 360M Q8_0 for the guarded
RAG workflow represented here, not for arbitrary unguarded question answering.
