# Gemma 4 26B-A4B Instruct Q4_K_M qualification attempt

This directory retains the support and qualification evidence for one exact
artifact. It passed model integration and guarded-RAG quality checks, but it is
not production-qualified because its time to first token missed the absolute
usable gate.

- source: `ggml-org/gemma-4-26B-A4B-it-GGUF`
- source revision: `ae4d537a6345467d1c86bb5cc0d4505ff3ebe0f3`
- file: `gemma-4-26B-A4B-it-Q4_K_M.gguf`
- ModelJar: `org.modeljars.huggingface:ggml-org.gemma-4-26b-a4b-it-gguf.q4_k_m:4.0.0-q4_k_m.1`
- SHA-256: `88f4a13b0bb95f031a7fad973e10854122fb67ebc34d214d39a2f65053046abc`
- size: 16,796,015,136 bytes
- license recorded by the catalog: Apache-2.0
- Models implementation commit: `5ec2c5d446e60baa955ef58bd7e4e3dd52281747`
- native boundary: ABI 4, plan `rust-ffm-v12`
- host: 8-vCPU AMD EPYC-Milan, 32 GB RAM, Linux amd64
- JVM: Eclipse Adoptium Java 25.0.3

The standard report uses corpus SHA-256
`4b27eba8f166c84ef19c53de825445a6d0097f9bd8efa20b2d7013f34621f83c`,
the nine-case `general` workload, top-1 retrieval, the `gemma4` prompt
template, 2,048-token context, 64-token output limit, one complete warmup,
three measured iterations, and eight threads.

## Retained reports

| Report | Requests | p95 TTFT | p50 decode | p95 E2E | Correct | Tier |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| `models-rust-ffm.json` | 27 | 3,793.7 ms | 10.86 tok/s | 6,497.5 ms | 27/27 | OFFLINE |
| `plain-java-framework.json` | 9 | 6,199.2 ms | 10.75 tok/s | 9,159.4 ms | 9/9 | OFFLINE |
| `langchain4j.json` | 9 | 5,902.8 ms | 10.83 tok/s | 8,871.7 ms | 9/9 | OFFLINE |
| `spring-ai.json` | 9 | 6,175.1 ms | 10.87 tok/s | 9,127.2 ms | 9/9 | OFFLINE |

The standard run produced 21 correct retained model answers, three extractive
fallbacks, and three correct retrieval abstentions. Retrieval recall, MRR,
fact coverage, citation recall, citation precision, abstention accuracy, and
complete-answer accuracy are all 1.0. Its p95 retrieval latency is 1.91 ms,
p95 TPOT is 104.62 ms, median prefill is 25.00 tok/s, and peak RSS is
15,577,640,960 bytes.

Diagnostics prove that mapped expert weights, eight persistent workers, native
quantized projection kernels, load warmup, and the architecture-specific
32-token `gemma4-batched-prefill` path were enabled. The generic Llama planner's
separate `batched-prefill` diagnostic remains unsupported because Gemma 4 uses
its own complete prefill graph.

## Direct inference controls

The same exact artifact and prompt produced `Hello! How can I help you today?`
through every retained direct run.

| Runtime | Prompt/prefill | Decode | Notes |
| --- | ---: | ---: | --- |
| Models Rust/FFM | 509 ms median | 11.41 tok/s median | five fresh JVM runs |
| llama.cpp `a582222` | 620 ms | 12.78 tok/s | same-host control |
| Ollama | about 168 ms warm median | about 17.85 tok/s | five runs; first load 21.93 s |

These short-prompt measurements are diagnostics, not substitutes for the RAG
workload and not claims of engine parity. Models beats this llama.cpp control's
short prompt time but reaches about 89% of its decode throughput; both figures
remain behind Ollama.

## Decision

Support and integration pass. Production qualification fails because p95 TTFT
is 3,793.7 ms, above the 2,000 ms usable ceiling. Quality, retrieval, TPOT, and
end-to-end usable gates pass. No performance profile should be inferred from
this failed qualification.

Reproduce the exact-model boundaries with:

```shell
./gradlew :backend-java:gemma426BA4BSlowTest
./gradlew :backend-native:gemma426BA4BNativeSlowTest
./gradlew :models-rag-bench:gemma426BA4BFrameworkSlowTest
```

After `:models-rag-bench:installDist`, reproduce the standard RAG shape with
the pinned file and native library:

```shell
models-rag-bench/build/install/models-rag-bench/bin/models-rag-bench \
  --framework plain-java \
  --backend rust-ffm \
  --model ~/.jvllm/models/gemma-4-26B-A4B-it-Q4_K_M.gguf \
  --model-id ggml_org_gemma_4_26b_a4b_it_gguf_q4_k_m \
  --workload general \
  --prompt-template gemma4 \
  --context 2048 \
  --threads 8 \
  --max-tokens 64 \
  --warmups 1 \
  --iterations 3
```
