# Gemma 4 26B-A4B Instruct Q4_K_M qualification

> Superseded on 2026-08-25: this Models run did not preserve trusted prompt
> segments through tokenization. Corrected evidence confirms the qualification;
> see
> [`certified-20260825/rag/gemma4-26b-a4b-q4_k_m/`](../../../certified-20260825/rag/gemma4-26b-a4b-q4_k_m/).

This directory retains the support and qualification evidence for one exact
artifact. The commit-bound standard report passes model integration,
guarded-RAG quality, and every absolute usable performance gate.

- source: `ggml-org/gemma-4-26B-A4B-it-GGUF`
- source revision: `ae4d537a6345467d1c86bb5cc0d4505ff3ebe0f3`
- file: `gemma-4-26B-A4B-it-Q4_K_M.gguf`
- ModelJar: `org.modeljars.huggingface:ggml-org.gemma-4-26b-a4b-it-gguf.q4_k_m:4.0.0-q4_k_m.2`
- SHA-256: `88f4a13b0bb95f031a7fad973e10854122fb67ebc34d214d39a2f65053046abc`
- size: 16,796,015,136 bytes
- license recorded by the catalog: Apache-2.0
- Models implementation commit: `37fc4a8b9d421505487b678c7ce841d1baa20eb4`
- standard report SHA-256: `2234bd65597ed56c2dbc2936e389e145c3bf471fa2d7202cd7e6a802f2523ed8`
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
| `models-rust-ffm.json` | 27 | 1,834.7 ms | 12.84 tok/s | 4,503.1 ms | 27/27 | USABLE |
| `llama.cpp.json` | 27 | 3,728.3 ms | 15.93 tok/s | 5,847.4 ms | 27/27 | OFFLINE |
| `ollama.json` | 27 | 4,111.8 ms | 13.65 tok/s | 6,556.7 ms | 27/27 | OFFLINE |
| `plain-java-framework.json` | 9 | 6,199.2 ms | 10.75 tok/s | 9,159.4 ms | 9/9 | OFFLINE |
| `langchain4j.json` | 9 | 5,902.8 ms | 10.83 tok/s | 8,871.7 ms | 9/9 | OFFLINE |
| `spring-ai.json` | 9 | 6,175.1 ms | 10.87 tok/s | 9,127.2 ms | 9/9 | OFFLINE |

The three framework reports are retained integration controls from the earlier
implementation; they are not the performance qualification. The standard run
produced 18 correct retained model answers, six extractive fallbacks, and three
correct retrieval abstentions. Retrieval recall, MRR,
fact coverage, citation recall, citation precision, abstention accuracy, and
complete-answer accuracy are all 1.0. Its p95 retrieval latency is 1.89 ms,
p95 TPOT is 89.05 ms, median prefill is 50.91 tok/s, and peak RSS is
15,652,118,528 bytes.

Diagnostics prove that mapped expert weights, eight persistent workers, native
quantized projection kernels, load warmup, and the architecture-specific
128-token `gemma4-batched-prefill` path were enabled. The generic Llama planner's
separate `batched-prefill` diagnostic remains unsupported because Gemma 4 uses
its own complete prefill graph.

## Relative controls

The machine-readable `qualification.json` applies
`production-rag-model-contribution-v4` to the standard report and the two
same-host controls. Models reaches 80.59% of llama.cpp and 94.03% of Ollama
median decode throughput. Its p95 end-to-end latency is 77.01% of llama.cpp
and 68.68% of Ollama. Both comparisons pass their backend-specific relative
floors and ceilings, and the final verdict is `QUALIFIED`.

## Historical direct inference controls

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

Support, integration, and production qualification pass. The p95 TTFT is
1,834.7 ms, below the 2,000 ms usable ceiling. Quality, retrieval, TPOT, and
end-to-end usable gates also pass, so the exact `.2` marker is qualified at the
`USABLE` tier. The result applies only to the pinned artifact, implementation,
settings, and qualification host recorded here.

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
  --backend-version models@37fc4a8b9d421505487b678c7ce841d1baa20eb4 \
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
