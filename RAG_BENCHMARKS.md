# Production RAG Benchmarks

Last updated: 2026-07-17

## Result

No model/backend combination measured in this study is ready for a production
claim yet.

Qwen3 1.7B Q8_0 is the strongest result. Its native paths are responsive for a
single request, and Java, Python, LangChain4j, and Spring AI all have effectively
the same inference performance. It nevertheless fails the quality gate: seven
of nine cases satisfy the exact fact, source-citation, and abstention contract.
The pure-Java backend answers the diagnostic case correctly, but reaches only
34.0% of llama.cpp decode throughput and takes 11.1 times as long to emit its
first token on the controlled CPU host.

The framework is not the bottleneck. Model instruction adherence and inference
engine performance are.

## What Acceptable Means

There is no universal RAG latency or quality threshold. llama.cpp and Ollama
expose timings, but neither defines a production service SLO. Published systems
also separate perceived latency from answer quality:

- The [DistServe OSDI 2024 presentation](https://www.usenix.org/system/files/osdi24_slides-zhong-yinmin.pdf)
  uses less than 1 second TTFT and roughly 100 ms per output token as interactive
  targets, with about 200 ms TTFT and 50 ms TPOT as more demanding targets.
- The [NVIDIA RAG benchmark methodology](https://docs.nvidia.com/rag/2.5.0/perf-benchmarks.html)
  measures TTFT, inter-token latency, end-to-end latency, input/output length,
  and concurrency separately. It warns that queued requests increase TTFT when
  KV-cache capacity is exhausted.
- [RAGAS](https://aclanthology.org/2024.eacl-demo.16/) treats retrieval quality,
  answer faithfulness, and answer relevance as distinct dimensions. Fast text
  generation is therefore insufficient for a production RAG claim.
- The [llama.cpp server](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/README.md)
  reports prompt and generation timings and supports continuous batching,
  metrics, schemas, prompt caching, and speculative decoding. The
  [Ollama API](https://docs.ollama.com/api/usage) reports load, prompt-evaluation,
  generation, and total durations.

Based on those anchors, this project uses explicit gates rather than calling a
model usable because it emits tokens:

| Gate | Production ready | Usable diagnostic |
| --- | ---: | ---: |
| Successful trials | 100% | 100% |
| Retrieval recall | >= 0.95 | >= 0.95 |
| MRR | >= 0.90 | >= 0.90 |
| Fact, citation, complete-answer accuracy | >= 0.90 | >= 0.90 |
| Abstention accuracy | 1.00 | 1.00 |
| p95 retrieval | <= 100 ms | <= 250 ms |
| p95 TTFT | <= 1,000 ms | <= 2,000 ms |
| p95 TPOT | <= 100 ms | <= 200 ms |
| p95 end to end | <= 5,000 ms | <= 10,000 ms |

These are ModelJars project SLOs, not an alleged industry standard. A deployment
must set tighter or looser limits from its own users, answer length, hardware,
concurrency, and risk profile.

## Controlled Workload

The executable `models-rag-bench` module runs the same workload through:

- plain Java and the Models generation API;
- LangChain4j `DefaultRetrievalAugmentor`;
- Spring AI `RetrievalAugmentationAdvisor`;
- the official Python Ollama client;
- Python against a revision-matched llama.cpp server; and
- optionally, the direct `llama-cpp-python` binding.

The committed synthetic Northstar corpus contains 12 short policy documents,
eight answerable questions, and one deliberately unanswerable question. Lucene
BM25 and Python BM25S use top-1 retrieval and agree on all retrieved documents
and rendered prompt hashes. The evaluator checks required facts, source IDs,
unsupported citations, and exact `INSUFFICIENT_CONTEXT` abstention without an
LLM judge. Temperature is zero, top-k is one, prompt caching is disabled, and
every measured Qwen3 1.7B path receives the same ChatML-no-think prompt bytes.

The strict abstention contract intentionally counts `INSUFFICIENT_CONTEXT.` as
instruction noncompliance. Normalizing the trailing period would raise Qwen3
1.7B to eight of nine semantically acceptable cases, or 88.9%, but it would
still miss the 90% gate because the telemedicine answer contradicts retrieved
context. Production applications should use schema-constrained output rather
than punctuation-sensitive control text.

## Environment

| Property | Controlled value |
| --- | --- |
| Host | Ubuntu Linux 6.8, x86_64 |
| CPU | AMD EPYC Milan, 8 logical CPUs |
| Memory | 30.6 GiB |
| JVM | Eclipse Temurin 25.0.3 |
| Models revision used for measurements | `f657a51` |
| Vectors revision | `8721107` |
| llama.cpp | `b10012`, commit `c71854292` |
| llama-cpp-python | `0.3.34` bundled native revision |
| Ollama | `0.32.0` |
| Main qualified artifact | Qwen3 1.7B Q8_0, 1,834,426,016 bytes |
| Artifact SHA-256 | `061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a` |
| Context / threads / output cap | 2,048 / 8 / 64 tokens |

The full Java and Python llama.cpp server rows use one warmup and three measured
iterations, or 27 requests. Framework, direct-binding, and Ollama rows use one
warmup and one measured iteration, or nine requests. They are sufficient to
find gross gaps, not to claim tail latency under production concurrency.

## Client And Framework Parity

All rows below use Qwen3 1.7B Q8_0 and the same nine cases. Throughput is median
decode throughput; latencies are p95.

| Application | Engine | Runs | Retrieval | Framework overhead | TTFT | TPOT | End to end | Decode | Exact quality |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Plain Java | llama.cpp | 27 | 5.1 ms | 1.4 ms | 1,294.6 ms | 40.4 ms | 3,756.5 ms | 25.46 tok/s | 77.8% |
| Python | llama.cpp | 27 | 0.3 ms | 0.4 ms | 1,322.1 ms | 41.0 ms | 3,789.6 ms | 25.53 tok/s | 77.8% |
| Python direct binding | llama.cpp 0.3.34 bundle | 9 | 0.3 ms | 1.3 ms | 1,306.8 ms | 45.6 ms | 3,624.0 ms | 24.99 tok/s | 77.8% |
| LangChain4j | llama.cpp | 9 | 3.7 ms | 2.7 ms | 1,258.1 ms | 39.8 ms | 3,712.2 ms | 25.56 tok/s | 77.8% |
| Spring AI | llama.cpp | 9 | 2.8 ms | 5.9 ms | 1,249.2 ms | 39.4 ms | 3,677.3 ms | 25.54 tok/s | 77.8% |
| Plain Java | Ollama | 9 | 2.0 ms | 1.0 ms | 676.1 ms | 66.2 ms | 3,873.4 ms | 18.40 tok/s | 77.8% |
| Python | Ollama | 9 | 0.3 ms | 0.3 ms | 686.1 ms | 65.5 ms | 4,018.6 ms | 18.11 tok/s | 77.8% |

The Java and Python llama.cpp server decode results differ by 0.3%. Their p95
TTFT differs by 2.1%. The direct Python binding is also close, although its
package bundles a different native revision. Java and Python Ollama results are
similarly close. The LangChain4j and Spring AI advisor layers add single-digit
milliseconds, which is immaterial beside model inference. A Java application
calling local Ollama experiences Ollama's engine speed, not the pure-Java
backend's speed.

The direct binding must call `Llama.reset()` before every measured request.
Without that reset, `llama-cpp-python` reuses the warmup prompt's KV prefix even
when its optional cache object is disabled; the benchmark initially observed an
invalid 42 ms TTFT. A regression test now enforces reset-before-generation. The
corrected result is 1,306.8 ms, which agrees with the server path.

On this host and model, Ollama cuts p95 TTFT roughly in half compared with the
pinned llama.cpp server but decodes 28% fewer tokens per second. Both remain
inside the latency portion of the project gate; both fail the same quality
portion.

## Pure Java Versus Native

These are one-case diagnostics, so they isolate engine behavior but do not
qualify a model. Each pair uses identical model-facing prompt bytes.

| Model | Format | Pure-Java TTFT | llama.cpp TTFT | Pure Java | llama.cpp | Java/native decode |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Qwen3 0.6B | Q4_0 | 3,150.8 ms | 331.8 ms | 21.46 tok/s | 98.38 tok/s | 21.8% |
| SmolLM2 360M | Q8_0 | 2,466.2 ms | 258.5 ms | 21.52 tok/s | 96.45 tok/s | 22.3% |
| Qwen3 1.7B | Q8_0 | 13,618.1 ms | 1,221.9 ms | 8.39 tok/s | 24.69 tok/s | 34.0% |

The long RAG prompt makes pure-Java prefill the larger practical problem:
TTFT is 9.5 to 11.1 times native in these diagnostics. Decode still remains
2.9 to 4.6 times slower. The Qwen3 1.7B pure-Java diagnostic takes 21.2 seconds
end to end, so it is not suitable for an interactive production request on
this CPU host.

## Model Findings

| Model | Scope | Best measured behavior | Production finding |
| --- | --- | --- | --- |
| SmolLM2 360M Q8_0 | Full suite plus engine diagnostics | Native diagnostic reaches 96.45 tok/s; pure Java reaches 21.52 tok/s | Full-suite exact quality is 25% native and 33.3% pure Java. Too small to follow the citation/abstention contract reliably. |
| Qwen3 0.6B Q4_0 | Engine and prompt diagnostics | llama.cpp reaches about 100 tok/s with ChatML-no-think | It extracts facts but repeatedly omits required citations. Not quality-qualified. |
| MiniCPM5 1B Q4_K_M | Prompt diagnostics | llama.cpp reaches about 70 tok/s | Raw prompts emit placeholder citations; ChatML spends the output budget in reasoning; the no-think prefix terminates immediately. Its template profile is unresolved. |
| Qwen3 1.7B Q8_0 | Full suite across seven application paths | Native paths satisfy the latency gate at concurrency one | Exact quality is 77.8%; semantically tolerant quality is 88.9%. Pure Java is too slow. |

Retrieval is perfect in every full run. The failed answerable case is therefore
generation, not search: Qwen3 1.7B says the context does not state whether a
telemedicine referral is required even though the retrieved passage explicitly
says no referral is required. This is exactly why token throughput cannot stand
in for RAG viability.

## Required Before A Production Claim

1. Qualify stronger 3B, 7B, and 8B instruction models. The sub-1B catalog is
   valuable for compatibility and constrained tasks, but it is not a credible
   default for this RAG contract.
2. Make GGUF chat-template metadata and model-specific reasoning controls
   first-class Models runtime behavior. A raw token generator is not yet a
   complete chat-model runtime.
3. Add schema- or grammar-constrained answers with structured citations and an
   explicit abstention field. Keep deterministic fact checks and add a reviewed
   faithfulness/relevance evaluator for less synthetic corpora.
4. Replace the tiny top-1 lexical control with a second, realistic benchmark:
   larger documents, chunking, embeddings, top-k retrieval, reranking, and
   adversarial unanswerable questions. Preserve this suite as the deterministic
   cross-engine control.
5. Run at least 30 measured iterations at concurrency 1, 2, 4, and 8, including
   warm/cold load, prompt-cache policy, cancellation, timeouts, and a reliability
   soak. Report queue time and throughput as well as per-request latency.
6. Measure the Ollama runner process tree rather than only `ollama serve`; the
   current Ollama RSS field covers the parent service and understates model
   memory.
7. Continue pure-Java prefill and decode optimization. The current RAG workload
   needs roughly a 3x decode gain and an order-of-magnitude TTFT reduction for
   Qwen3 1.7B to match llama.cpp on this host.
8. Repeat the matrix on Apple Silicon and a production-shaped x86 server. These
   results describe one controlled AVX2 CPU host, not every deployment target.

## Reproduction

Build and invocation examples are in
[`models-rag-bench/README.md`](models-rag-bench/README.md). The benchmark emits
JSON containing artifact and corpus hashes, raw answers, failures, environment,
settings, percentiles, retrieval metrics, quality metrics, CPU, and RSS. The
Python environment is locked by `uv.lock`; the direct Python binding is labeled
separately because its bundled llama.cpp revision may differ from the pinned
server.

Related framework references:

- [LangChain4j RAG](https://docs.langchain4j.dev/tutorials/rag/)
- [Spring AI RAG](https://docs.spring.io/spring-ai/reference/api/retrieval-augmented-generation.html)
- [Spring AI ChatModel API](https://docs.spring.io/spring-ai/reference/api/chatmodel.html)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [Ollama Python](https://github.com/ollama/ollama-python)
- [BM25S](https://github.com/xhluca/bm25s)
