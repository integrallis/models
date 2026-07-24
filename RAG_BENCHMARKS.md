# Production RAG Benchmarks

Last updated: 2026-07-24

## Result

No measured path is ready for a production RAG claim.

Ollama has acceptable single-request latency on this workload: p95 TTFT is
693.6 ms and p95 end-to-end latency is 4.03 seconds. llama.cpp is usable but
misses the project's 1-second production TTFT gate at 1.29 seconds. Both
produce seven of nine exact contract-compliant answers, or 77.8%. Treating a
trailing period after `INSUFFICIENT_CONTEXT` as semantically equivalent raises
both native paths to eight of nine, or 88.9%, which still misses the 90%
quality gate.

The tuned pure-Java Qwen3 1.7B candidate decodes at 18.39 tok/s, essentially
the same as Ollama's 18.60 tok/s and 71.8% of llama.cpp's 25.62 tok/s. Its
prompt path is not competitive: p95 TTFT is 5.35 seconds, 4.13x llama.cpp and
7.71x Ollama. It also produces only six of nine exact answers and leaks Qwen
control tokens or continues beyond an end marker in the failed cases.

LangChain4j and Spring AI do not materially change these results. Their
measured adapter overhead remains below 24 ms at p95, while inference consumes
hundreds or thousands of milliseconds. The remaining blockers are pure-Java
prefill, correct chat/stop-token behavior, and model instruction adherence.

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
instruction noncompliance. Normalizing that trailing period raises each native
path to eight of nine semantically acceptable cases, or 88.9%, but it still
misses the gate because the telemedicine answer contradicts retrieved context.
The same normalization would raise pure Java only to seven of nine, or 77.8%;
its extra citation and control-token failures remain. Production applications
should use schema-constrained output rather than punctuation-sensitive control
text.

## Environment

| Property | Controlled value |
| --- | --- |
| Host | Ubuntu Linux 6.8, x86_64 |
| CPU | AMD EPYC Milan, 8 logical CPUs |
| Memory | 30.6 GiB |
| JVM | GraalVM Community Java 25.0.3 for pure Java; Temurin 25.0.3 for native clients |
| Models revision used for measurements | `10949c7` |
| Vectors revision | `fde9858` |
| ModelJars revision | `b6575a1` |
| llama.cpp | `b10012`, commit `c71854292` |
| llama-cpp-python | `0.3.34` bundled native revision |
| Ollama | `0.32.0` |
| Main qualified artifact | Qwen3 1.7B Q8_0, 1,834,426,016 bytes |
| Artifact SHA-256 | `061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a` |
| Context / threads / output cap | 2,048 / 8 / 64 tokens |

The fresh plain-Java rows use one warmup over all cases and three measured
iterations, or 27 requests per backend. LangChain4j and Spring AI rows use one
warmup and one measured iteration, or nine requests per adapter/backend pair.
They are sufficient to find gross gaps, not to claim tail latency under
production concurrency.

## Client And Framework Parity

All rows below use Qwen3 1.7B Q8_0 and the same nine cases. Throughput is median
decode throughput; latencies are p95.

| Application | Engine | Runs | Retrieval | Framework overhead | TTFT | TPOT | End to end | Decode | Exact quality |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Plain Java | pure Java candidate | 27 | 2.0 ms | 23.2 ms | 5,347.0 ms | 56.3 ms | 8,836.2 ms | 18.39 tok/s | 66.7% |
| LangChain4j | pure Java candidate | 9 | 2.2 ms | 23.7 ms | 5,445.9 ms | 54.9 ms | 8,916.9 ms | 18.31 tok/s | 66.7% |
| Spring AI | pure Java candidate | 9 | 2.2 ms | 23.7 ms | 5,339.7 ms | 54.8 ms | 8,804.7 ms | 18.53 tok/s | 66.7% |
| Plain Java | llama.cpp | 27 | 4.0 ms | 1.7 ms | 1,293.4 ms | 39.9 ms | 3,746.6 ms | 25.62 tok/s | 77.8% |
| LangChain4j | llama.cpp | 9 | 6.5 ms | 1.6 ms | 1,264.4 ms | 39.8 ms | 3,678.8 ms | 25.52 tok/s | 77.8% |
| Spring AI | llama.cpp | 9 | 4.6 ms | 4.8 ms | 1,252.9 ms | 40.2 ms | 3,700.6 ms | 25.36 tok/s | 77.8% |
| Plain Java | Ollama | 27 | 1.9 ms | 1.0 ms | 693.6 ms | 61.1 ms | 4,030.2 ms | 18.60 tok/s | 77.8% |
| LangChain4j | Ollama | 9 | 2.1 ms | 1.6 ms | 677.3 ms | 67.9 ms | 3,655.6 ms | 19.90 tok/s | 77.8% |
| Spring AI | Ollama | 9 | 2.1 ms | 1.9 ms | 671.6 ms | 60.2 ms | 3,841.2 ms | 17.68 tok/s | 77.8% |

The three client styles agree within normal short-run variance. A Java
application calling local Ollama experiences Ollama's engine speed, not the
pure-Java backend's speed. The framework choice is therefore an ergonomics and
integration decision, not an inference-performance decision.

The plain native paths produce the same answers and quality failures. Ollama
cuts p95 TTFT by 46.4% relative to the pinned llama.cpp server but decodes
27.4% fewer tokens per second. Ollama satisfies the latency part of the
project's production gate at concurrency one; llama.cpp satisfies the usable
diagnostic gate. Both fail quality.

## Pure Java Versus Native

The current full RAG run makes the bottleneck explicit:

| Metric | Pure Java candidate | llama.cpp | Ollama |
| --- | ---: | ---: | ---: |
| p95 TTFT | 5,347.0 ms | 1,293.4 ms | 693.6 ms |
| p95 TPOT | 56.3 ms | 39.9 ms | 61.1 ms |
| p95 end to end | 8,836.2 ms | 3,746.6 ms | 4,030.2 ms |
| Median decode | 18.39 tok/s | 25.62 tok/s | 18.60 tok/s |
| Exact contract quality | 66.7% | 77.8% | 77.8% |

The RAG clients receive backend-specific prompt-evaluation timing fields, so
the separate controlled inference matrix is the authoritative cross-engine
prefill comparison. Pure-Java decode has reached 71.8% of llama.cpp and 98.8%
of Ollama on this RAG workload. Prompt processing and output lifecycle
behavior, not decode, prevent an interactive production claim.

The pure-Java output failures are concrete:

- `telemedicine-benefit` emits `INSUFFICIENT_CONTEXT`, then continues through
  Qwen end markers into an incomplete contradictory answer.
- `break-glass` answers the facts but invents an extra citation and continues
  into another assistant prompt.
- `unsupported-lunar-vehicle` includes the correct abstention marker but leaks
  a control token and continues generating a rationale.

This is not evaluator noise alone. Models must consume GGUF chat-template and
end-of-generation metadata correctly and stop before exposing control tokens.

## Model Findings

| Model | Scope | Best measured behavior | Production finding |
| --- | --- | --- | --- |
| SmolLM2 360M Q8_0 | Engine diagnostics plus historical full suite | Pure Java reaches 44.86 tok/s; llama.cpp reaches 98.31 tok/s | Historical full-suite quality was poor. The current speedup does not qualify answer quality. |
| Qwen3 0.6B Q4_0 | Engine and prompt diagnostics | Pure Java reaches 60.34 tok/s, 125.7% of Ollama and 59.4% of llama.cpp | It is responsive, but the current study does not quality-qualify it for RAG. |
| MiniCPM5 1B Q4_K_M | Prompt diagnostics | Pure Java reaches 15.87 tok/s, but p95 TTFT is 7.17 seconds | Its prompt/template profile and pure-Java K-quant prefill remain unresolved. |
| Qwen3 1.7B Q8_0 | Full suite across plain Java, LangChain4j, and Spring AI | Pure-Java decode is near Ollama; Ollama meets the latency gate at concurrency one | Native exact/tolerant quality is 77.8%/88.9%. Pure Java is 66.7%/77.8%, leaks control tokens, and misses usable TTFT. |

Retrieval is perfect in every full run. The failed answerable case is therefore
generation, not search: Qwen3 1.7B says the context does not state whether a
telemedicine referral is required even though the retrieved passage explicitly
says no referral is required. This is exactly why token throughput cannot stand
in for RAG viability.

## Remote API Context

The controlled host does not have OpenAI or Anthropic API credentials, so this
study does not claim a same-prompt, same-time remote API benchmark. The current
external context is useful but not interchangeable with the local p95 results:

- Artificial Analysis reports median P50 measurements over the preceding 72
  hours. Its current OpenAI data shows low-latency non-reasoning paths around
  0.64 to 0.70 seconds to first answer token and 147 to 161 output tok/s.
- Its current Anthropic data reports a 0.87-second latency floor and 91 to 94
  output tok/s for Claude 4.5 Haiku, with Claude Sonnet 5 non-reasoning at
  1.42 seconds.
- Those measurements use different prompts, models, service load, percentiles,
  and quality levels. They are an external service baseline, not a score in
  this benchmark.

Sources:

- [Artificial Analysis OpenAI performance](https://artificialanalysis.ai/providers/openai)
- [Artificial Analysis Anthropic performance](https://artificialanalysis.ai/providers/anthropic)
- [OpenAI latency optimization](https://developers.openai.com/api/docs/guides/latency-optimization)
- [Anthropic prompt caching](https://platform.claude.com/docs/en/build-with-claude/prompt-caching)

The defensible broad comparison is that local Ollama's 0.69-second p95 TTFT is
in the same user-perceived range as the fastest observed cloud P50, and
llama.cpp's 1.29-second p95 remains plausible for interactive use. Pure Java's
5.35-second p95 TTFT is not competitive with either. Its 18.39 tok/s RAG decode
rate is also well below the cited cloud throughput range. Remote APIs add
network, variable service latency, data-governance constraints, and recurring
cost; local execution removes those dependencies but does not excuse a slow or
incorrect answer.

## Required Before A Production Claim

1. Qualify stronger 3B, 7B, and 8B instruction models. The sub-1B catalog is
   valuable for compatibility and constrained tasks, but it is not a credible
   default for this RAG contract.
2. Make GGUF chat-template, stop-token, and model-specific reasoning metadata
   first-class Models runtime behavior. No control token may reach application
   text, and generation must stop at the model's declared end markers.
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
7. Continue pure-Java prefill optimization without surrendering the decode
   gains. Qwen3 1.7B decode is now 71.8% of llama.cpp and 98.8% of Ollama on
   the RAG workload, while controlled prefill remains only 26.7% of either
   native engine. TTFT needs about a 4.1x reduction to match llama.cpp.
8. Repeat the matrix on Apple Silicon and a production-shaped x86 server. These
   results describe one controlled AVX2 CPU host, not every deployment target.
9. Add optional OpenAI and Anthropic controls using the identical rendered
   prompts, streamed first-token timestamps, and explicit model revisions.
   Keep cloud P50/P95 separate from local results and record network region.

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

The raw reports and exact controls for this refresh are committed under
[`benchmark-results/certified-20260724`](benchmark-results/certified-20260724/).
