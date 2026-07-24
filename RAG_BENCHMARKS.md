# Production RAG Benchmarks

Last updated: 2026-07-24

## Result

No measured **local Qwen3 1.7B path** is ready for a production RAG claim.
The hosted control changes the broader developer-facing result: OpenAI
GPT-5.4 nano passes every latency and strict-quality gate on this workload.
Anthropic Claude Haiku 4.5 and DeepSeek V4 Flash pass the latency gates, but
their deterministic exact-phrase score is 88.9%.

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

The hosted comparison runs the same Lucene retrieval, canonical RAG text, case
order, output cap, and evaluator from the same controlled VPS. The provider
applies its private chat template and tokenizer, so this is a user-experience
comparison rather than a same-GGUF engine benchmark.

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

## Hosted API Comparison

Each hosted row covers one warmup over all nine cases followed by three
measured iterations, or 27 measured requests. TTFT and end-to-end latency are
p95; decode is the median. Cost is calculated from provider-reported token
usage and the pricing snapshot embedded in each schema-v4 report.

| Provider model | p95 TTFT | Median decode | p95 end to end | Strict quality | Audited semantic quality | Measured API cost / 1K |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenAI GPT-5.4 nano, `2026-03-17` | 440.9 ms | 141.29 tok/s | 727.9 ms | 100.0% | 100.0% | $0.0724 |
| Anthropic Claude Haiku 4.5, `20251001` | 1,086.1 ms | 139.30 tok/s | 1,581.5 ms | 88.9% | 100.0% | $0.3666 |
| DeepSeek V4 Flash, non-thinking | 758.6 ms | 102.43 tok/s | 1,118.8 ms | 88.9% | 100.0% | $0.0134 measured |

OpenAI is the only row that passes the automated `PRODUCTION_READY` policy.
The Haiku and DeepSeek failures are three repetitions of one semantically
correct telemedicine answer. Haiku says “does not require a referral” and
DeepSeek says “No referral is needed”; the deliberately simple evaluator
requires the literal substring “do not require a referral.” The raw answers
are retained, and the table keeps the strict score instead of changing the
evaluator after seeing results. “Audited semantic quality” is a transparent
human review of those three false negatives, not an LLM-judge score.

DeepSeek automatically served 3,456 of 4,440 measured input tokens from its
context cache. Its displayed $0.0134 per 1,000 requests is therefore the
observed repeated-workload cost; applying the pinned cache-miss rate to the
same average input/output counts gives approximately $0.031 per 1,000 unique
requests. OpenAI and Anthropic reported no cache-read tokens.

These API costs exclude client compute, retrieval, network transit, warmups,
retries, storage, and provider price changes. Local execution has no
per-request provider charge, but it still consumes owned or rented hardware.
More importantly, the local path keeps retrieved documents, questions, and
generated text in the application process and works offline. Hosted requests
send that material across the network to a third party and require credentials
and provider availability. This is a deployment and data-governance
distinction, not a claim that a particular provider is insecure.

The measured result developers care about is direct: GPT-5.4 nano and DeepSeek
V4 Flash return this short RAG workload faster than the tested local Qwen3 1.7B
paths. Ollama remains in the same interactive TTFT range without data egress,
while pure Java is not yet competitive on long-prompt TTFT. Models' local value
proposition is privacy, offline operation, predictable marginal cost, and
in-process deployment; it cannot presently be marketed as faster than these
hosted controls.

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
9. Repeat the hosted matrix from multiple regions and at concurrency 1, 2, 4,
   and 8. Preserve provider-reported cache usage and exact model revisions so
   queueing, routing, and cache effects remain visible.

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
