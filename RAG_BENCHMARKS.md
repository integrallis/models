# Production RAG Benchmarks

Last updated: 2026-07-26

## Result

Sixteen exact artifacts representing fourteen distinct model identities now
clear the current absolute RAG SLOs, minimum model contribution, and same-host
Ollama comparison:

- **SmolLM2 360M Q8_0** is `PRODUCTION_READY` for the general guarded-RAG
  workload through plain Java, LangChain4j, and Spring AI.
- **Qwen3 0.6B Q4_0** is `PRODUCTION_READY` for the coding guarded-RAG workload
  through pure Java. Its profiled unsigned-pairwise Q4 kernel reaches 99.7% of
  Ollama and 46.7% of llama.cpp median decode throughput.
- **Qwen3 1.7B Q8_0** is `USABLE` for the general guarded-RAG workload. Its
  Models Rust/FFM path reaches 100.1% of Ollama and 70.6% of llama.cpp median
  decode throughput, with p95 end-to-end latency 1.15x and 1.26x respectively.
- **Qwen2.5-Coder 0.5B Q8_0** is `PRODUCTION_READY` for the coding workload
  through both pure Java and Rust/FFM. Rust/FFM reaches 132.8% of Ollama and
  67.0% of llama.cpp median decode throughput while preserving 15 correct model
  answers across 27 trials.
- **Qwen2.5-Coder 0.5B Q4_0** is `PRODUCTION_READY` for the coding workload
  through its measured hybrid profile. Rust/FFM prefill, the unsigned-pairwise
  Java Q4 decode kernel, and four row workers reach 80.7% of Ollama median
  decode throughput while preserving 12 correct model answers.
- **Qwen2.5-Coder 1.5B Q4_0** is `USABLE` for the coding workload through its
  measured hybrid profile. It reaches 80.2% of Ollama and 48.9% of llama.cpp
  median decode throughput while preserving 15 correct model answers.
- **Qwen2.5-Coder 1.5B Q8_0** is `USABLE` for the coding workload through
  Rust/FFM prefill. It reaches 89.9% of Ollama and 69.4% of llama.cpp median
  decode throughput while preserving 15 correct model answers.
- **EuroLLM 1.7B Q4_K_M** is `USABLE` for the multilingual guarded-RAG
  workload. Its marker-selected Rust/FFM profile reaches 109.8% of Ollama and
  68.8% of llama.cpp decode throughput.
- **Qwen2.5 0.5B Q4_K_M** is `PRODUCTION_READY` for general guarded RAG and
  reaches 146.6% of Ollama and 87.1% of llama.cpp decode throughput.
- **Qwen2.5 1.5B Q4_K_M** is `USABLE` for general guarded RAG and reaches
  108.7% of Ollama and 65.3% of llama.cpp decode throughput.
- **UmarTransit 1B Q4_K_M** is `USABLE` for transportation guarded RAG and
  reaches 115.2% of Ollama and 68.8% of llama.cpp decode throughput.
- **MiniCPM5 1B Q4_K_M** is `USABLE` for coding guarded RAG and reaches 131.2%
  of Ollama and 67.6% of llama.cpp decode throughput.
- **Llama 3.2 1B Q4_K_M** is `USABLE` for general guarded RAG and reaches
  111.4% of Ollama and 67.9% of llama.cpp decode throughput.
- **Gemma 3 1B Q4_K_M** is `PRODUCTION_READY` for general guarded RAG. Its
  marker-selected native decode reaches 137.3% of Ollama and 84.0% of
  llama.cpp, with 27 of 27 grounded answers correct.
- **Indian-Legal-Qwen2.5 3B Q4_K_M** is `USABLE` for legal guarded RAG. Its
  marker-selected Rust/FFM path reaches 84.3% of Ollama and 64.3% of llama.cpp
  decode throughput, with 12 of 12 retained model answers correct.
- **DeepSeek-R1-Distill-Qwen 1.5B Q4_K_M** is `USABLE` for general guarded
  RAG. Its marker-selected Rust/FFM path reaches 99.0% of Ollama and 64.2% of
  llama.cpp decode throughput, with 9 of 9 retained model answers correct.

These are guarded, workload-specific qualifications, not claims of unrestricted
question-answering quality. Every report preserves raw output, final grounded
output, decision type, artifact SHA-256, corpus SHA-256, workload ID, runtime
controls, and same-host comparator evidence. Exact reports are under
`benchmark-results/certified-20260724/rag/`,
`benchmark-results/certified-20260725/rag/`, and
`benchmark-results/certified-20260726/rag/`.

Quantization variants are retained as independently qualified artifacts but do
not increase the distinct-model launch count. The current launch count is
therefore fourteen, not sixteen.

The older Qwen3 framework and hosted-provider tables below remain useful
historical diagnostics, but they predate cross-request KV reuse and the current
grounding/model-contribution policies. The certified sections and executable
`CertifiedRagEvidenceTest` are authoritative when the results differ.

## SmolLM2 Q8_0 Launch Qualification

The current-schema SmolLM2 matrix uses an 8-vCPU EPYC Milan host, GraalVM Java
25.0.3, a fixed 1 GiB heap, the exact 386,404,992-byte artifact, one complete
warmup, and 27 measured requests per row. All engines receive the same 27
prompt hashes.

| Java path | Backend | Prefix cache | Tier | p95 TTFT | Median decode | p95 end to end | Grounded quality |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: |
| Plain Java | pure Java | no | USABLE | 1,621.9 ms | 44.12 tok/s | 3,007.3 ms | 100% |
| Plain Java | Rust/FFM | no | USABLE | 1,082.3 ms | 44.15 tok/s | 2,466.9 ms | 100% |
| Plain Java | pure Java | yes | PRODUCTION_READY | 756.6 ms | 43.69 tok/s | 2,145.5 ms | 100% |
| Plain Java | Rust/FFM | yes | PRODUCTION_READY | 552.8 ms | 43.92 tok/s | 1,935.0 ms | 100% |
| LangChain4j | Rust/FFM | yes | PRODUCTION_READY | 552.2 ms | 43.93 tok/s | 1,943.9 ms | 100% |
| Spring AI | Rust/FFM | yes | PRODUCTION_READY | 550.9 ms | 43.96 tok/s | 1,944.2 ms | 100% |
| Plain Java | Ollama | engine-managed | PRODUCTION_READY | 197.2 ms | 43.73 tok/s | 2,485.7 ms | 100% |
| Plain Java | llama.cpp | disabled | PRODUCTION_READY | 365.5 ms | 102.23 tok/s | 1,004.4 ms | 100% |

Models retained 3,153 of 5,064 input tokens as an exact cross-request KV
prefix. Rust/FFM p95 TTFT fell 48.9%, p95 end-to-end latency fell 21.6%, and
CPU time fell 40.7% without changing one raw output. Cached Rust/FFM reaches
91.4% of Ollama's median end-to-end performance and exceeds Ollama at p95 for
this response mix. It remains behind on TTFT and behind llama.cpp decode, so
the broader optimization work is not complete.

## Qwen Qualifications

All rows use the exact named artifact on the same 8-vCPU EPYC Milan host, one
complete warmup, 27 measured requests, 2,048-token context, 64-token output
cap, and deterministic sampling.

| Model and workload | Backend | Tier | p95 TTFT | Median decode | p95 end to end | Grounded quality |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Qwen3 0.6B Q4_0, coding | Models pure Java | PRODUCTION_READY | 555.2 ms | 51.02 tok/s | 1,416.1 ms | 100% |
| Qwen3 0.6B Q4_0, coding | Models Rust/FFM | PRODUCTION_READY | 568.0 ms | 50.56 tok/s | 1,379.3 ms | 100% |
| Qwen3 0.6B Q4_0, coding | Ollama 0.32.0 | PRODUCTION_READY | 336.2 ms | 51.16 tok/s | 1,381.0 ms | 100% |
| Qwen3 0.6B Q4_0, coding | llama.cpp b10012 | PRODUCTION_READY | 388.8 ms | 109.28 tok/s | 795.8 ms | 100% |
| Qwen3 1.7B Q8_0, general | Models pure Java | OFFLINE | 4,736.8 ms | 18.11 tok/s | 7,336.4 ms | 100% |
| Qwen3 1.7B Q8_0, general | Models Rust/FFM | USABLE | 1,747.6 ms | 17.99 tok/s | 4,785.1 ms | 100% |
| Qwen3 1.7B Q8_0, general | Ollama 0.32.0 | PRODUCTION_READY | 697.7 ms | 17.97 tok/s | 4,155.5 ms | 100% |
| Qwen3 1.7B Q8_0, general | llama.cpp b10012 | USABLE | 1,308.3 ms | 25.48 tok/s | 3,795.2 ms | 100% |
| Qwen2.5-Coder 0.5B Q8_0, coding | Models pure Java | PRODUCTION_READY | 617.0 ms | 51.63 tok/s | 1,173.2 ms | 100% |
| Qwen2.5-Coder 0.5B Q8_0, coding | Models Rust/FFM | PRODUCTION_READY | 434.6 ms | 53.01 tok/s | 980.4 ms | 100% |
| Qwen2.5-Coder 0.5B Q8_0, coding | Ollama 0.32.0 | PRODUCTION_READY | 301.6 ms | 39.93 tok/s | 1,164.1 ms | 100% |
| Qwen2.5-Coder 0.5B Q8_0, coding | llama.cpp b10012 | PRODUCTION_READY | 336.9 ms | 79.19 tok/s | 733.4 ms | 100% |
| Qwen2.5-Coder 0.5B Q4_0, coding | Models pure Java | PRODUCTION_READY | 982.9 ms | 38.95 tok/s | 2,509.9 ms | 100% |
| Qwen2.5-Coder 0.5B Q4_0, coding | Models Rust/FFM, 4 workers | PRODUCTION_READY | 390.3 ms | 39.50 tok/s | 1,941.2 ms | 100% |
| Qwen2.5-Coder 0.5B Q4_0, coding | Ollama 0.32.0 | PRODUCTION_READY | 283.4 ms | 48.95 tok/s | 1,592.8 ms | 100% |
| Qwen2.5-Coder 0.5B Q4_0, coding | llama.cpp b10012 | PRODUCTION_READY | 298.2 ms | 112.46 tok/s | 837.1 ms | 100% |
| Qwen2.5-Coder 1.5B Q4_0, coding | Models Rust/FFM, default Java Q4 | USABLE | 1,163.6 ms | 13.06 tok/s | 3,677.1 ms | 100% |
| Qwen2.5-Coder 1.5B Q4_0, coding | Models Rust/FFM, profiled Java Q4 | USABLE | 1,162.1 ms | 23.72 tok/s | 2,545.6 ms | 100% |
| Qwen2.5-Coder 1.5B Q4_0, coding | Ollama 0.32.0 | PRODUCTION_READY | 584.0 ms | 29.58 tok/s | 1,830.0 ms | 100% |
| Qwen2.5-Coder 1.5B Q4_0, coding | llama.cpp b10012 | PRODUCTION_READY | 943.3 ms | 48.46 tok/s | 1,659.5 ms | 100% |
| Qwen2.5-Coder 1.5B Q8_0, coding | Models pure Java | USABLE | 1,884.0 ms | 19.77 tok/s | 3,588.8 ms | 100% |
| Qwen2.5-Coder 1.5B Q8_0, coding | Models Rust/FFM | USABLE | 1,288.0 ms | 19.70 tok/s | 3,003.5 ms | 100% |
| Qwen2.5-Coder 1.5B Q8_0, coding | Ollama 0.32.0 | PRODUCTION_READY | 660.3 ms | 21.92 tok/s | 2,325.8 ms | 100% |
| Qwen2.5-Coder 1.5B Q8_0, coding | llama.cpp b10012 | USABLE | 1,083.3 ms | 28.37 tok/s | 2,317.3 ms | 100% |

Each Qwen3 artifact contributes 12 correct model answers, uses 12 extractive
fallbacks, and abstains for 3 unsupported requests. Qwen2.5-Coder preserves 15
correct model answers with deterministically derived source IDs, uses 9
conservative fallbacks for incomplete answers, and abstains for 3 unsupported
requests. The latter clears both relative gates: its p95 end-to-end latency is
0.84x Ollama and 1.34x llama.cpp. Pure Java independently clears both gates at
129.3% of Ollama and 65.2% of llama.cpp decode; Rust/FFM remains the recommended
profile because it improves prefill by 45.5%, p95 TTFT by 29.6%, p95
end-to-end latency by 16.4%, and measured CPU by 20.8% on the same workload.

The Qwen2.5-Coder Q4_0 artifact contributes 12 correct model answers, uses 12
extractive fallbacks, and abstains for 3 unsupported requests. Its selected
hybrid reaches 80.7% of Ollama and 35.1% of llama.cpp decode, with p95
end-to-end ratios of 1.22 and 2.32. Only Ollama clears the relative policy, as
required for qualification. The pure-Java control misses both Ollama relative
limits at 79.6% decode and 1.58x p95 end-to-end latency. Rust prefill plus four
row workers improves prefill throughput 2.59x, p95 TTFT 60.3%, p95 end-to-end
22.7%, and measured CPU 49.3% relative to that control.

The 1.5B Q4_0 artifact contributes 15 correct model answers, uses 9 extractive
fallbacks, and abstains for 3 unsupported requests. Its selected hybrid reaches
80.2% of Ollama and 48.9% of llama.cpp decode, with p95 end-to-end ratios of
1.39 and 1.53; both comparators clear the relative policy. It is classified
`USABLE` because p95 TTFT is 1.16 seconds, above the stricter 1-second
`PRODUCTION_READY` target but below the 2-second interactive ceiling. Relative
to the unprofiled Rust/FFM path, the exact unsigned-Q4/four-worker profile
improves decode 1.82x, reduces p95 end-to-end latency 30.8%, and reduces
measured CPU 50.3%. All 27 raw generations are byte-identical.

The 1.5B Q8_0 artifact also contributes 15 correct model answers, uses 9
extractive fallbacks, and abstains for 3 unsupported requests. Rust/FFM reaches
89.9% of Ollama and 69.4% of llama.cpp decode, with p95 end-to-end ratios of
1.29 against both controls. Pure Java independently remains `USABLE` but
narrowly misses Ollama's relative end-to-end ceiling at 1.54x. Rust prefill
improves prefill throughput 1.46x, p95 TTFT 31.6%, p95 end-to-end latency
16.3%, and measured CPU 20.2%. Pure Java and Rust/FFM both produce 27 correct
grounded answers, but their raw generations and contribution decisions differ;
this is semantic parity, not byte-identical output.

## Gemma 3 1B Qualification

The Gemma 3 matrix uses the exact 806,058,496-byte Q4_K_M artifact with SHA-256
`12bf0fff8815d5f73a3c9b586bd8fee8e7b248c935de70dec367679873d0f29d`.
All rows ran sequentially on the same 8-vCPU EPYC-Milan host with one complete
warmup and 27 measured requests.

| Backend | Tier | p95 TTFT | Median decode | p95 end to end | Grounded quality |
| --- | --- | ---: | ---: | ---: | ---: |
| Models Rust/FFM, Java decode | PRODUCTION_READY | 976.7 ms | 14.63 tok/s | 3,134.9 ms | 100% |
| Models Rust/FFM, marker profile | PRODUCTION_READY | 954.3 ms | 41.06 tok/s | 1,619.7 ms | 100% |
| Ollama 0.32.0 | USABLE | 1,130.8 ms | 29.90 tok/s | 2,101.0 ms | 100% |
| llama.cpp b10012 | PRODUCTION_READY | 940.3 ms | 48.85 tok/s | 1,524.0 ms | 100% |

The marker profile activates only native quantized decode and eight native
kernel workers. It improves decode throughput 2.806x and lowers p95 end-to-end
latency 48.3% over Java decode while preserving all 27 outputs exactly. A
separately measured 64-token prefill and batched-attention candidate was
rejected because it was slower. The raw reports and executable evidence test
are under
`benchmark-results/certified-20260726/rag/gemma-3-1b-q4_k_m/`.

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

The committed `general` and `coding` corpora each contain 12 short documents,
eight answerable questions, and one deliberately unanswerable question. Lucene
BM25 and Python BM25S use top-1 retrieval and agree on all retrieved documents
and rendered prompt hashes. The evaluator checks required facts, source IDs,
unsupported citations, and exact `INSUFFICIENT_CONTEXT` abstention without an
LLM judge. Temperature is zero and top-k is one. The historical Qwen3 1.7B
matrix disables prompt caching and sends the same ChatML-no-think prompt bytes.
The current qualification matrices include longest-common-prefix KV reuse and
retain cache read/write token counts in every report.

The strict abstention contract intentionally counts `INSUFFICIENT_CONTEXT.` as
instruction noncompliance. Normalizing that trailing period raises each native
path to eight of nine semantically acceptable cases, or 88.9%, but it still
misses the gate because the telemedicine answer contradicts retrieved context.
The same normalization would raise pure Java only to seven of nine, or 77.8%;
its extra citation and control-token failures remain. Production applications
should use schema-constrained output rather than punctuation-sensitive control
text.

## Historical Qwen3 1.7B Baseline

This section records the pre-prefix-cache diagnostic matrix. It explains the
bottlenecks that motivated the retained runtime changes but is superseded for
qualification purposes by the current tables above.

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
| Measured artifact | Qwen3 1.7B Q8_0, 1,834,426,016 bytes |
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
| SmolLM2 360M Q8_0 | Current guarded RAG suite across plain Java, LangChain4j, and Spring AI | Rust/FFM reaches 43.92 tok/s and about 0.55 seconds p95 TTFT with prefix reuse | `PRODUCTION_READY` for the committed trusted-retrieval policy; raw model-only correctness is 66.7%, so unguarded QA is not qualified. |
| Qwen3 0.6B Q4_0 | Current coding guarded-RAG suite | Profiled pure Java reaches 51.02 tok/s, 99.7% of Ollama and 46.7% of llama.cpp | `PRODUCTION_READY`; 12 of 27 accepted answers retain model text and all 12 are correct. |
| MiniCPM5 1B Q4_K_M | Prompt diagnostics | Pure Java reaches 15.87 tok/s, but p95 TTFT is 7.17 seconds | Its prompt/template profile and pure-Java K-quant prefill remain unresolved. |
| Qwen3 1.7B Q8_0 | Current general guarded-RAG suite | Rust/FFM reaches 17.99 tok/s, 100.1% of Ollama and 70.6% of llama.cpp, at 1.15x Ollama p95 end-to-end latency | `USABLE`; 12 of 27 accepted answers come from the model and all 12 are correct. |
| Qwen2.5-Coder 0.5B Q8_0 | Current coding guarded-RAG suite | Pure Java and Rust/FFM both qualify; Rust/FFM reaches 53.01 tok/s, 132.8% of Ollama and 67.0% of llama.cpp, with lower p95 end-to-end latency than Ollama | `PRODUCTION_READY`; 15 of 27 accepted answers retain model text and all 15 are correct. |
| Qwen2.5-Coder 0.5B Q4_0 | Current coding guarded-RAG suite | Profiled Rust/FFM reaches 39.50 tok/s, 80.7% of Ollama and 35.1% of llama.cpp, at 1.22x Ollama p95 end-to-end latency | `PRODUCTION_READY`; 12 of 27 accepted answers retain model text and all 12 are correct. Pure Java remains a nonqualifying control. |
| Qwen2.5-Coder 1.5B Q4_0 | Current coding guarded-RAG suite | Profiled Rust/FFM reaches 23.72 tok/s, 80.2% of Ollama and 48.9% of llama.cpp, at 1.39x Ollama p95 end-to-end latency | `USABLE`; 15 of 27 accepted answers retain model text and all 15 are correct. |
| Qwen2.5-Coder 1.5B Q8_0 | Current coding guarded-RAG suite | Rust/FFM reaches 19.70 tok/s, 89.9% of Ollama and 69.4% of llama.cpp, at 1.29x Ollama p95 end-to-end latency | `USABLE`; 15 of 27 accepted answers retain model text and all 15 are correct. Pure Java narrowly misses the relative end-to-end gate. |

Retrieval is perfect in every full run. The historical failed answerable case
was therefore generation, not search: Qwen3 1.7B said the context did not state
whether a telemedicine referral was required even though the retrieved passage
explicitly said no referral was required. The current grounding policy
conservatively replaces that incomplete output. This is exactly why token
throughput cannot stand in for RAG viability.

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

## Required Before A Broad Production Claim

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
7. Continue prefill and decode optimization without surrendering exact output
   behavior. Prefix reuse moved SmolLM2 Q8_0 into the production latency gate,
   but cached Rust/FFM still reaches only 66.1% of llama.cpp p95 TTFT
   performance and 43.0% of its median decode throughput.
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
