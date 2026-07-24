# Controlled Benchmark Evidence: 2026-07-24

This directory contains the raw evidence for the controlled inference and RAG
results published in `INFERENCE_BENCHMARKS.md` and `RAG_BENCHMARKS.md`.

## Scope

- Host: Ubuntu Linux 6.8, AMD EPYC Milan, 8 logical CPUs, 30.6 GiB RAM
- Models: `10949c7c4273a2769365cdca80091ba91c7f2015` for local
  engine/RAG rows; `d2f5072737d2d91ae03b399467a0d782618972c1` for hosted
  provider rows
- Vectors: `fde9858901624d1661a1cf51195d2c59737bcf87`
- ModelJars: `b6575a10a600bd1838ea9033b32c4acecce2fd25`
- llama.cpp: build 10012, commit `c71854292`
- Ollama: `0.32.0`
- Java: GraalVM Community Java 25.0.3 for profiled pure-Java runs
- Workload: CPU only, one request at a time, eight inference threads

The `inference/` reports contain two warmups and ten measured 64-token
generations per backend. Each backend receives the same GGUF bytes and
deterministic nonce-prefixed prompt sequence. The comparison reports verify the
artifact, prompt, controls, host, and trial count before calculating ratios.

The `rag/` reports contain a 12-document synthetic corpus and nine cases. Plain
Java uses one warmup and three measured iterations, for 27 requests per
backend. LangChain4j and Spring AI use one warmup and one measured iteration,
for nine requests per adapter/backend pair. All reports use the same artifact,
corpus, case prompts, retrieval settings, and generation controls.

The `rag/providers/` reports use the same controlled host, retrieval corpus,
canonical prompt text, case order, one warmup, and three measured iterations.
They call the providers directly from Java 25 and measure the first non-empty
streamed text delta. Schema v4 records provider-reported input, cache-read,
cache-write, and output tokens; the exact pricing snapshot; measured API cost;
and the provider-specific request controls. No API keys or key values are
stored in the reports.

Provider report SHA-256:

| Report | SHA-256 |
| --- | --- |
| OpenAI GPT-5.4 nano | `70e5cea05247b3c3ed6a51f2a1a3a67ca2341ae617943c75731b4d2443472e6f` |
| Anthropic Claude Haiku 4.5 | `9591af46a7dacd4826b0b9da19f86e29415723cde4e1317ab5f2cc9549dd46b0` |
| DeepSeek V4 Flash | `292276cd8b6cee61b9bdbe9d060b6b71b5b8d0ffab5bb16a1a496050b0d71f76` |

## Candidate Status

Qwen3 1.7B Q8_0 is labeled `row-candidate`. Its row-accumulated Q8 execution
plan matched the untuned Graal reference output SHA-256 in all five
corresponding screening trials. The faster float-lane candidate matched only
three of five and was rejected for this comparison. The row candidate was then
used for the ten-trial inference report and the RAG matrix.

This evidence does not certify production concurrency, other hardware, or
model quality beyond the committed deterministic workload. Ollama RSS covers
the observed service process and can understate the separate runner's memory.
