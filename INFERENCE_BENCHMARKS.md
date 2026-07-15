# Controlled inference benchmarks

This study compares the pure-Java backend with Ollama and llama.cpp using the
same GGUF bytes on one CPU-only host. It is an initial single-request latency
study, not a concurrency, cost, model-quality, or production-capacity claim.

## Results

The values below are from ten measured generations after two warmups. TTFT and
TPOT are p95; prefill and decode rates are p50. Lower is better for latency and
higher is better for token rates. `Match` is the percentage of complete greedy
outputs whose SHA-256 equals llama.cpp for the same per-trial prompt.

| Model | Backend | Load ms | TTFT ms | TPOT ms | Prefill tok/s | Decode tok/s | Peak RSS GiB | vs llama.cpp | Match |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3 0.6B Q4_0 | pure Java | 177.5 | 8,682.2 | 60.3 | 17.85 | 16.67 | 0.84 | 15.7% | 30% |
| Qwen3 0.6B Q4_0 | llama.cpp | 1,119.0 | 383.5 | 10.4 | 459.23 | 106.21 | 1.19 | 100% | 100% |
| Qwen3 0.6B Q4_0 | Ollama | 1,475.2 | 510.7 | 25.9 | 459.50 | 41.03 | 1.01 | 38.6% | 100% |
| SmolLM2 360M Q8_0 | pure Java | 125.6 | 10,553.0 | 70.8 | 15.11 | 14.38 | 0.77 | 14.3% | 50% |
| SmolLM2 360M Q8_0 | llama.cpp | 585.0 | 288.3 | 10.8 | 581.07 | 100.57 | 0.58 | 100% | 100% |
| SmolLM2 360M Q8_0 | Ollama | 848.8 | 344.2 | 25.3 | 585.04 | 44.39 | 0.62 | 44.1% | 100% |
| MiniCPM5 1B Q4_K_M | pure Java | 171.7 | 24,111.2 | 164.6 | 6.32 | 6.14 | 0.96 | 8.5% | 10% |
| MiniCPM5 1B Q4_K_M | llama.cpp | 1,121.0 | 515.4 | 14.5 | 304.94 | 71.97 | 1.12 | 100% | 100% |
| MiniCPM5 1B Q4_K_M | Ollama | 1,449.9 | 675.8 | 28.9 | 307.63 | 38.82 | 0.86 | 53.9% | 100% |

The current conclusion is narrow: in-process model mapping is fast and memory
use is bounded, but pure-Java text generation is not yet interactive on this
host. Prompt processing is the largest gap because the forward path evaluates
prompt tokens serially instead of using a batched prefill/GEMM path. Decode is
also 6.4x to 11.7x slower than llama.cpp for these artifacts. Output parity is
prompt-sensitive and must improve before performance alone can qualify a model.

## Optimization evidence

The benchmark found scalar work in the Q8_0 and Q4_K fused matvec kernels. Both
were fixed test-first in `vectors-core` and merged as vectors PRs 8 and 9.

| Change | Before | After | Improvement |
| --- | ---: | ---: | ---: |
| Q8_0 kernel, JMH 1024x2048 | 2.460 ms/op | 1.071 ms/op | 2.30x |
| SmolLM2 pure-Java TTFT | about 29.95 s | 10.55 s | about 2.84x |
| SmolLM2 pure-Java decode | 5.10 tok/s | 14.38 tok/s | 2.82x |
| Q4_K kernel, JMH 1024x2048 | 0.683 ms/op | 0.453 ms/op | 1.51x |
| MiniCPM5 pure-Java TTFT | 34.21 s | 24.11 s | 1.42x |
| MiniCPM5 pure-Java decode | 4.34 tok/s | 6.14 tok/s | 1.41x |

The MiniCPM baseline also exposed a `llama-bpe` pre-tokenizer incompatibility.
The comparison rejected the evidence because Java reported a different input
token series. Exact llama.cpp token fixtures were added before correcting the
pre-tokenizer mapping; only the corrected rerun appears in the results table.

## Latency policy

`BenchmarkPolicy` assigns a tier only when every measured trial succeeds:

| Tier | p95 TTFT | p95 TPOT | Approximate decode floor |
| --- | ---: | ---: | ---: |
| Interactive | <= 500 ms | <= 40 ms | >= 25 tok/s |
| Responsive | <= 1,000 ms | <= 100 ms | >= 10 tok/s |
| Usable | <= 2,000 ms | <= 200 ms | >= 5 tok/s |
| Offline | above either usable limit | above either usable limit | no interactive claim |

These are ModelJars project policy, not an industry standard. They are informed
by published serving workloads, then made stricter at the top tier for local,
single-user inference. DistServe evaluates application-specific SLOs ranging
from 125 ms to 15 s TTFT and 100 ms to 200 ms TPOT, and requires at least 90%
attainment. NVIDIA's GenAI-Perf defines TTFT and inter-token latency as primary
user-perceived LLM metrics. The benchmark uses the same TPOT formula:
`(request latency - TTFT) / (output tokens - 1)`.

- [DistServe, OSDI 2024](https://www.usenix.org/system/files/osdi24-zhong-yinmin.pdf)
- [NVIDIA GenAI-Perf metrics](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_benchmark/genai-perf-README.html#metrics)

Relative decode performance is reported separately: at least 80% of llama.cpp
is `COMPETITIVE`, at least 50% is `VIABLE`, and lower is
`NEEDS_OPTIMIZATION`. A latency tier never implies output correctness or model
quality.

## Controls

All three backends receive the same prompt bytes and exact GGUF file. The runner
rejects reports unless model SHA-256, file size, hardware/JDK identity,
generation controls, input and output token counts, and measured-trial counts
match. It recalculates summaries from raw trials before producing a comparison.

- Host: Hetzner dedicated-vCPU VM, AMD EPYC Milan, 8 vCPU, 30.6 GiB RAM
- OS: Ubuntu 24.04, Linux 6.8.0-124, no swap
- JVM: Eclipse Temurin 25.0.3, `jdk.incubator.vector`, 256-bit vector cap
- Native references: llama.cpp b10012 (`c71854292`), Ollama 0.32.0
- Workload: one request at a time, 8 backend threads, 2 warmups, 10 trials
- Prompt: fixed 723-byte production-review prompt plus a deterministic,
  per-trial 16-character SHA-256 nonce prefix
- Generation: 64 output tokens, 2,048 context, temperature 0, top-k 1, top-p 1,
  repetition penalty 1, seed 42, raw completion mode
- Cache control: Linux page cache dropped between backend runs; nonce prefixes
  prevent backend prompt/KV cache reuse while preserving identical inputs

Artifact identities:

| Model | Bytes | SHA-256 |
| --- | ---: | --- |
| Qwen3 0.6B Q4_0 | 428,970,080 | `da2572f16c06133561ce56accaa822216f2391ef4d37fba427801cd6736417d4` |
| SmolLM2 360M Q8_0 | 386,404,992 | `48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201` |
| MiniCPM5 1B Q4_K_M | 688,065,920 | `81b64d05a23b17b34c475f42b3e72fbde62d4b92cc34541f7a8031d0752deafa` |

Ten trials are sufficient for the repository's initial diagnostic gate but
provide limited tail-latency confidence. Release or hardware qualification
should use longer runs and workload-appropriate concurrency. Load times are
also directional: pure Java measures in-process parse/map time, llama.cpp
measures server readiness, and Ollama reports its request load phase.

## Reproduce

Bootstrap a clean Linux host with pinned, checksum-verified native tools:

```bash
scripts/bootstrap-inference-bench-host.sh
```

Run all three backends and produce raw JSON plus Markdown comparison evidence:

```bash
BENCH_DROP_CACHES=1 scripts/run-controlled-inference-benchmarks.sh \
  ~/.jvllm/models/Qwen3-0.6B-Q4_0.gguf \
  qwen3-0.6b-q4_0
```

The script records exact `models` and `vectors` commits in the pure-Java backend
version. Do not publish comparisons from dirty checkouts or mixed hosts.
