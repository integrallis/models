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
| Qwen3 0.6B Q4_0 | pure Java | 194.1 | 3,145.6 | 39.0 | 49.54 | 25.86 | 1.04 | 25.0% | 20% |
| Qwen3 0.6B Q4_0 | llama.cpp | 1,118.0 | 364.6 | 10.5 | 457.13 | 103.51 | 1.19 | 100% | 100% |
| Qwen3 0.6B Q4_0 | Ollama | 1,475.7 | 515.7 | 25.5 | 460.93 | 47.39 | 1.01 | 45.8% | 100% |
| SmolLM2 360M Q8_0 | pure Java | 131.2 | 2,643.7 | 47.4 | 60.16 | 21.23 | 0.69 | 20.6% | 50% |
| SmolLM2 360M Q8_0 | llama.cpp | 471.0 | 298.7 | 10.6 | 575.14 | 103.13 | 0.58 | 100% | 100% |
| SmolLM2 360M Q8_0 | Ollama | 830.9 | 336.5 | 26.2 | 587.08 | 43.18 | 0.63 | 41.9% | 100% |
| MiniCPM5 1B Q4_K_M | pure Java | 172.9 | 8,527.9 | 76.3 | 17.82 | 13.19 | 0.88 | 18.4% | 10% |
| MiniCPM5 1B Q4_K_M | llama.cpp | 1,244.0 | 513.1 | 14.3 | 306.88 | 71.74 | 1.12 | 100% | 100% |
| MiniCPM5 1B Q4_K_M | Ollama | 1,450.9 | 630.0 | 29.9 | 307.44 | 37.42 | 0.87 | 52.2% | 100% |

Across these three formats, the arithmetic mean of the per-model pure-Java
decode ratios is 21.3% of llama.cpp and 46.3% of Ollama. Dividing summed
throughput gives 21.7% and 47.1%. Qwen is currently strongest at 25.0% of
llama.cpp and 54.6% of Ollama; MiniCPM remains weakest at 18.4% and 35.2%.

Relative to the preceding matrix, summed pure-Java decode increased 6.6%:
MiniCPM improved 36.4%, Qwen improved 1.6%, and SmolLM2 changed -0.9%. Summed
llama.cpp throughput changed +0.3%, while Ollama changed -9.4%. The arithmetic
mean therefore moved +1.6 percentage points against llama.cpp and +7.5 points
against Ollama, but most of the latter movement is native-run variance rather
than a Java speedup.

Prompt processing moved further. Summed pure-Java prefill increased 46.3%, its
ratio to llama.cpp rose from 6.5% to 9.5%, and mean p95 TTFT fell 39.0%.
SmolLM2's retained Q8_0 batching accounts for the largest gain. In-process
model mapping remains fast and memory use is bounded, but all three complete
pure-Java requests remain `OFFLINE` on this long-prompt workload. Decode is
still 4.0x to 5.4x slower than llama.cpp and 1.8x to 2.8x slower than Ollama.
Output parity is prompt-sensitive and must improve before performance alone can
qualify a model.

### Controlled JVM compiler matrix

The unchanged backend was then run under GraalVM Community
`25.2.4-dev-20260717_0119` on the same idle host, model bytes, prompt, context,
thread count, warmups, and ten-trial protocol. This is a JVM deployment choice,
not a kernel rewrite:

| Model | HotSpot decode | Graal decode | Decode change | HotSpot p95 TTFT | Graal p95 TTFT | RSS change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3 0.6B Q4_0 | 25.86 tok/s | 19.65 tok/s | -24.0% | 3,145.6 ms | 5,900.3 ms | +58.5% |
| SmolLM2 360M Q8_0 | 21.23 tok/s | 44.70 tok/s | +110.6% | 2,643.7 ms | 1,299.1 ms | +13.8% |
| MiniCPM5 1B Q4_K_M | 13.19 tok/s | 15.34 tok/s | +16.3% | 8,527.9 ms | 7,377.2 ms | +8.4% |

Every corresponding Java output SHA-256 matched across the two compilers. A
separate Qwen control with six warmups still decoded at 19.75 tok/s, so its
regression is not explained by the standard two-warmup protocol.

Choosing the best measured compiler per model raises summed pure-Java decode to
30.9% of llama.cpp and 67.1% of Ollama, versus the HotSpot-only 21.7% and 47.1%.
Individually, the selected paths reach 25.0%/54.6% for Qwen, 43.4%/103.5% for
SmolLM2, and 21.4%/41.0% for MiniCPM against llama.cpp/Ollama. This is evidence
for model-, quantization-, topology-, and host-specific planning; Graal is not a
global default. The exact-SHA-bound recommendations and evidence are published
through ModelJars performance profile schema v1.

### Current staged Q8_0 prefill checkpoint

The later SmolLM2 prefill path combines the model-scoped Graal deployment, batch-32 projection,
seven-stage attention/FFN schedule, and Vectors weight-conversion reuse. The final incremental gate
compared only Vectors `7fb6fa5` and `25aa094`; Models `6c306f0`, model bytes, prompt strategy,
GraalVM Java 25.0.3, fixed 1 GiB heap, eight workers, and every model setting remained constant.
Five warmups and five one-token measurements in each of six counterbalanced process pairs produced:

| Metric | Before weight reuse | Current pure Java | Change |
| --- | ---: | ---: | ---: |
| p50 TTFT | 1,003.739 ms | 925.665 ms | -7.78% |
| p95 TTFT | 1,031.209 ms | 935.595 ms | -9.27% |
| p50 prefill | 158.580 tok/s | 170.463 tok/s | +7.49% |
| p50 process CPU | 7,545 ms | 6,960 ms | -7.75% |

Every one of the six pair medians improved and all 30 corresponding output hashes matched. Using
the recorded same-host native controls above, current SmolLM2 prefill is 29.64% of llama.cpp and
29.04% of Ollama; median TTFT is 3.10x llama.cpp and 2.75x Ollama. The native engines were not
rerun for this incremental gate, so these are pinned comparisons rather than a fresh cross-engine
study. Fixed-position decode remained neutral at 42.46 versus 42.23 tok/s median with identical
checksums and zero GC.

A subsequent layout gate kept Vectors `d295f32`, Models `29b5169`, model bytes, and the complete
runtime profile constant, changing only
`models.purejava.blockMajorQ8Activations=false/true`. The retained representation keeps canonical
batch-major Q8 bytes and one compact block-major byte copy so matrix rows consume all activation
rows for one block contiguously. It does not widen activations to integers.

| Metric | Packed activation bytes | Block-major activation bytes | Change |
| --- | ---: | ---: | ---: |
| p50 TTFT | 939.347 ms | 925.622 ms | -1.46%; -1.69% paired-process median |
| p95 TTFT | 965.011 ms | 943.771 ms | -2.20% |
| p50 prefill | 167.890 tok/s | 170.006 tok/s | +1.26% |
| p50 process CPU | 7,040 ms | 6,920 ms | -1.70% |
| median process RSS | 1,020,530,688 B | 1,018,851,328 B | no regression observed |

Five of six process medians favored the candidate, all 60 trials succeeded, and every corresponding
input count, output count, and output SHA-256 matched. A 1024x2048 batch-32 kernel gate improved
11.5% on local AVX2 and 4.6% on controlled EPYC/GraalVM. A pre-widened `int[]` alternative was
rejected despite an 18.8% local batch-32 improvement because it was neutral on controlled EPYC and
its four-times-larger payload did not fix the cache-locality boundary.

A follow-up gate addressed the serial FFN preparation stage identified by JFR after block-major
activation retention. Vectors candidate `4dcf935`, merged as `523f3aa`, added exact disjoint Q8
activation batch-range quantization, while Models `4887cde` partitions residual addition, RMSNorm,
and quantization by active batch row only for eligible all-Q8 staged layers. ModelJars remained
fixed at `6db5163`.
The model, prompt, GraalVM Java 25 runtime, fixed 1 GiB heap, eight workers, batch 32, and every
other model setting remained constant across six counterbalanced fresh-process pairs:

| Metric | Serial FFN preparation | Batch-row preparation | Change |
| --- | ---: | ---: | ---: |
| p50 TTFT | 913.102 ms | 899.064 ms | -1.54%; -1.68% paired-process median |
| p95 TTFT | 930.788 ms | 914.340 ms | -1.77% |
| p50 prefill | 172.543 tok/s | 175.592 tok/s | +1.77% |
| p50 process CPU | 6,860 ms | 6,840 ms | -0.29% |
| median process RSS | 1,015,816,192 B | 1,013,358,592 B | -0.24% |

All six process-pair medians and 28 of 30 corresponding trials favored the candidate. All 60
trials succeeded with exact corresponding input counts, output counts, and output SHA-256 values.
Maximum observed RSS moved by +0.76%, which is not a material memory regression. The option is
default-off and requires all-Q8 gate/up projections, the staged layer plan, block-major
activations, and at least two row partitions.

The next gate replaced the block-by-block scattered output updates with one row-local batch
accumulator. Vectors keeps this as an explicit kernel strategy because local Java 25 HotSpot/C2 was
neutral, while the controlled EPYC/GraalVM batch-32 JMH improved from 20.571 to 4.335 ms/op
(-78.9%). Models plan schema `pure-java-v15` requires an exact recommendation, Graal JVMCI, and
eligible staged block-major Q8 topology before selecting it.

| Metric | Scattered output updates | Row-local accumulator | Change |
| --- | ---: | ---: | ---: |
| p50 TTFT | 898.794 ms | 878.801 ms | -2.22% |
| p95 TTFT | 919.981 ms | 886.209 ms | -3.67% |
| p50 prefill | 175.219 tok/s | 179.479 tok/s | +2.43% |
| p50 process CPU | 6,840 ms | 6,680 ms | -2.34% |
| median process RSS | 1,009,147,904 B | 1,015,336,960 B | +0.61% |

All 30 corresponding TTFT, prefill, and CPU trials favored the row-local strategy, as did all six
process-pair medians. Token counts and output hashes remained exact, and maximum RSS changed by
+0.41%. Against the pinned same-host native controls, the resulting 179.479 tok/s is approximately
31.2% of llama.cpp prefill and 30.6% of Ollama prefill; 878.801 ms TTFT is approximately 2.94x
llama.cpp and 2.61x Ollama. These native values were not rerun for this incremental gate.

### Controlled mixed K-quant projection result

MiniCPM5 stores query and key projections as Q4_K and the narrower value
projection as Q6_K. The retained path quantizes the shared attention activation
once and publishes all three row ranges through one persistent-worker dispatch.
The immutable execution plan enables this only for the exact
Q4_K/Q4_K/Q6_K topology; `models.purejava.mixedKProjections=false` is the
explicit control.

Two warmups and ten 64-token trials were run in each order on the controlled
eight-vCPU AMD EPYC Milan host. The table combines control-first and
candidate-first runs, so each cell covers 20 measured generations. Decode and
TTFT values are arithmetic means over those trials.

| JVM | Independent decode | Mixed decode | Decode change | Independent TTFT | Mixed TTFT | TTFT change |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| OpenJDK 25.0.3 HotSpot | 13.05 tok/s | 13.37 tok/s | +2.52% | 8,425.9 ms | 8,427.2 ms | +0.02% |
| GraalVM CE 25.2.4-dev | 15.28 tok/s | 15.47 tok/s | +1.31% | 7,320.3 ms | 7,316.1 ms | -0.06% |

All 40 paired output SHA-256 values matched. Both process orders improved on
each JVM; HotSpot's 20 paired decode deltas ranged from +1.32% to +4.26%.
Graal's combined result remained positive despite one -0.08% trial-level delta.
The retained Vectors implementation deliberately leaves the existing standalone
Q4_K and Q6_K hot-loop source shapes unchanged. An initial helper extraction
coincided with a severe Graal slowdown, but an untouched baseline reproduced the
slowdown while the host was contended, so no compiler regression is attributed
to that refactor. Preserving the established source shape removes that risk from
the retained change.

The evidence is pinned to Models `00de3059`, Vectors `e957a50e`, model SHA-256
`81b64d05a23b17b34c475f42b3e72fbde62d4b92cc34541f7a8031d0752deafa`,
2,048-token context capacity, eight inference threads, and the same deterministic
nonce-prefix prompt strategy used by the compiler matrix. Reports are retained
under `/opt/inference-mixed-20260718/results-{hotspot,graal}-final*` on the
controlled host.

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

### Controlled Q8 grouped-projection result

A later test-first optimization groups SmolLM2's equal-format Q8_0 gate/up
projections so they share one activation quantization and one row dispatch. Q8_0
Q/K/V remains independent because the grouped narrow-KV microbenchmark regressed
by 5.9% on the development host.

Ten trials per mode on the controlled eight-vCPU AMD EPYC host, OpenJDK 25.0.3,
used Vectors commit `2a3466e`, Models commit `51f287d`, the same SmolLM2 Q8_0
bytes, prompt strategy, 128-token context, two warmups, and 24 generated tokens:

| Metric | Independent | Grouped gate/up | Change |
| --- | ---: | ---: | ---: |
| p95 TTFT | 1,080.8 ms | 713.1 ms | -34.0% |
| p95 TPOT | 65.82 ms | 44.56 ms | -32.3% |
| p50 prefill | 16.65 tok/s | 25.28 tok/s | +51.8% |
| p50 decode | 15.24 tok/s | 22.65 tok/s | +48.6% |

All ten corresponding output SHA-256 values matched. Each independent 2,560x960
gate/up matrix is below the Q8 parallel threshold; the combined row range crosses
it and uses the persistent workers. The benchmark policy changed from `USABLE`
to `RESPONSIVE`. This focused A/B isolates grouping; the refreshed cross-backend
table above uses a different prompt workload and includes the later Q8_0 prefill
path.

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
- Pure Java: Models `a1f0919`, Vectors `4144202`
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
