# SmolLM2 360M Q8_0 Native Kernel Gate

This directory contains the raw reports for the first controlled
`backend-native` Q8_0 gate. It is implementation evidence, not a launch
qualification.

## Controls

- Host: Ubuntu 24.04, AMD EPYC Milan, 8 vCPU, 30.6 GiB RAM
- Java: GraalVM Community Java 25.0.3, fixed 1 GiB heap
- Models: `5395d8342015be62be587bd838366f395037204e`
- Vectors: `fde9858901624d1661a1cf51195d2c59737bcf87`
- ModelJars: `936034cd55174fccd42121aa1dc7309c49d55d6c`
- llama.cpp: build 10012, commit `c71854292`
- Ollama: `0.32.0`
- Model SHA-256:
  `48ab3034d0dd401fbc721eb1df3217902fee7dab9078992d66431f09b7750201`
- Workload: 157-token production-review prompt, 64 generated tokens,
  context 2,048, eight inference workers
- Repetition: five warmups and five measurements in each of six fresh client
  processes, for 30 measurements per backend

Pure Java and Rust were run as six counterbalanced fresh-process pairs. The
llama.cpp server disabled prompt caching. The Ollama model was stopped between
processes so its persistent runner could not reuse the repeated measurement
prompt series. An earlier persistent-runner experiment was rejected after it
exposed that cross-process prompt reuse reduced warm-resident TTFT to about
77 ms; those reports are not included.

## Aggregate Results

Percentiles use the benchmark runner's nearest-rank calculation over all 30
successful measurements.

| Backend | p50 TTFT ms | p95 TTFT ms | p50 prefill tok/s | p50 decode tok/s | p95 TPOT ms | p50 E2E ms | p95 E2E ms | Max RSS GiB |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Pure Java | 1,121.74 | 1,207.41 | 139.58 | 45.57 | 22.20 | 2,506.35 | 2,594.50 | 0.983 |
| Models Rust/FFM | 842.27 | 910.61 | 189.33 | 45.44 | 22.49 | 2,231.19 | 2,305.46 | 0.932 |
| Ollama | 328.84 | 339.26 | 586.42 | 51.84 | 22.31 | 1,540.21 | 1,739.55 | 0.615 |
| llama.cpp | 281.89 | 294.82 | 583.81 | 97.60 | 11.62 | 917.57 | 1,012.59 | 0.568 |

Relative to the matched pure-Java control, the Rust Q8_0 projection path
improves median TTFT by 24.91%, prefill by 35.64%, and median end-to-end
latency by 10.98%. Decode is unchanged because batch-one generation remains on
the Java kernel.

The Rust path reaches:

- 37.26% of Ollama and 32.38% of llama.cpp by p95 TTFT;
- 87.65% of Ollama and 46.55% of llama.cpp by median decode throughput; and
- 69.03% of Ollama and 41.12% of llama.cpp by median end-to-end latency.

For latency ratios, the comparator duration is divided by the Rust duration;
for decode, Rust throughput is divided by comparator throughput.

## Output Evidence

All four backends reproduced each of their five complete output hashes in all
six fresh processes. Corresponding input-token counts matched in all 30
measurements. Ollama and llama.cpp matched each other in 30 of 30 complete
outputs. Pure Java matched llama.cpp in 12 of 30, while the Rust path matched
llama.cpp in 6 of 30 and pure Java in 12 of 30.

The pinned eight-token integration oracle and synthetic projection tests pass,
but the additional complete-output drift means this kernel is not yet eligible
for a ModelJar recommendation. It must pass the model-specific RAG quality
gate, and prompt processing remains below the release performance target.
