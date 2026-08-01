# MiniCPM5 Q6_K Register-Tile Qualification

This gate measures the Q6_K strategy selected by the immutable Models execution plan through the
complete MiniCPM5 1B Q4_K_M graph. It is a prompt-prefill test: each request generates one token so
decode throughput cannot mask changes in time to first token.

## Controls

| Control | Value |
| --- | --- |
| Host | 8-vCPU AMD EPYC-Milan, 32 GiB RAM, Ubuntu 6.8 |
| JVM | Eclipse Temurin 25.0.3, HotSpot C2 |
| SIMD | Vector API, 256-bit species |
| Heap | `-Xms1g -Xmx1g` |
| Model | MiniCPM5 1B Q4_K_M, 688,065,920 bytes |
| Model SHA-256 | `81b64d05a23b17b34c475f42b3e72fbde62d4b92cc34541f7a8031d0752deafa` |
| Prompt SHA-256 | `2db2d875631cc7e3af3f6e4471ae4c9b2b7dfdb31ab561a41ef78182a31532e6` |
| Backend Java JAR | `0af32a25c1e32cbcf85a6d64777e912d9242374cac08a869bfbab781b155009b` |
| Vectors Core 0.1.5 JAR | `f2671118ad1be7ff142eb68ea71647d95341f2c375415a6c1b1b10973947630f` |
| Benchmark JAR | `796f3d11c5bc078654f59261adac96c87cee1590ea30bfb9c33e33a55a08093c` |
| Plan | `pure-java-v19`, batch 32, eight workers |
| Process order | `ONE`, `TWO`, `TWO`, `ONE` |
| Per process | two warmups, 16 measured requests, one generated token |

The benchmark adds a deterministic nonce to each request. Corresponding one-query and two-query
processes therefore receive the same sequence of 149-152-token inputs without reusing a KV prefix.

## Results

| Metric | One-query block | Two-query block | Change |
| --- | ---: | ---: | ---: |
| p50 TTFT | 6,146.903 ms | 5,899.447 ms | -4.026% |
| p95 TTFT | 6,233.667 ms | 5,967.546 ms | -4.269% |
| p50 prefill | 24.423 tok/s | 25.419 tok/s | +4.076% |
| p50 process CPU | 45,640 ms | 43,520 ms | -4.645% |
| maximum observed RSS | 1,426,980,864 B | 1,412,833,280 B | -0.991% observed |
| completed young collections | 112 | 8 | -92.857% |
| young-GC pause time | 265.969 ms | 57.807 ms | -78.266% |

The nonparametric bootstrap used 50,000 deterministic resamples. Its 95% interval is 3.561-4.462%
for p50 TTFT reduction and 3.799-4.383% for p50 prefill gain. The complete observed TTFT ranges do
not overlap: one-query measured 6,050.038-6,235.386 ms and two-query measured
5,848.431-5,982.920 ms.

Every trial succeeded. Corresponding input-token sequences and output hashes match. The sole output
SHA-256 is `01ba4719c80b6fe911b091a7c05124b64eeece964e09c058ef8f9805daca546b`.
Backend diagnostics prove that both runs used plan `pure-java-v19`; the candidate reports
`q6-batched-kernel=two-query-block`, while the control reports `one-query-block`.

The RSS values are retained as observations, not as evidence of reduced memory requirements. The
large GC-count reduction is corroborated by both independent processes, but it describes this
fixed-heap workload and is not generalized to other model graphs.

## Reproduction

Run each mode in a fresh process, preserving the ABBA order:

```bash
export JAVA_OPTS='-Xms1g -Xmx1g -XX:ActiveProcessorCount=8 \
  -Xlog:gc*:file=MODE.gc.log:uptime,level,tags \
  -Dmodels.purejava.q6BatchedKernel=MODE'

models-bench \
  --backend pure-java \
  --model /path/to/MiniCPM5-1B-Q4_K_M.gguf \
  --prompt-file models-bench/prompts/completion.txt \
  --max-tokens 1 \
  --warmups 2 \
  --iterations 16 \
  --context 2048 \
  --output MODE.json
```

Replace `MODE` with `one-query-block` or `two-query-block`, and use a distinct report and GC-log
path for every fresh process.
