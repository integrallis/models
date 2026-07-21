# Inference Benchmarks

`models-bench` contains the controlled comparison harness and a decode-only JFR profiler for the
pure-Java backend.

## Retained grouped K-quant prefill gate

The mixed Q4_K/Q4_K/Q6_K grouped batched path was retained on the controlled eight-vCPU AMD
EPYC-Milan host with Java 25. Using `prompts/completion.txt`, MiniCPM5 1B Q4_K_M, prefill batch 32,
one warmup, and 12 counterbalanced measurements per mode produced:

| Mode | Median TTFT | Median prefill | Output hash |
| --- | ---: | ---: | --- |
| Independent batched projections | 8330.16 ms | 17.95 tok/s | `01ba4719...546b` |
| Grouped batched projections | 7948.31 ms | 18.78 tok/s | `01ba4719...546b` |

This is a 4.58% TTFT reduction and a 4.61% prefill-throughput increase with bit-identical output.
Use `-Dmodels.purejava.groupedProjections=false` for the controlled baseline or rollback. This gate
does not justify a global batch-size increase: Q4_0 and Q8_0 models plateaued near batches 32-64,
while MiniCPM's ungrouped mixed K-quant path was fastest at batch one.

## Retained grouped Q4_0 prefill gate

Qwen3 0.6B Q4_0 was measured independently on the same Java 25 EPYC-Milan host, with prefill batch
32, the committed completion prompt, and six counterbalanced process pairs. Across 18 trials per
mode, grouped Q/K/V and gate/up projection dispatch improved median TTFT from 3127.92 to 3096.45 ms
and median prefill throughput from 49.59 to 50.19 tok/s. Every paired trial had identical input and
output token counts and an identical output SHA-256.

An allocation-enabled JFR follow-up used a fixed 2 GiB heap, five warmups, and five measurements.
Grouped median TTFT was 3076.4 ms versus 3124.0 ms, while total recorded TLAB plus outside-TLAB
allocation differed by 9.99 MB over the complete ten-prompt process. Peak RSS differed by 8.28 MB.
The much larger delta seen with only one warmup was transient JIT scalar-replacement behavior, not
steady per-prefill allocation. The same `models.purejava.groupedProjections` property controls this
path and provides its rollback gate.

## Retained grouped Q8_0 gate/up prefill gate

SmolLM2 360M Q8_0 was measured with the same Java 25 host, batch 32, completion prompt, and six
counterbalanced process pairs. Across 18 trials per mode, grouping only the batched gate/up
projection improved median TTFT from 2612.09 to 2581.73 ms and median prefill throughput from 60.22
to 61.01 tok/s. Mean process CPU time fell from 18,875 to 18,698 ms, median RSS did not increase,
and every pair had identical input/output token counts and output SHA-256 values.

Q8_0 Q/K/V remains independent. A direct follow-up against the retained dual-only baseline measured
2584.27 versus 2580.77 ms median TTFT and 61.01 versus 61.04 tok/s for dual-only versus dual plus
triple Q/K/V. Three of six pairs were faster and three were slower, while median RSS increased by
6.68 MB. The route was rejected as noise despite exact outputs; no runtime flag exposes it.

## Retained Q5_0 batched prefill gate

DeepSeek-Coder 1.3B Q4_K_M contains Q5_0 tensors, so the missing Q5_0 batched operation forced its
entire prefill plan to batch one. The retained route was measured on the same Java 25 EPYC-Milan
host with the pinned 873,582,624-byte GGUF, a fixed 2 GiB heap, eight inference threads, one warmup,
three trials per process, and six counterbalanced batch-one/batch-32 process pairs. The committed
`prompts/code-completion-medium.txt` produced 193-196 tokens after the per-trial nonce.

The first process, batch one, had a 33.40-second median while every later batch-one process measured
16.81-16.90 seconds. That global-order cold-start outlier cannot be separated from mode, so the
steady aggregate uses the remaining five pairs (15 trials per mode):

| Metric | Batch 1 | Batch 32 | Change |
| --- | ---: | ---: | ---: |
| Median TTFT | 16,837.21 ms | 16,181.43 ms | -3.90% |
| Median prefill | 11.524 tok/s | 12.110 tok/s | +5.09% |
| Mean trial CPU | 127,165 ms | 123,253 ms | -3.08% |
| Median peak RSS, all six processes | 1,493,338,112 B | 1,590,452,224 B | +97,114,112 B |

All six paired medians favored batch 32. Corresponding input-token sequences and output hashes were
identical; the sole output hash was
`1a0f564ddc6039457b2fb26b3d6a316c15eba20a886449847c3210c35821a693`. A separate six-pair
23-24-token gate measured a smaller 1.80% TTFT reduction, while a single 421-token corroborating
trial measured 7.13%; those results show the expected prompt-length scaling but do not replace the
steady medium-prompt gate. Set `-Dmodels.purejava.prefillBatchSize=1` when memory capacity is more
important than prompt latency.

## Retained final-layer prefill FFN pruning gate

Ordinary prefill returns logits only for the final prompt token. After the final layer has written
every K/V cache row and completed attention, FFN results for earlier prompt rows cannot affect
future decoding. The retained plan skips those FFN rows for the measured homogeneous Q4_0 and Q8_0
layouts, while speculative verification and observer-backed diagnostics continue to compute every
requested output row. Other tensor layouts remain on their exact full-row path pending a separate
gate.

Two quantization families were measured on the same Java 25 EPYC-Milan host with prefill batch 32,
the committed completion prompt, fixed heaps, one warmup, and six counterbalanced process pairs.
Each mode contains 18 measured trials:

| Model | Metric | Full final FFN | Pruned final FFN | Change |
| --- | --- | ---: | ---: | ---: |
| Qwen3 0.6B Q4_0 | Median TTFT | 3103.87 ms | 3037.63 ms | -2.13% |
| Qwen3 0.6B Q4_0 | Median prefill | 50.021 tok/s | 51.167 tok/s | +2.29% |
| Qwen3 0.6B Q4_0 | Mean trial CPU | 22,455 ms | 21,942 ms | -2.29% |
| SmolLM2 360M Q8_0 | Median TTFT | 2580.03 ms | 2528.12 ms | -2.01% |
| SmolLM2 360M Q8_0 | Median prefill | 61.097 tok/s | 62.381 tok/s | +2.10% |
| SmolLM2 360M Q8_0 | Mean trial CPU | 18,696 ms | 18,289 ms | -2.17% |

All six pair medians improved for both models. Corresponding input counts, output counts, and output
SHA-256 values matched in all 36 paired trials. The default-on Qwen path also passed its pinned
llama.cpp greedy-token reference. Median process peak RSS changed from 1,192,462,336 to
1,221,462,016 bytes for Qwen and from 688,959,488 to 705,396,736 bytes for SmolLM2. The
implementation adds no retained buffers; these overlapping process-level observations are recorded
without claiming a memory improvement. Use
`-Dmodels.purejava.finalLayerPrefillPruning=false` for rollback or reproduction.

## Retained final-layer K/V-only prefill gate

The next graph stage moves the final-layer output-row boundary ahead of Q projection. Every prompt
row still runs attention normalization and K/V projection, normalization, RoPE, and cache storage.
Only the requested final row runs Q projection, attention, output projection, and FFN. The stage is
enabled only when all final-layer attention and FFN projections use the same validated Q4_0 or Q8_0
format; the retained FFN-only route remains the fallback.

The incremental baseline kept final-layer FFN pruning enabled and disabled only K/V-only prefill.
The candidate used the same Java 25 EPYC-Milan host, batch 32, committed completion prompt, fixed
heaps, one warmup, three measured trials per process, and six counterbalanced process pairs:

| Model | Metric | FFN-only baseline | K/V-only prompt rows | Change |
| --- | --- | ---: | ---: | ---: |
| Qwen3 0.6B Q4_0 | Mean trial TTFT | 3039.72 ms | 3023.80 ms | -0.52% |
| Qwen3 0.6B Q4_0 | Mean prefill | 51.024 tok/s | 51.341 tok/s | +0.62% |
| Qwen3 0.6B Q4_0 | Mean trial CPU | 21,989 ms | 21,858 ms | -0.60% |
| SmolLM2 360M Q8_0 | Mean trial TTFT | 2532.43 ms | 2515.50 ms | -0.67% |
| SmolLM2 360M Q8_0 | Mean prefill | 62.203 tok/s | 62.623 tok/s | +0.68% |
| SmolLM2 360M Q8_0 | Mean trial CPU | 18,298 ms | 18,207 ms | -0.50% |

All six paired medians improved for both models, and all 36 corresponding trials matched input
count, output count, and output SHA-256. Median process peak RSS changed from 1,200,603,136 to
1,203,666,944 bytes for Qwen and from 708,460,544 to 694,071,296 bytes for SmolLM2. Synthetic Q4_0
and Q8_0 tests additionally require bit-identical complete K/V buffers and next-token state for
batch one and batch 32. The default-on Qwen path passed its pinned llama.cpp greedy reference and
every-layer state test. Use `-Dmodels.purejava.finalLayerKvOnlyPrefill=false` to return to the
retained FFN-only baseline.

## Exact determinism audit

Audit the raw float bits of every generated logit vector across repeated greedy inference trials:

```shell
./gradlew :models-bench:installDist
models-bench/build/install/models-bench/bin/models-bench determinism \
  --modeljar qwen3_0_6b_q4_0 \
  --prompt-file models-bench/prompts/completion.txt \
  --tokens 4 \
  --iterations 3 \
  --context 128 \
  --output build/reports/determinism/qwen3_0_6b_q4_0.json
```

The schema-2 report records the artifact hash, prompt tokens, winning and runner-up logits, winner
margins, raw-bit logit hashes, JVM and host details, the complete vectors execution configuration,
and the selected backend plan and ModelJar profile decisions. The command exits non-zero when any
trial differs.

Use exactly one model source. `--modeljar <alias>` resolves the descriptor from the aggregate
catalog on the application runtime classpath and preserves its model-scoped performance profile
when loading the backend. `--model /path/to/model.gguf` loads unregistered bytes without a profile.

Use `scripts/run-purejava-determinism-matrix.sh` for fresh-JVM coverage of the default Panama path,
serial execution, FMA-disabled execution, 128-bit vectors, and the scalar reference. Provider
selection and vector constants are fixed at class initialization, so these modes must not share a
JVM.

## Decode-only profile

Run prompt prefill and warmup before starting JFR, then record a fixed-token autoregressive decode
loop:

```shell
./gradlew :models-bench:run --args='profile-decode \
  --modeljar qwen3_0_6b_q4_0 \
  --prompt "Explain vectorized inference." \
  --context 2048 \
  --warmup-tokens 64 \
  --measure-tokens 256 \
  --output build/reports/inference/decode-profile.jfr'
```

The command reports prompt, warmup, and measured token counts, decode throughput, and a deterministic
logit checksum. The recording excludes model loading, prompt prefill, and decode warmup so CPU
attribution describes steady autoregressive decode. Use `--prompt-file` for a file-backed prompt and
`--token-id` to select the repeated token explicitly.

The requested context must hold the prompt, warmup, and measured tokens. The resolved artifact is
validated before loading, and the command rejects token IDs outside the model vocabulary. As with
the determinism command, `--model /path/to/model.gguf` is the explicit unregistered alternative.

## ModelJar-backed comparison run

Use a catalog alias for pure-Java measurements that must apply and record an exact model-scoped
execution plan:

```shell
./gradlew :models-bench:installDist
models-bench/build/install/models-bench/bin/models-bench \
  --backend pure-java \
  --modeljar qwen3_0_6b_q4_0 \
  --prompt-file models-bench/prompts/completion.txt \
  --max-tokens 64 \
  --warmups 2 \
  --iterations 10 \
  --context 2048 \
  --output build/reports/inference/qwen3_0_6b_q4_0-pure-java.json
```

Benchmark schema 5 records `backendDiagnostics`, including the pure-Java plan version, runtime
fingerprint, optimization decisions, and any enabled ModelJar performance profile. External
Ollama and llama.cpp reports record explicit `unavailable` diagnostics.
