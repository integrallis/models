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

## Exact determinism audit

Audit the raw float bits of every generated logit vector across repeated greedy inference trials:

```shell
./gradlew :models-bench:installDist
models-bench/build/install/models-bench/bin/models-bench determinism \
  --model /path/to/model.gguf \
  --model-id model-id \
  --prompt-file models-bench/prompts/completion.txt \
  --tokens 4 \
  --iterations 3 \
  --context 128 \
  --output build/reports/determinism/model-id.json
```

The report records the artifact hash, prompt tokens, winning and runner-up logits, winner margins,
raw-bit logit hashes, JVM and host details, and the complete vectors execution configuration. The
command exits non-zero when any trial differs.

Use `scripts/run-purejava-determinism-matrix.sh` for fresh-JVM coverage of the default Panama path,
serial execution, FMA-disabled execution, 128-bit vectors, and the scalar reference. Provider
selection and vector constants are fixed at class initialization, so these modes must not share a
JVM.

## Decode-only profile

Run prompt prefill and warmup before starting JFR, then record a fixed-token autoregressive decode
loop:

```shell
./gradlew :models-bench:run --args='profile-decode \
  --model /path/to/model.gguf \
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

The requested context must hold the prompt, warmup, and measured tokens. The model path is validated
before loading, and the command rejects token IDs outside the model vocabulary.
