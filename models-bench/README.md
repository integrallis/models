# Inference Benchmarks

`models-bench` contains the controlled comparison harness and a decode-only JFR profiler for the
pure-Java backend.

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
