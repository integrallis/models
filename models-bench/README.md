# Inference Benchmarks

`models-bench` contains the controlled comparison harness and a decode-only JFR profiler for the
pure-Java backend.

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
