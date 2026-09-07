# Continuous batching through the public runtime

Date: 2026-09-07

## Question

Can a scheduler above `InferencePipeline.openGenerationSession()` turn the existing
`BatchInferenceBackend` kernels into a measurable serving benefit without changing generated
output or mixing request KV state?

## Environment

- macOS 26.6.2, x86-64
- Intel Core i7-9750H, 6 cores / 12 logical processors, 32 GiB physical memory
- Eclipse Temurin 25.0.3, HotSpot C2
- Vector API provider, 256 active bits, 12 persistent GGUF workers
- pure-Java execution plan `pure-java-v20`

Every comparison uses the same process launch, exact model bytes, prompt, greedy sampling,
request count, and output-token limit. Serialized and continuous modes run in fresh JVMs. The
reports retain per-request output hashes, external request-boundary TTFT/total latency, runtime
token counts, process/JVM memory, CPU, GC, backend diagnostics, and scheduler utilization.

## Results

| Model and workload | Mode | Concurrent requests | Completed tokens | Aggregate tokens/s | p50 TTFT | p95 request total | Peak RSS | Mean physical batch |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3 0.6B Q4_0, 26-token prompt | serialized | 2 | 48 | 6.250 | 905.7 ms | 2,601.3 ms | 1,222.4 MB | — |
| Qwen3 0.6B Q4_0, 26-token prompt | continuous | 2 | 48 | 5.123 | 1,249.7 ms | 3,247.9 ms | 1,284.7 MB | 2.00 |
| Qwen3 0.6B Q4_0, 149-token prompt | serialized | 2 | 32 | 1.225 | 6,165.1 ms | 13,139.9 ms | 1,251.8 MB | — |
| Qwen3 0.6B Q4_0, 149-token prompt | continuous | 2 | 32 | 1.190 | 11,851.2 ms | 13,763.1 ms | 1,258.5 MB | 2.00 |
| MiniCPM5 1B Q4_K_M, 25-token prompt | serialized | 4 | 112 | 2.092 | 10,965.4 ms | 29,357.4 ms | 825.4 MB | — |
| MiniCPM5 1B Q4_K_M, 25-token prompt | continuous | 4 | 112 | 3.406 | 4,457.4 ms | 17,965.8 ms | 898.9 MB | 3.50 |

Output hashes match within every serialized/continuous pair. MiniCPM improves aggregate
throughput by **62.83%**, reduces p50 TTFT by **59.35%**, and reduces p95 request time by
**38.80%**, while peak RSS rises **8.90%**. Qwen's short profile regresses aggregate throughput by
**18.02%**. With bounded 128-token prompt chunks, its longer profile regresses by **2.81%** and
does not qualify as a throughput optimization on this host.

## Product decision

Retain the Java scheduler as an explicit, bounded runtime option. It preserves one sampler,
constraint, stream, prompt-prefix lineage, and KV cache per request; batches only sessions owned by
one physical model; replaces completed rows; and uses bounded prompt chunks so an admitted long
prompt cannot occupy an unbounded scheduling turn.

Do not enable it automatically and do not infer a profitable batch size from backend capacity.
`maximumBatchSize` is therefore required explicitly. The MiniCPM result proves a complete public-
API win, while Qwen proves that batch correctness and batch capacity are not performance claims.
Deployment qualification must cover the exact model, quantization, concurrency, JDK, and host.

## Reproduction

Run the two modes in separate processes, changing only `--mode` and the output path:

```bash
./gradlew :models-bench:run --args='profile-runtime-sessions \
  --model /absolute/path/to/model.gguf \
  --mode serialized \
  --concurrency 4 \
  --warmups 1 \
  --iterations 2 \
  --max-tokens 16 \
  --output build/reports/inference/runtime-serialized.json'

./gradlew :models-bench:run --args='profile-runtime-sessions \
  --model /absolute/path/to/model.gguf \
  --mode continuous \
  --concurrency 4 \
  --warmups 1 \
  --iterations 2 \
  --max-tokens 16 \
  --output build/reports/inference/runtime-continuous.json'
```

The retained raw reports in this directory are the evidence for the table.
