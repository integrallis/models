# Ragged multi-session prefill profile

This experiment compares prompt ingestion for four independent sessions over one loaded physical
model. `sequential` invokes the ordinary session-prefill path once per prompt. `ragged` advances the
same token index from every still-active session through one shared transformer pass, retains
separate positions and KV caches, and performs one final vocabulary projection for the batch.

The candidate deliberately does not flatten several positions from one session into one physical
pass. That later design needs an explicit within-session causal boundary; this control needs no
cross-user attention mask because every session owns its attention and KV state.

## Protocol

- Host: Intel Core i7-9750H, 12 logical processors, 32 GiB RAM, macOS 26.6.2
- JVM: Eclipse Temurin 25.0.3, HotSpot C2, Vector API at 256 bits
- Context capacity: 512 tokens per session
- Concurrency: 4 independent sessions with intentionally unequal prompt lengths
- Warmup: 2 rounds
- Measurement: 5 rounds in fresh JVM processes
- Timed work: prompt prefill only
- Correctness after the clock: argmax of every final logit row plus one retained-session continuation
- Artifact identity: exact SHA-256 is stored in each JSON report

## Results

| Model | Sequential | Ragged | Change | Correctness |
| --- | ---: | ---: | ---: | --- |
| MiniCPM5 1B Q4_K_M | 5.84 prompt tok/s | 12.10 prompt tok/s | +107.2% | exact continuation hash |
| Qwen3 0.6B Q4_0 | 21.36 prompt tok/s | 19.60 prompt tok/s | -8.2% | exact continuation hash |

The implementation is profitable for MiniCPM on this host and not for Qwen. Runtime selection must
therefore stay explicit and disabled by default; backend support and numerical correctness alone do
not establish a performance win.

The complete MiniCPM generation-session profile also compared the existing chunked scheduler with
the same scheduler plus ragged prompt batching:

| Runtime mode | Aggregate | p50 TTFT | p95 TTFT | p50 total | p95 total | Peak RSS |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Chunked per-session prefill | 3.53 tok/s | 4.69 s | 9.79 s | 15.62 s | 16.08 s | 841.4 MiB |
| Ragged prefill | 3.73 tok/s | 7.56 s | 7.85 s | 14.07 s | 15.96 s | 1,552.6 MiB |

Ragged prefill improved aggregate throughput 5.54%, p95 TTFT 19.85%, and p50 total latency 9.89%,
but worsened p50 TTFT 61.16% and raised peak RSS 84.52%. Sequential prompt service lets the first
request decode early while later requests wait; ragged service completes the admitted prompts
together, improving tail fairness at the expense of the first rows. The output hashes match.

## Reproduction

Run each mode in a separate JVM, replacing `<MODEL>` with the exact GGUF path:

```shell
./gradlew :models-bench:run --args='profile-ragged-prefill \
  --model <MODEL> \
  --mode sequential \
  --context 512 \
  --concurrency 4 \
  --warmups 2 \
  --iterations 5 \
  --prompt "Explain why a Java application should isolate request state." \
  --output ../benchmark-results/2026-09-07-ragged-prefill/result.json' \
  --no-daemon
```

Use `--mode ragged` for the candidate. Compare `artifactSha256`, `promptTokens`, and
`outputSha256` before comparing throughput. The reports also retain process memory, CPU, GC, JVM,
and backend optimization diagnostics.
