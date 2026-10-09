# Catalogue-wide default-configuration smoke on Models 0.3.54

`CAMPAIGN-RUNBOOK.md` requires a default-configuration smoke across the whole catalogue whenever the
released library changes the grounding policy, the decoder, or the kernel. 0.3.54 changed kernel
dispatch defaults, so every published entry needed a current proof that it answers its whole workload
correctly under library defaults.

Run 2026-10-09 on eight `m6a.4xlarge` (16 vCPU AMD EPYC Milan), `smoke-worker-0354b.sh`, payload
`models-rag-bench-0.3.54-v23.tar`. **The payload's eight Integrallis jars are the artifacts published
to Maven Central**, not a local build, so this measures the library that ships — including
`backend-native-0.3.54.jar`, whose `linux-x86_64/libjmodels_kernels.so` the fleet actually loads.

Protocol per entry, from the runbook's gate: library defaults, **no `-Dmodels.*` of any kind**,
`warmups=0`, `iterations=1`, each entry's own published `promptTemplate`, `workload` and `backend`.
The worker refuses to start if any JVM options variable carries a `models.*` property, because a
`RagBenchmarkCli` report has no tuning field and the artifact therefore cannot prove "nothing was
configured".

## Result: 65/65 reported, 57 PASS

| outcome | n | |
|---|---|---|
| PASS | 57 | answers its whole workload correctly under defaults |
| FAIL, `correctAnswerRate=0.889` | 6 | one case of nine, in a single cold iteration |
| run failed | 1 | `qwen_qwen2_5_0_5b_instruct_bf16` |
| download failed | 1 | `facebook_mobilemoe_s_qat_int4_g32` |

Neither hard failure is a library fault:

- **`qwen_qwen2_5_0_5b_instruct_bf16`** — `MalformedGgufException: Invalid GGUF magic: 0x00007E18`.
  The catalogue's `downloadUri` for this entry does not return a GGUF. A manifest problem, and one
  this smoke found because it fetches what the catalogue says to fetch.
- **`facebook_mobilemoe_s_qat_int4_g32`** — `format: safetensors`, a Hugging Face *directory*. This
  worker handles single-file artifacts only; `qual-worker-two-arm.sh` handles the `files` array form.
  A worker limitation, stated rather than papered over: this entry is **unmeasured here**, not passing
  and not failing.

## The six FAILs are not a 0.3.54 regression

All six publish `correctAnswerRate = 1.0`, measured at `warmups=1, iterations=3`. This smoke runs
`0/1` by the runbook's own gate, so the two are not comparable, and a 0.889 proves nothing about the
release on its own.

So the same six were re-run under the **identical** worker, shard and gate against the previous
release, with the library as the only variable — shard-510 on 0.3.54 and shard-511 on a
`models-rag-bench-0.3.53-v23-baseline.tar` payload assembled from the Central-published 0.3.53 jars.
Raw arms: `ab-arm-0.3.54.tsv`, `ab-arm-0.3.53.tsv`.

**`correctAnswerRate = 0.8888888888888888` on both releases, for all six.** 0.3.54 did not cause it.
Each misses a *different* case — `telemedicine-benefit` (abstained), `berlin-sicherung`,
`http-retry-rules` (factCoverage 0.5) — with retrieval perfect in every one (`retrievalRecall 1.0`,
`reciprocalRank 1.0`). These are model-quality outcomes in a single cold iteration, not a decoder
change.

## And the A/B measured what 0.3.54 is worth

The 0.3.53 arm's gate output carries `native-quantized-decode=false` on every model — the old default,
observed rather than argued. Decode throughput, same host, same protocol, one variable:

| model | 0.3.53 | 0.3.54 | speedup |
|---|---|---|---|
| qwen2.5-coder-1.5b q4_0 | 7.51 tok/s | 27.88 | **3.71x** |
| smollm2-360m q8_0 | 16.30 | 42.72 | **2.62x** |
| eurollm-1.7b q4_k_m | 11.35 | 27.33 | **2.41x** |
| umartransit-1b q4_k_m | 11.25 | 26.08 | **2.32x** |
| smollm2-1.7b q4_k_m | 8.88 | 18.76 | **2.11x** |
| qwen2.5-coder-1.5b q8_0 | 10.22 | 18.18 | **1.78x** |

**Median 2.36x decode, with `correctAnswerRate` identical on every model.** That is the measured
value of making the fast path the default, on the hardware the catalogue is qualified on — and it is
larger than the 6.2x-on-one-model figure suggested at the low end because the old default was only
*off* when no profile matched, which on these entries it did not.

## What this does and does not license

Licenses: republishing the catalogue against 0.3.54. 57 entries carry a current default-configuration
proof on the shipped library, and the 6 gate failures are shown to predate the release.

Does not license: calling those 6 regressions, or calling the two unmeasured entries passing. The
`bf16` entry needs its `downloadUri` fixed; MobileMoE needs a worker that fetches an HF directory.

Reproduce: `scripts/fleet/smoke-worker-0354b.sh` with `smoke-shard-500..507.json`; the A/B with
shards 510/511 and `PAYLOAD`/`BACKEND_VERSION` overridden per arm.
