# mellum: library-default correctness smoke — PASS

The record `catalog/qualifications.json` needs before a new entry is admitted.
`tools/qualification-smoke-gate.mjs` in modeljars refuses any new entry, or one whose evidence
changed, unless it carries a `defaultConfigurationSmoke`.

Measured 2026-11-02, pod `ibky46v57ft42l`, 16 vCPU of an **AMD EPYC 9754** (avx2, avx512f present
on the host), workload `coding`, template `chatml-no-think`, top-k 1, context 2048, max-tokens 64,
threads 16.

## Result

    PREDICATE PASS
    SHIM ROUTE PASS  (native-quantized-decode == "true")

`smoke.json`, emitted in the sanctioned shape:

| field | value |
|---|---|
| `kind` | `model-backend-default-correctness-smoke` |
| `configuration` | `library-defaults` |
| `backend` | `rust-ffm` |
| `modelId` | `mellum2_12b_a2_5b_thinking_q4_k_m` |
| `artifactSha256` | `489cf0d7…3ccb` |
| `report` | `default-correctness/models-rust-ffm.json` |
| `reportSha256` | `3e95e3e096f028eb…0d1c` |
| `totalAttempts` / `successfulAttempts` | 9 / 9 |
| `tuningSystemProperties` | `[]` |

The committed report hashes to exactly the `reportSha256` the record names — checked, because the
gate matches evidence to claim by SHA-256 and not by filename.

## This reproduces the sanctioned script, it does not invent a check

`scripts/run-controlled-rag-qualification.sh` is the source of truth and the runbook requires a
fleet equivalent to reproduce its jq predicate rather than write a new one. Reproduced here:
its `COMMON_ARGS`, its `--warmups 0 --iterations 1` default-correctness phase, its predicate
verbatim (`backend`, `modelId`, `artifactSha256`, `settings.warmups == 0`,
`settings.iterations == 1`, `promptCache == "longest-common-prefix"`, `totalAttempts > 0`,
`successfulAttempts == totalAttempts`, `correctAnswerRate == 1`, `abstentionAccuracy == 1`,
`(.failures | length) == 0`), its `native-quantized-decode == "true"` route assertion, and its
record shape. `smokerun.sh` is the exact script that ran.

Its hygiene is mirrored too: the run refuses to start if any competing inference process is
present. **One deviation, recorded rather than hidden:** `drop_file_cache` could not run —
`/proc/sys/vm/drop_caches` is read-only in the container. That affects cold-read timing fairness,
which the *performance* phase cares about; it does not affect any assertion in this correctness
predicate.

## Why the throughput here is far below the qualification run

This report shows decode **14.88** tok/s and prefill **15.50** tok/s, against 44.99 / 68.35 in
`2026-11-02-mellum-q4k-kernel-verification`. That is expected and is not a regression: the smoke
runs at `--warmups 0`, so nothing is JIT-warm, and at 64 max-tokens rather than 256, so start-up
weighs far more per token. The script keeps correctness and performance as deliberately separate
phases for exactly this reason, and the runbook states the qualification runs cannot stand in for
the smoke. **No performance claim should be read from this record.**

## The library under test

The native library was built from this branch's Rust source on the pod and hashes to
`0c6127f17821773f…bdd474` — **byte-identical** to the library measured in the qualification run on a
different pod in a different data centre. Same source, same binary, so the correctness record and
the performance record describe the same code. The build is the same `cargo build --release` that
`cargoBuildRelease` runs before packaging the library as a platform resource, so this is what the
jar bundles.

`-Dmodels.native.kernels.library` points at that build, which is what the sanctioned script itself
does; the runbook's prohibition is on *tuning* properties, and `tuningSystemProperties` is `[]`.

## Files

`smoke.json` is the record; `default-correctness/models-rust-ffm.json` the report it names, with
`run.log` beside it. `HOST.txt` carries the CPU, core count, ISA and the library digest.
`smoke.sh` provisioned the box and `smokerun.sh` ran the phase.
