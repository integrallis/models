# Does `models.native.loadWarmup` close the first-generation divergence?

**Status: protocol frozen before data collection. Not yet run.**

## Observation being explained

In the cross-surface exercise pass (shards `exercise-400/401/402/403/410/411`, 65 distinct models),
58 agree on every prompt and **5 are `NOT_REPRODUCIBLE`**:

- `unsloth_qwen3_5_2b_gguf_q4_k_m`
- `qwen2_5_math_1_5b_instruct_q4_k_m`
- `king3djbl_nexus_finance_gguf_q5_k_s`
- `king3djbl_nexus_finance_gguf_q4_k_m`
- `king3djbl_nexus_legal_gguf_q5_k_m`

All five fail on the *same* prompt, all on `rust-ffm`, all under greedy sampling, and all with one
shape: the first `plainJava` run differs while `langchain4j`, `springAi` and `plainJavaRepeat` are
byte-identical to each other. `FrameworkExerciseCli.java:134-160` runs all four in one process with
`backend.reset()` between each, and that prompt is first, so the odd run is **the first generation in
the process**; every later generation agrees. No adapter is implicated.

## Already ruled out

The worker pool. `execute_matrix_job_partition`
(`backend-native/src/main/rust/model-kernels/src/lib.rs:799-800`) splits by whole rows
(`start_row = matrix.rows * worker_index / total_threads`), so every `out[row]` is one complete dot
product on one worker. The reduction order does not depend on partition count, and pool warmth
therefore cannot change the arithmetic.

## Hypothesis

JIT state. `models.native.loadWarmup` defaults to `false`
(`NativeKernelSettings.java:97-99` — `booleanSetting` returns `false` when the value is null), so
generation 1 executes the Java transformer path (norms, softmax, rope, attention) cold. Once C2
compiles those loops, FMA contraction and reduction reassociation change float rounding; at a
near-tie that flips one argmax token and the whole continuation diverges. `RustFfmBackend.warmup`
(`RustFfmBackend.java:159-170`) prefills a throwaway prompt and resets, which would move that
compilation ahead of the first real generation.

## Arms

One box, both arms per model, back to back, same download and same process image — so host, build
and network are controlled.

| arm | JVM properties |
| --- | --- |
| `off` (control) | none — library defaults, exactly as the original evidence was produced |
| `on` (treatment) | `-Dmodels.native.loadWarmup=true` |

Everything else is held at the values that produced the original finding: the same
`models-rag-bench-0.3.51-exercise-91336eb7316b.tar` dist, the same
`models-kernels-linux-x86_64.jar`, the same prompts, `--context 2048`, `--max-tokens 96`, greedy.

Reusing the 0.3.51 dist is deliberate: it is the build that produced the observation, which makes
the `off` arm a true replication. A confirmation here must then be re-verified on the shipped
library before anything is claimed about a release.

## Subjects

7 models, from `exercise-shard-420.json`: the 5 `NOT_REPRODUCIBLE` subjects plus 2 controls that
scored `SURFACES_AGREE` 5/5 in shard 410 — `unsloth_qwen3_5_0_8b_gguf_q5_k_m` and
`ibm_granite_granite_4_1_3b_gguf_q4_k_m`.

## Metric

`plainJavaReproducibleForEveryPrompt` from each run's `*.exercise.json`, plus the derived verdict.

## Decision rule, fixed in advance

- **Kill criterion (check first).** If the `off` arm does not reproduce `NOT_REPRODUCIBLE` for at
  least 4 of the 5 subjects, the experiment is **void** — the baseline did not reproduce, so nothing
  can be concluded either way, and no result is reported.
- **Confirmed** if, with the baseline reproduced, ≥4 of 5 subjects flip
  `plainJavaReproducibleForEveryPrompt` from `false` to `true` in the `on` arm.
- **Refuted** if ≥3 of 5 subjects remain `false` in the `on` arm. The mechanism is then not JIT
  warmup, and the hypothesis is recorded as failed rather than reframed.
- **Ambiguous** otherwise (2 or 3 of 5 flip): report as ambiguous and do not act on it.
- **Controls:** both control models must stay `SURFACES_AGREE` in both arms. If a control changes
  verdict, the harness or the host is implicated and the run is void.

A confirmation does **not** by itself justify flipping the default: enabling warmup moves work to
load time, and that cost must be measured before any default changes.

## Reproduce

```
aws s3 cp exercise-shard-420.json      s3://models-qual-077051030817/payload/
aws s3 cp exercise-warmup-worker.sh    s3://models-qual-077051030817/payload/
EX_SHARD=420 bash exercise-warmup-worker.sh     # on an m6a.4xlarge with the models-qual-worker role
```

Results land in `s3://models-qual-077051030817/results/exercise-420/`.

## Result: VOID. 2026-10-06, shard `exercise-420`, one `m6a.4xlarge`.

Raw, 7 models x 2 arms, every run `rc=0`:

| model | `off` | `on` |
| --- | --- | --- |
| `unsloth_qwen3_5_2b_gguf_q4_k_m` | NOT_REPRODUCIBLE | NOT_REPRODUCIBLE |
| `qwen2_5_math_1_5b_instruct_q4_k_m` | NOT_REPRODUCIBLE | SURFACES_AGREE |
| `king3djbl_nexus_finance_gguf_q5_k_s` | NOT_REPRODUCIBLE | SURFACES_AGREE |
| `king3djbl_nexus_finance_gguf_q4_k_m` | NOT_REPRODUCIBLE | SURFACES_AGREE |
| `king3djbl_nexus_legal_gguf_q5_k_m` | NOT_REPRODUCIBLE | SURFACES_AGREE |
| *control* `unsloth_qwen3_5_0_8b_gguf_q5_k_m` | **NOT_REPRODUCIBLE** | **NOT_REPRODUCIBLE** |
| *control* `ibm_granite_granite_4_1_3b_gguf_q4_k_m` | SURFACES_AGREE | SURFACES_AGREE |

4 of 5 subjects flipped, which is the threshold this document set for "confirmed". **It is not
claimed**, because the rule had a prior condition and that condition failed.

**1. A control changed verdict, which this protocol declared a void condition.**
`unsloth_qwen3_5_0_8b_gguf_q5_k_m` scored `SURFACES_AGREE` 5/5 in shard 410 and came back
`NOT_REPRODUCIBLE` in *both* arms here.

**2. That control was never valid, and choosing it was the error.** Its history:
`SURFACES_DIFFER` 4/5 (shard 401) then `SURFACES_AGREE` 5/5 (shard 410) then `NOT_REPRODUCIBLE`
(here, both arms). Three verdicts in three runs. It was picked off shard 410's single row without
checking the other shards -- the same "one dataset, measured carefully, produced a confident wrong
answer" failure the working agreement names.

**3. Which exposes the real defect in the design: the phenomenon is intermittent per run, so one
run per arm cannot test it.** `unsloth_qwen3_5_2b_gguf_q4_k_m` likewise moved `SURFACES_DIFFER` 3/5
(400) to `NOT_REPRODUCIBLE` 4/5 (410) to `NOT_REPRODUCIBLE` in both arms (here). If divergence fires
with some probability per run, a subject reading `false` under `off` and `true` under `on` is equally
consistent with the warmup fixing it and with the divergence simply not firing that once.

**4. And the toggle was not observable.** The exercise report's keys were
`[allSurfacesAgree, anyEmptyAnswer, backend, chatTemplate, maxOutputTokens, modelId,
plainJavaReproducibleForEveryPrompt, promptCount, prompts, promptsAgreeingAcrossSurfaces, sampling,
schemaVersion]` -- no settings, no diagnostics -- and neither arm's log mentions warmup. There is no
artifact showing `-Dmodels.native.loadWarmup=true` took effect. On its own this disqualifies the run.

### The hypothesis is untested, not refuted

Nothing here says JIT state is the wrong explanation. It says this experiment could not test it.

### What the redesign requires, before another box is spent

- **Repetitions, not verdicts.** N >= 10 runs per arm per model, comparing divergence *rate*, since
  the per-run outcome is stochastic. A single-run verdict is not a measurement of a flaky effect.
- **An observable toggle.** Fixed: `FrameworkExerciseCli.recordConfiguration` now writes
  `backendPlanVersion`, `backendEnvironment` and every `models.*` property into the report, with
  tests pinning that absent and empty stay distinguishable.
- **Controls chosen on full history across every shard**, ideally models that have never diverged in
  any run -- not on one row of one shard.
