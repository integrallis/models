# Qualification fleet worker

`qual-worker-two-arm.sh` is the EC2 user-data a qualification worker runs. It lives here because the
evidence behind every catalog entry comes out of it, and for the whole 2026-09 campaign it existed only
in a scratch directory: the script that produced the numbers was not reviewable, not diffable, and not
recoverable.

## Why two arms

Qualification is comparative. `RagProductionQualificationPolicy` compares a candidate against baselines
measured on the **same host** and the **same workload**, and returns `NO_COMPARABLE_BASELINE` when there
are none. The single-arm worker this replaces produced fourteen candidate reports and zero verdicts,
which is how a campaign came to report models as qualified that had never been assessed.

Per model the worker now:

1. downloads the GGUF and runs the candidate arm (`rust-ffm`);
2. `ollama create`s **that same file** so `artifactSha256` matches, and runs the comparator arm;
3. runs `RagQualificationCli` and uploads candidate, comparator and verdict.

## The settings that must match, and why threads are explicit

`sameWorkload` compares workload, corpus sha, case ids, prompt template, retrieval top-k, max output
tokens, context length, **threads**, grounding policy, minimum retrieval score, and the matched
generation controls. Any difference excludes the comparator with `benchmark workload differs`. A run
that passed `--threads 8` to the candidate and let the comparator default to the host's core count was
rejected for exactly that, so both arms are given `--threads $THREADS` from one variable.

## What a verdict can say

- `QUALIFIED` — contribution gate and relative gate both passed.
- `FAILED_MODEL_CONTRIBUTION_GATE` — under a 1/3 model-answer rate, or under 90% correct among the
  answers the model did contribute. The pipeline answered; the model did not.
- `FAILED_RELATIVE_GATE` — slower than the comparator beyond the policy's floor and ceiling.
- `NO_COMPARABLE_BASELINE` — the comparator was excluded; `exclusions` in the verdict names why.

## The manifest is the knob, not the script

`fleet-shard-NNN.json` is a list of jobs, each `{id, uri, tpl, gb, arch, wl, dt, mt}`:

| field | meaning |
| --- | --- |
| `tpl` | prompt template, from what the architecture already qualified with |
| `wl` | **the workload corpus**, derived from the model's declared capabilities |
| `dt` | decode threads, or omit for the pool default |
| `mt` | max output tokens, 256 when omitted |

Two of these were learned by getting them wrong, both on 2026-10-09.

**`wl` must match the capability.** A shard built with `wl: "general"` for every model returned
`FAILED_MODEL_CONTRIBUTION_GATE` for a math specialist and a translation model, while the same base
models were already qualified in the catalog on `math` and `multilingual`. Re-running
`eurollm-1.7b-instruct Q4_K_S` on `multilingual` with nothing else changed returned **QUALIFIED**.
Retrieval was perfect in the failing runs -- recall, MRR, factCoverage and both citation metrics all
1.0 -- so the corpus was the only variable. Put one workload per shard so a box loads one corpus.

**`mt` must fit the model.** `fin-r1-7b` returned `truncatedAnswerRate 0.778` and
`modelAnswerCorrectRate 0.0` at the old hard-coded 256: seven of nine answers were cut off
mid-sentence, so the metric measured the cap rather than the model. It was hard-coded in two places
and is now one per-job variable handed to **both** arms, because `sameWorkload()` compares max
tokens and a mismatch excludes the comparator.

## Building shard files: `build-shards.py`

`fleet-shard-NNN.json` used to be written by hand, and both fields that were learned the hard way
above were learned by getting them wrong by hand on the same day. `build-shards.py` builds them
from recorded evidence instead:

```
python3 scripts/fleet/build-shards.py \
    --ids ids.txt --reports ./prior-reports --catalog ../model-jars/catalog/models.json \
    --out-dir /tmp --start-shard 640 [--mt-override id=2048] [--wl-override id=finance]
```

**It copies, it does not derive.** For each requested model it reads `wl`, `tpl`, `dt` and `mt`
out of a prior candidate report for *that same model* and reuses them exactly, printing the path
it read each one from. Where no prior report exists it prints the precedent it found and **fails**,
writing nothing.

That restraint is deliberate, and it is not caution for its own sake. The obvious design —
derive `wl` from the model's capabilities and `tpl` from its architecture — does not survive the
data. Measured against the catalog on 2026-10-09:

- `medical-reasoning` maps to workload `general`, not `healthcare`
- `math` splits between `general` and `math`; `reasoning` is mostly `general`
- `chat` and `text-generation` appear against all eight workloads
- architecture `llama` has qualified under **eight** different prompt templates, `qwen2` under three

So a model's declared capabilities do not determine its workload and its architecture does not
determine its template. A script that picked anyway would be guessing with a script's authority,
which is worse than a person guessing, because nobody would look again.

### `dt` is not `--threads`

Two different knobs, and conflating them was this tool's first bug — found by running it against
real reports rather than by reading them:

| job field | becomes | recorded in the report as | fleet value |
| --- | --- | --- | --- |
| `dt` | `-Dmodels.native.kernels.decodeThreads` | `backendDiagnostics.environment.native-kernel-decode-threads` | 8 |
| *(not a job field)* | `--threads`, to **both** arms | `settings.threads` | 16 |

The first version read `settings.threads` and emitted `dt: 16`, silently undoing the decode-thread
setting the measured runs had actually used. `dt` is omitted entirely when the prior run used the
whole pool, so the job inherits the worker's default rather than pinning a number nobody pinned.

### Why a confirmation run reuses the settings

Re-measuring a model against a released build is only a comparison if everything except the build
is held fixed. Copying the recorded settings is what makes that true; an override is per field and
is marked `OVERRIDDEN` in the output so it cannot pass for a copied value.

Shard numbers are immutable — a number that has been launched names a payload and an S3 result
prefix — so the tool refuses to overwrite an existing shard file rather than reusing a number.

Tests: `python3 -m unittest discover -s scripts/fleet -p 'build_shards_test.py'` (stdlib only).
