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
