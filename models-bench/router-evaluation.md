# Paired model-selection evaluation

Run `./gradlew :models-bench:run --args="router-evaluate /absolute/manifest.json /absolute/report.json"`.
The output must not already exist. This command spends nothing: it replays previously collected
provider observations through the actual `ModelFleet` and scores answers, rather than task labels.

Before collecting observations, freeze the prompt template, answer scorer, calibration/evaluation
split, model and client revisions, generation settings, hardware, costs and supported workloads.
Use at least two public datasets with references, for example the GSM8K and ARC sources already
recorded in model-router-corpus's provenance. That repository contains **prompts only**; its task
classification scores and historical bakeoff file cannot supply generation-quality observations.
Obtain references from the original pinned dataset revisions and keep their licenses/provenance.
Do not tune on evaluation answers. Exact stripped answer matching is the only current scorer; use
short-answer workloads that can exercise it and record prompts that ask for that answer format.
It is unsuitable for evaluating open-ended prose or judging executable code.

Collect every model on every calibration and evaluation case under the same protocol, with the
same explicit output limits. Record failures and partial answers, reported usage (or null), TTFT,
completion time, hardware/runtime, and a receipt location. Include classifier predictions and
measured classifier duration separately; never replace predictions with the reference task labels.
Local classification is assumed in this evaluator; remote classifier cost is outside its scope.
Keep prompt preparation consistent and randomize/interleave collection order to avoid comparing
warm and cold runs accidentally. Disable automatic provider retries or include their work in the
observation. Only run billable external providers within an explicitly authorized spend cap.

Manifest schema version 1:

```json
{
  "schemaVersion": 1,
  "protocolRevision": "commit + prompt-template/scorer digest",
  "modelsRevision": "Models commit",
  "classifierRevision": "model/index/client digests",
  "hardware": "exact host/CPU/GPU/memory and locality",
  "runtime": "JDK/backend/OS versions",
  "generationSettings": "temperature, seed, output caps, cache/warmup policy",
  "cases": {"path": "cases.json", "sha256": "actual digest"},
  "observations": {"path": "observations.json", "sha256": "actual digest"},
  "datasets": [
    {"id": "gsm8k", "source": "original dataset URI", "revision": "pinned revision", "license": "source license"},
    {"id": "arc", "source": "original dataset URI", "revision": "pinned revision", "license": "source license"}
  ],
  "models": [
    {"id": "local-a", "revision": "artifact SHA-256", "clientRevision": "client commit", "local": true,
     "capabilities": ["chat"], "contextWindow": 8192, "inputPrice": 0, "outputPrice": 0},
    {"id": "local-b", "revision": "artifact SHA-256", "clientRevision": "client commit", "local": true,
     "capabilities": ["chat"], "contextWindow": 8192, "inputPrice": 0, "outputPrice": 0}
  ]
}
```

`cases.json` is an array. Each entry has unique `id`, `dataset`, `split` (`calibration` or `eval`),
`prompt`, reference `answers` array, reference `task`, separately predicted `predictedTask`,
`classifierMillis`, integer `inputBound` and `outputBound`, `budget`, positive `deadlineMillis`,
optional `localOnly` and `capabilities`. Every dataset must have both splits. Duplicate stripped
prompts, including across splits or datasets, are rejected. Preserve original row IDs and source
split names in the IDs or accompanying source manifest. The pinned files include all prompts,
answers and observations; avoid redistributing them where source licensing forbids it.

`observations.json` is an array with exactly one entry per case/model pair:
`caseId`, `modelId`, boolean `success`, raw `response` string, `latencyMillis`, `ttftMillis`,
`inputTokens`/`outputTokens` (both non-negative integers or both absent/null), and `receipt`.
Failures still need response text (possibly empty), timing and a receipt. Partial paired data is
rejected rather than silently dropping a difficult case. The evaluator verifies file hashes;
reviewers must also inspect the provenance and receipts. A string claiming a revision is not proof
that collection used that revision.

Quality and TTFT candidate inputs are derived exclusively from calibration observations. Every arm
uses the same evaluation observations; router health starts fresh per case so arm ordering does not
create a hidden adaptation advantage. Static cheapest and static best are fixed calibration choices;
if a fixed model is ineligible the case is rejected, not replaced with a different baseline model.
The model-only arms also enforce request requirements and budgets.

The report includes each attempt, final selection, exact-answer quality, conservative ledger cost
including failed/unknown attempts, replayed completion latency, violations, rejected/failed cases,
and per-dataset/task p50/p95/p99. Quantiles use nearest rank; small samples are not precise tail
estimates. Router replay includes recorded classifier time. It does not include new router CPU
execution time, model warmup, concurrency interference or real cancellation of an observation.
Deadline overruns are reported; replay cannot rewind or cancel a recorded provider call. A bound
violation records actual successful usage and rejects the result. Unknown and failed usage stays
charged at the admission bound. Token-price zero does not imply infrastructure is free.

Ablations disable classification, quality weight or latency weight independently. Changed-selection
counts make an inactive switch observable: zero changes is **no evidence**, not proof that a component
is useless. Unit tests use explicitly synthetic fixtures only. A production quality claim requires
new, pinned public-data observations and a predeclared workload acceptance threshold; this change
does not claim that such qualification has already happened.
