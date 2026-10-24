# Bonsai 2 27B PTQ1_0 final-runtime admission, 2026-10-23 — protocol frozen first

Step 6 of the Bonsai execution plan (`docs/findings/2026-10-11-bonsai-production.md`): *"Release
any runtime repair, repeat with downloaded Central runtime artifacts, then follow ModelJars' marker
verification, Central publication and site process."*

Steps 1-5 are recorded in `../2026-10-11-bonsai-production/`. This directory exists because every
measurement there **predates the BF16 determinism repair** and therefore cannot admit a release.
Nothing here re-grades those results; they stand as the source-run record.

## Why a repeat is required rather than a re-reading

`../2026-10-11-bonsai-production/rag-source-06c75314/verdict.json` already says QUALIFIED under
`production-rag-publisher-reference-v1`. It is not admissible for publication for one reason: it
was measured at source commit `06c75314`, before `BFloat16Matrix.finishRow` was changed from
`reduceLanes(ADD)` -- whose addition order changes after compilation -- to Vectors' fixed-order
reduction. A qualification must be measured on the library that ships.

## What is being measured

| shard | model | arch | wl | tpl | mt | gb |
| --- | --- | --- | --- | --- | --- | --- |
| 800 | `prism_ml_ternary_bonsai_2_27b_gguf_ptq1_0` | qwen35 | `general` | chatml non-thinking | 64 | 5.54 |

One model, one workload, one host, two arms. `wl: general` and the ChatML non-thinking template
reproduce the source run's settings exactly, because the question is whether the *runtime change*
moved the result -- changing a workload at the same time would confound that.

## Prediction, registered before the run

Publishing the null is the rule, so the expectation goes on the record first.

**Expected QUALIFIED, and the decode ratio is the number at risk.**

The source run cleared the throughput gate at `decodeThroughputRatio 0.9006` against a floor of
`0.80`. That is a margin of **0.10**, and the repair replaces a compiler-reorderable reduction with
a fixed-order one, which cannot choose the cheapest addition order. So:

- If the repair is throughput-neutral, expect a decode ratio near 0.90 and QUALIFIED.
- **If the repair costs more than ~11% of decode throughput, the ratio falls below 0.80 and the
  verdict becomes `FAILED_RELATIVE_GATE`.** That is the interesting result, and it would be a
  finding about the cost of determinism, not about the model.
- Correctness metrics are expected unchanged: `modelAnswerRate` 0.444, `modelAnswerCorrectRate`
  1.000. A drop in `modelAnswerCorrectRate` would mean the repair changed answers, not just their
  reproducibility, and would be a defect rather than a cost.

## Decision rule

Admission requires, on one host in one session:

1. `$id.comparator.json` **and** `$id.verdict.json` both present. A candidate report alone is
   `NO_COMPARABLE_BASELINE` and is not a result.
2. `verdict == QUALIFIED` under `production-rag-publisher-reference-v1`, with
   `modelAnswerRate >= 1/3`, `modelAnswerCorrectRate >= 0.90`,
   `decodeThroughputRatio >= 0.80`, `endToEndLatencyRatio <= 1.5`.
3. A separate default-configuration smoke record under public library defaults: no `-Dmodels.*`,
   `native-quantized-decode == "true"`, `warmups == 0`, `iterations == 1`,
   `correctAnswerRate == 1`, `abstentionAccuracy == 1`, `failures` empty. The qualification run
   cannot stand in for it -- it uses a tuned decode thread count and a 0.90 correct-rate gate.

## Kill criterion

Stop and record the null, without re-running at different settings, if:

- the candidate arm fails to load the artifact or diverges from the pinned publisher continuation;
- `modelAnswerCorrectRate` falls below 0.90 (a correctness regression, not a tuning problem);
- either arm cannot complete on the host, making the pair unusable.

A `FAILED_RELATIVE_GATE` at an otherwise healthy correctness profile is **not** grounds for
re-running at a different `dt`. The source run's thread count is part of what is being reproduced;
changing it after seeing the number is moving the goalposts. Any re-run at a different `dt` is a
new, separately registered experiment.

## Inputs

Filled in before launch; the payload name freezes its commit.

| input | value |
| --- | --- |
| artifact | `Ternary-Bonsai-2-27B-PTQ1_0.gguf`, 5,946,648,928 bytes, sha256 `53107f530aa52eb00912263ab1ee29bd199261c87cd7b4ad4ca1318c1fe33ee3` |
| artifact revision | `b072e1d3b35a0a630cece372c2127528e0994386` |
| publisher reference | `PrismML-Eng/llama.cpp` at `e8fd209c975c80aa7fb36a8423571ff3b1b7e955`, CPU only, never called Ollama |
| policy | `production-rag-publisher-reference-v1`, opt-in via `--publisher-reference prism-bonsai2-ptq1_0` |
| models version | 0.5.4 (released, Maven Central) |
| vectors version | 0.1.30 (released, Maven Central) |
| product bytes | downloaded from Central and SHA1-verified by `prepare-released-payload.py`; no locally built product JARs |
| harness | `models-rag-bench`, source-built, revision recorded in `provenance.json` |

## Local gates completed before any spend

| gate | result |
| --- | --- |
| live vCPU quota read (`L-1216C47A`) | 192.0 |
| instances holding vCPUs (`running`/`pending`/`shutting-down`/`stopping`) | none |
| `qual-worker-guards-test.sh` | all guard checks passed |
| GGUF header contract, range-fetched | 5 `prism.hadamard.*` keys matched; `transform=normalized-sylvester-walsh-hadamard`, `axis=input-last-dimension`, `sign_mode` explicit with widths and values |
| weights-free math | `PrismHadamardTest`, `TernaryDequantizerTest`, `TensorOpsPtqTest`, `Qwen35QuantizedExpertsTest`, `GatedDeltaNetRecurrenceTest` all pass |
| exact-artifact oracle parity | recorded below |

## Results

Recorded after the run. Nothing above this line is edited once data exists.
