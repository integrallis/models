# Process failure: fourteen models reported as qualified without the qualification step

Written 2026-09-30, after the question "how did models qualify before and now are not?"

## The short answer

**The process did not fail. It was not run.** Qualification in this repository is a five-step pipeline
ending in a tool whose only job is to produce a verdict. The 2026-09-29 decoder campaign ran step one,
fourteen times, and reported its output as a verdict.

## Evidence that the process itself is sound

Every entry in ModelJars' `catalog/qualifications.json` carries the two model-contribution rates the
gate thresholds. Checked against the gate as it stands today -- `modelAnswerRate >= 1/3` and
`modelAnswerCorrectRate >= 0.90`:

| entries | marked qualified | pass the gate | marked qualified but failing the gate |
|---|---|---|---|
| 35 | 34 | 34 | **0** |

The one entry that fails, `h2oai_h2o_danube3_500m_chat_gguf_q4_k_m` at an answer rate of 0.111, is
recorded with `qualified: false`. It is the manifest's single `rejectedModels` entry. **The gate was
applied, and it rejected a model on exactly the criterion the fourteen failed.** No published
qualification is in question.

The certified evidence is also enforced in CI, not only asserted once. `CertifiedRagEvidenceTest` names
what each record had to clear: `cachedRustFfmQualifiesAgainstTheSameHostOllamaControl`,
`qwen3RustFfmQualifiesAgainstBothExactArtifactControls`,
`qwen25Coder15BQ8RequiresRustPrefillToClearTheOllamaGate`. Every one of those qualified **against
llama.cpp or Ollama comparator arms measured on the same host**.

## The catalog does not shrink, and nothing here shrinks it

Stated separately because it is the first thing to check when a qualification claim is withdrawn.

**No entry was removed, downgraded, or reinterpreted.** modeljars.org served 47 qualified entries before
this and serves the same 47 now. What was withdrawn was a claim that fourteen *additional* models had
qualified -- models that were never in the catalog. They exist in `catalog/models.json` as candidates,
where 128 candidates sit awaiting evidence, and they stayed there. The count did not go from 61 to 47;
it was 47 throughout, and 61 was a number I had projected on the strength of the wrong verdict.

Three independent checks that the existing 47 are untouched:

- All 34 entries marked qualified pass the contribution gate as it stands today, recomputed from their
  own stored rates.
- `CertifiedRagEvidenceTest` re-asserts every certified record's verdict against the current policy in
  CI: **34 tests, 0 skipped, 0 failures**, green after every change described below.
- The only ModelJars change proposed is PR #192, which edits exactly one line of
  `gradle.properties` and **zero files under `catalog/`**.

Two changes made today alter measured behaviour, and both are versioned precisely so that no existing
record is reinterpreted. Reasoning-trace handling became grounding policy **v21** rather than an edit to
v20, so records written under v20 keep their meaning. The output-token cap default changed, and every
report stores the cap it ran with. Neither touches a stored verdict. The one change to
`RagProductionQualificationPolicy` itself -- the tier gate becoming a blocklist of real failures rather
than a whitelist of fast tiers -- can only admit models, never exclude them, and the contribution gate
was not touched at all.

## The designed pipeline, and what the campaign actually did

| step | tool | campaign |
|---|---|---|
| 1. measure our arm | `RagBenchmarkCli` | **ran, 14 times** |
| 2. measure comparator arms on the same corpus and host (llama.cpp / Ollama) | `RagBenchmarkCli` | **never ran** |
| 3. assess candidate against comparators | `RagQualificationCli` -> `RagProductionQualificationPolicy.assess` | **never ran** |
| 4. commit reports and hashed evidence; CI asserts the verdict | `CertifiedRagEvidenceTest` | **never ran** |
| 5. contribute the entry with its evidence | ModelJars `catalog/qualifications.json` | **never ran** |

Step 3 is the whole point. `assess` returns `FAILED_MODEL_CONTRIBUTION_GATE` when the model did not
supply enough of its own answers, and `RagQualificationCli --require-qualified` throws on a failure. Run
against the campaign's own fourteen reports it returns that verdict for nine of them -- so the control
that existed would have caught this immediately, on the evidence the campaign already had.

## The four specific errors, in order

1. **A tier was read as a verdict.** `RagBenchmarkCli` emits `performanceTier`, computed by
   `RagPerformancePolicy`, which answers "is this fast enough and not broken". It is not a
   qualification and nothing in the report says so. `PERFORMANCE_REDUCED` was reported as though it
   meant "qualified, slower".
2. **`correctAnswerRate` was read as model quality.** It scores the pipeline. Under this grounding
   policy an answer failing citation screening is replaced with text extracted from the retrieved
   document, and the replacement is scored. 69% of the campaign's 378 attempts were answered that way.
3. **No comparator arms were run at all.** Qualification is comparative by design. The campaign
   measured one arm and compared it to nothing, which cannot produce a verdict even in principle.
4. **The claim was published before the pipeline finished.** It reached two release notes and their
   evidence files while steps 2 to 5 had not happened.

## What made error 2 possible, beyond the mistake

`RagBenchmarkSummary` published `correctAnswerRate` and withheld `rawCorrectAnswerRate`,
`modelAnswerRate`, `modelAnswerCorrectRate` and `extractiveFallbackRate`. Those existed only inside
`runs[]`. Every consumer of a summary -- a report, a release note, a person reading either -- saw the
pipeline's score with nothing beside it to suggest the model had not produced the answer. The policy was
never wrong; the summary gave no hint that it needed to be consulted.

## Controls that existed, and why none fired

| control | why it did not fire |
|---|---|
| `RagProductionQualificationPolicy` contribution gate | never invoked; it lives in a separate CLI |
| `RagQualificationCli --require-qualified` | never invoked |
| `CertifiedRagEvidenceTest` in CI | only guards records that are committed; none were |
| ModelJars build validation of `qualifications.json` | only guards contributed entries; none were |
| `defaultConfigurationSmoke` review requirement | only applies at contribution time |

Every control sits at or after **contribution**. Nothing sits between "a benchmark produced numbers" and
"a person calls those numbers a qualification", which is exactly where this went wrong.

## Changes made so it cannot recur the same way

- The four contribution rates are now part of `RagBenchmarkSummary` and the CLI's summary line, beside
  `correct=`, so the pipeline's score can no longer be read alone.
- `truncatedAnswerRate` too: a truncated answer cannot carry a citation, so it fails screening and is
  replaced, which is indistinguishable in the old summary from a model with nothing to say.
- The output-token cap defaults to 256 rather than 64, which is what truncated most of the campaign.
- `RagStatisticsTest` pins the exact failure: a perfect `correctAnswerRate` beside a 1/9 model-answer
  rate and an 8/9 fallback rate, with a closing assertion that this is below the qualification floor.
- The benchmark CLI now states in its own output that a tier is not a verdict and names the tool that
  produces one.

## The gap that remains, and it is a real one

Steps 2 through 5 are manual and undocumented as a sequence. Nothing forces comparator arms to exist
before a candidate report is treated as meaningful, and the fleet harness that ran this campaign has no
notion of a comparator at all -- it runs one arm per model and uploads the result. A campaign can still
produce fourteen candidate reports and no verdicts, exactly as this one did. The reporting is now honest
about what those reports are; making the pipeline refuse to stop halfway is not done.
