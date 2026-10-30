# Result: VibeThinker 1.5B fails the `math` contribution gate in all six cells

Measured here, on a Hetzner `ccx33` (8 dedicated x86_64 vCPU, AVX2, JDK 25.0.5 Temurin), on the
`rust-ffm` route with the qualification fleet's own payload and kernels jar
(`models@0.5.7+vibethinker1-ec3ae4ff2fd3`, kernel sha256 `cdc7e877c386`). Artifact sha256
`59fa32aaaf5c…`, verified on the host to be the same GGUF shard 703 measured.

Every number below is produced by `derive.py` and `PolicyReplay.java` from the reports in
`evidence/`. Nothing is hand-typed.

```
python3 experiments/vibethinker-template-ablation/derive.py
CP=$(ls models-rag-bench/build/install/models-rag-bench/lib/*.jar | tr '\n' ':')
javac -cp "$CP" -d /tmp/replay experiments/vibethinker-template-ablation/PolicyReplay.java
java -cp "$CP:/tmp/replay" PolicyReplay experiments/vibethinker-template-ablation/evidence/*.json
```

## The grid

`truncatedAnswerRate` / **`modelAnswerRate`** / `modelAnswerCorrectRate`, 9 cases per cell:

| cap | `chatml-direct` | `chatml-answer` | `chatml-no-think` |
| --- | --- | --- | --- |
| **768** | 1.000 / **0.000** / 0.000 | 1.000 / **0.000** / 0.000 | 0.333 / **0.000** / 0.000 |
| **1792** | 0.778 / **0.111** / 0.000 | 0.889 / **0.000** / 0.000 | 0.222 / **0.000** / 0.000 |

Gate: `modelAnswerRate >= 0.333` **and** `modelAnswerCorrectRate >= 0.90`. No cell clears either
term. `modelAnswerCorrectRate` is 0.000 in all six cells, including the one cell that produced an
accepted answer at all.

Total output tokens, same grid:

| cap | `chatml-direct` | `chatml-answer` | `chatml-no-think` |
| --- | --- | --- | --- |
| **768** | 6912 | 6912 | **3296** |
| **1792** | 15868 | 15211 | **5491** |

## What the template does, and does not, buy

`chatml-no-think` -- the empty closed `<think>\n\n</think>` prefill -- is the one lever that moves
anything. It cuts truncation from 1.000 to 0.333 at cap 768 and from 0.778 to 0.222 at 1792, and it
cuts output tokens by 2.1x and 2.9x respectively. On the qualification question it buys nothing:
`modelAnswerRate` stays at 0.000.

That those are real tokens for this model was checked before the arm was included: `<think>` is
151665 and `</think>` is 151666 in this GGUF's 151936-token vocabulary.

## Why it fails, measured rather than inferred

Under `chatml-no-think` at cap 1792, **7 of 9 generations finished on their own** -- several in
101-256 tokens -- and none was accepted. So for those cases the cap is not the binding constraint,
and the reason had to come from the policy itself.

`PolicyReplay.java` re-applies `GroundedAnswerPolicy.productionDefault()` to every recorded
generation, using the question and retrieved documents out of the report, and **fails if it does
not reproduce the decision the report recorded** -- the first version passed the case id in place
of the question, changed three decisions, and would have been reporting a different policy than the
one that ran.

Across the six admissible cells, of the **16 generations that finished before the cap, 1 was
accepted as produced, and 5 would be accepted with the model's shown working removed.**

The mechanism, on `amber-determinant-trace` (177 tokens, finished, rejected):

> The determinant of the Amber matrix [[3, 1], [2, 4]] is calculated as (3\*4) - (1\*2) = 12 - 2 =
> 10. The trace is the sum of the diagonal elements, 3 + 4 = 7.

The retrieved document says only `the determinant is 10. The trace is 7.` The derivation is not in
the evidence, so `hasOnlySupportedClaims` rejects it and the pipeline serves the extractive answer
instead. Handing the policy the same answer with the working removed -- `The determinant is 10 and
the trace is 7.` -- returns `MODEL_ANSWER_WITH_DERIVED_CITATIONS`. The answer was never the
problem.

The same shape appears as commentary rather than arithmetic: `"This matches the given context"`,
`"The answer should directly use the provided mean and median values"`.

**So this model fails `math` because it shows its work.** A grounding policy that accepts only
claims supported by the retrieved text cannot accept a derivation, and a model post-trained to
derive cannot comply without suppressing what it was trained to do. That is a statement about the
pairing, not about the model's arithmetic -- which was right.

## The one accepted answer was wrong

`chatml-direct` at 1792 produced the grid's only `MODEL_ANSWER`, on `delta-sixth-term`:
`factCoverage` 0.5, `citationRecall` 1.0, `citationPrecision` 1.0, `correct: false`. It cited
correctly and covered half the required facts. So across 54 generations, one was accepted and it
was incorrect.

## Verdict, and what the registered rule can and cannot support

Two separate statements, kept separate on purpose.

**The gate is failed, measured.** `modelAnswerRate` is 0.000 in five cells and 0.111 in the sixth,
against a floor of 0.333; `modelAnswerCorrectRate` is 0.000 in all six. That is not a close call
and it does not depend on any interpretation.

**The registered decision rule returns INCONCLUSIVE on *why*.** The rule asks truncation first, and
truncation never fell below 0.05 in any cell -- 0.222 is the best any template reached. Its
remaining branch says "needs a higher cap", and there is no higher cap: 1792 is the ceiling the
harness can run. So the pre-registered rule cannot, by itself, attribute the failure to the model
rather than to the cap. `derive.py` prints exactly that, and does not print a selected template.

**The attribution comes from a post-hoc refinement, declared as such.** The per-case replay was not
in the registered protocol. It observes that under `chatml-no-think` at 1792, 7 of 9 generations
finished before the cap, and that the policy accepted none of them -- so for that subset the cap is
demonstrably not the binding constraint, and the shown working is. This is the stronger evidence,
and it is weaker provenance: it was chosen after seeing the grid. It is reported as a mechanism
worth believing, not as a pre-registered result.

**Operationally:** shard 911 is not run as a qualification, and no `fleet-shard-911.json` is
committed -- a runnable manifest for a run we decided against is a trap, and a future attempt
should be authored against whatever the grounding policy is then.  A two-arm fleet box would spend money
to re-confirm a gate already failed by a wide margin in six cells on the production route and the
production build. Qualifying this model on `math` is not blocked on more measurement of the same
kind; it is blocked on a grounding policy that can accept a derivation, which is a policy change
affecting every model and not a per-model rescue.

## What this does not say

- It does **not** say VibeThinker 1.5B is a weak model. Its arithmetic was correct in the cases
  inspected; what the gate rejects is the derivation accompanying it.
- It does **not** establish, from the registered rule alone, that the cap is irrelevant. The rule
  is inconclusive on that; the per-case replay is what rules the cap out, and it is post-hoc.
- It does **not** say reasoning models cannot qualify. It says this grounding policy
  (`bounded-context-injection-screened-citation-safe-statement-grounding-v23`) and a
  show-your-working model are a poor pairing on this corpus.
- It says nothing about other workloads. `math` was chosen for this model; a corpus whose documents
  contain derivations rather than bare results would test something different, and has not been run.
- The `answerOnly` stripping in `PolicyReplay.java` is a **diagnostic, not a proposal**. It is a
  crude line filter whose only job is to show which rejections are attributable to the working. It
  is not a post-processor anyone should ship, and the 5-of-16 figure is not a projected score.
- Run-to-run text stability was not tested: every cell is `iterations = 1`.

## Two harness defects found on the way

1. **A cap at or near the context is unrunnable.** `--max-tokens 2048` with `--context 2048`
   failed with `position out of range: 2048` on 7 of 9, 8 of 9 and 2 of 9 cases. Prompts here are
   156-172 tokens, so against a 2048 context the ceiling is 1876. Those three cells are in
   `evidence-rejected/`, excluded rather than reported, because the cases that failed are the
   long-generating ones and the surviving attempts are the short ones -- every rate over them reads
   better than the truth. `derive.py` now refuses any cell whose `successfulAttempts` is below its
   `totalAttempts`.

   The worker pinned `--context 2048` in two places, so no shard could ever have run a cap near
   2048. It now has one `CONTEXT` variable and a `cap_fits_context` guard that skips such a job with
   `CAP_EXCEEDS_CONTEXT` instead of burning a box, with cases in
   `scripts/fleet/qual-worker-guards-test.sh` verified to fail on reintroduction.

2. **`mt: 64` on shard 910 was a known-bad value.** It is documented as such at
   `models-rag-bench/src/main/java/com/integrallis/models/rag/RagBenchmarkCli.java:164` -- "a
   thinking model spent the whole budget on its reasoning trace" -- and the default is 256. Shard
   910 measured that cap, not the model, and is superseded by this experiment rather than reported.
