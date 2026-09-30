# How much accuracy is available, and what it costs

Measured 2026-09-24 on the dedicated CCX33 that every other figure here uses. Cohort is
JevBench v1.2's public easy and original tiers, 120 items, temperature 1.0, nothing
fitted. Baseline is the shipped 0.3.46 composition with the native recurrence:
**accuracy 0.8917, Intelligence 88.0**.

The noise floor for this cohort is one item, or about one point of Intelligence,
established in NOTES.md section 5 by swapping a kernel and changing nothing else.

## What worked

| change | Intelligence | cost |
|---|---|---|
| a rubric at all | 72.2 -> 88.0 | none, once it is in the shared prefix |
| rubric before the criterion, letters after | 86.1 -> 88.9 | none |

Those are the release, and together they are the whole of the accuracy story so far.
Both are in NOTES.md sections 6 and 8.

## What did not work

| change | Intelligence | verdict |
|---|---|---|
| read the labels' own logits, no lettered list | 88.9 -> 78.6 | rejected, 10 floors |
| rubric and letters both before the criterion | 88.9 -> 79.6 | rejected, 9 floors |
| reverse the label order | 88.0 -> 87.3 | rejected, within noise |
| add an instruction line | 88.0 -> **82.9** | rejected, 5 floors |

And one that worked, at a price:

| change | Intelligence | cost |
|---|---|---|
| average both label orders | 88.0 -> **89.8** | about 2x per question |

The instruction line is the one worth dwelling on. It was `Answer with the letter of the
option that applies.`, it sits in the shared prefix so it is free per question, and it
looked like an obvious improvement to a lettered multiple choice. It costs five points
of Intelligence. Calibration goes the other way, 73.0 to 85.0, so the model becomes
much better calibrated and noticeably worse at answering -- which is a reminder that
those two axes are not the same thing and that a prompt change has to be scored on both.

**Label order is not neutral even though reversing it nets out to noise.** Per family
the swings are large and they point in opposite directions: reversed, `extraction` falls
1.0000 to 0.8333 and `policy` 0.7500 to 0.5833, while `routing` rises 0.8333 to 1.0000
and `adequacy` 0.5000 to 0.7500. So there is a real position bias of a few items per
family; the shipped order simply happens to be on the right side of it slightly more
often than not. Picking the better order per family would be fitting to the benchmark
and is not on the table.

That bias pointing different ways per family is exactly what averaging the two orders
removes, and that is the one lever left that is not free.

## What it costs to go further

**Averaging both label orders.** Score the question with the labels forward and
reversed, average the two distributions. It removes the position bias rather than
guessing which way it points. It needs a second pass over the question's own tokens, so
it roughly doubles the per-question cost -- the shared prefix is read once either way.

Measured, arm `prompt-arms/T3-averaged.tsv` against `S-gdn-true.tsv`:

| | baseline | both orders averaged |
|---|---|---|
| accuracy | 0.8917 (107/120) | **0.9083 (109/120)** |
| Intelligence | 88.0 | **89.8** |
| Calibration | 73.0 | 69.2 |
| Speed | 66.9 (p50 1.879 s) | 61.1 (p50 3.713 s) |

**It works: +1.8 Intelligence, nearly two noise floors, and the best number measured on
this cohort.** It also more than pays back the one item the native recurrence costs, so
averaging plus the fast recurrence beats the slow recurrence alone -- 89.8 against 88.9.

The families move the way the bias predicted: `routing` 0.8333 to 1.0000 and `adequacy`
0.5000 to 0.6667, the two that preferred the reversed order, while `policy` and
`extraction` give back a little.

Calibration falls, which is what averaging two distributions does to confidence, and the
p50 roughly doubles as expected because the question's own tokens are read twice.

**Not made the default**, because it would spend the release's latency gain and more. It
belongs behind a caller's own choice: a decision worth 1.8 points of accuracy and twice
the time is a judgement about the decision, not about the runtime.

**Q8_0 weights.** MEASURED in an earlier session on the full 231-item cohort: Intelligence
78.3 against 73.3 for Q4_K_M, so about five points. It roughly doubles the weight bytes,
which is the thing a decision's arithmetic is proportional to, so it is the most
expensive accuracy on offer and it was rejected once on the composite. If accuracy is
the only axis that matters it is the largest single lever known.

**A fitted temperature** is deliberately not here. It moves Calibration and cannot move
Intelligence, because it is monotone and cannot change an argmax. Fitting one on this
benchmark is something this project has refused before and the refusal stands.
