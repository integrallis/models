# Where a typed decision's time goes, and what changes its answer

Measured 2026-09-24 on one dedicated host, one protocol, both arms of every
comparison measured rather than one extrapolated. Nothing here is read from a paper
or a vendor page.

**This file was rewritten after a first pass got a central thing wrong. Read
section 0 before the numbers.**

## 0. The first pass measured the JIT

Every ad-hoc harness here was originally written without warmup. That is not a
detail: unwarmed Panama vector code takes a different path with a different
accumulation order, so an unwarmed comparison of two code paths measures which one
the compiler reached first.

It produced a confident and wrong diagnosis. Two identical calls -- one forward of
one token, run twice -- differed on **all 248,320 logits**, which reads exactly like
a state leak between sessions. It is not. Warmup 0 against warmup 3:

| comparison | warmup 0 | warmup 3 |
|---|---|---|
| two fresh sessions, one token at position 0 | max 3.810e-01 | **0.000e+00** |
| default session, reset between | max 7.863e-01 | **0.000e+00** |
| fresh session vs default session | max 1.057e+00 | **0.000e+00** |
| ten further repeats against each other | 0.000e+00 | 0.000e+00 |

**There is no determinism bug and no state leak.** The tell was in the first run all
along: the ten repeats agreed, because warmup had happened by then.

This project's working agreement already records this failure mode, from a previous
sweep it corrupted. It was walked into anyway. Every harness in `harness/` now warms
each path it is about to compare, and `harness/Determinism.java` exists to check the
premise rather than assume it.

**The bench harness was never affected.** `JevBenchRunner` warms three items and
discards them. Re-run at warmup 25, its probabilities are byte-identical to warmup 3
(`prompt-arms/shipped-w3.tsv` against `shipped-w25.tsv`, comparing every column but
latency). So every accuracy number below is warmup-clean, and so were the ones
published before.

Numbers withdrawn from the first pass: the "alone" column of `Split` (3.745 s and
5.330 s for the medium and long criteria, actually 0.621 s and 2.121 s), and the
answer-drift column of `Batch2`.

## 1. The host

Hetzner CCX33: 8 vCPU on AMD EPYC Milan, 4 physical cores with SMT, 32 GiB, Ubuntu
24.04, Temurin 25.0.3. Idle apart from the run, one measurement at a time.

Harriet, `org.modeljars.composite:harriet`, over a frozen Qwen3.5-4B at Q4_K_M,
2.55 GiB, through `RustFfmBackend` at native kernel ABI 6. Evidence is a 143-token
contract; the decision cases are `cases.tsv`; the accuracy cohort is JevBench v1.2's
public easy and original tiers, 120 items, prepared by
`options-first/prepare_tasks.py`.

## 2. A decision is compute bound (`harness/Threads`, `harness/Split`)

Prefill against token count, one batch, evidence prefilled separately:

| tokens | seconds | ms/token |
|--------|---------|----------|
| 17     | 0.315   | 18.5     |
| 30     | 0.571   | 19.0     |
| 107    | 2.064   | 19.3     |
| 143    | 2.638   | 18.4     |

Linear, no fixed cost. If reading the 2.55 GiB of weights dominated there would be a
large intercept and a flat slope.

Prefill of 143 tokens against thread count:

| threads | seconds | ms/token |
|---------|---------|----------|
| 1       | 8.649   | 60.5     |
| 2       | 5.130   | 35.9     |
| 4       | 2.965   | 20.7     |
| 8       | 2.641   | 18.5     |

It halves, halves again, then saturates on the box's four physical cores. That is
arithmetic at the ceiling. **N questions cost N questions' arithmetic however they
are arranged.**

The single-token step that ends a decision behaves the other way -- 0.167, 0.091,
0.065, 0.067 s over the same thread counts. Flat past two threads: bandwidth.

One warm decision over shared evidence, 17-token question: 0.315 s prefill + 0.083 s
final step + 0.010 s = 0.408 s, and `Split` measures 0.408 s end to end.

## 3. Grouping costs answers and buys nothing here (`harness/Batch2`)

Twenty questions over one piece of evidence, both arms measured, warmed.

With the answer-preservation gate on, which is the shipped default:

| n | one at a time | grouped | speedup | max answer drift |
|---|---|---|---|---|
| 2 | 0.818 | 0.828 | 0.99x | 0.00e+00 |
| 5 | 2.244 | 2.142 | 1.05x | 0.00e+00 |
| 10 | 4.217 | 4.221 | 1.00x | 0.00e+00 |
| 20 | 8.153 | 8.480 | 0.96x | 0.00e+00 |

Forced on with `-Dmodeljars.decisions.allowInexactGrouping=true`:

| n | one at a time | grouped | speedup | max answer drift |
|---|---|---|---|---|
| 2 | 0.833 | 1.154 | 0.72x | 7.15e-02 |
| 5 | 2.115 | 2.450 | 0.86x | 7.15e-02 |
| 10 | 4.144 | 4.460 | 0.93x | 7.15e-02 |
| 20 | 8.167 | 7.959 | 1.03x | 7.15e-02 |

So grouping is a regression below twenty questions, a wash at twenty, and it moves
every answer by 0.07 of probability. It is off by default and the backend is what
says so.

Two defects were found and fixed on the way to that answer, and both were worth
fixing regardless:

- The grouped path called `reset()` and re-read the whole evidence on every call,
  then discarded the resumption point so the next one-at-a-time decision paid again.
  The evidence read costs 2.64 s against 0.41 s for a question, so a group paid more
  for its evidence than for every question in it. Twenty questions: 10.79 s to
  8.22 s.
- The Gated DeltaNet kernel stays on the calling thread for one token of one
  sequence, which is right for decode. The grouped path called it once per branch, so
  every branch ran on one core. Handing it the whole group: 12.39 s to 10.79 s.

## 4. What actually changes an answer (`harness/Cross`, `harness/Matmul`, `harness/Grouped`)

All warmed. One projection, real Q4_K weights, the same activations:

| comparison | relative |
|---|---|
| native batch N vs Java batch N | 3e-7 |
| native batch N vs native one row at a time | 3e-7 at N>=2, 0 at N=1 |
| native, the same call twice | 0 |
| Java batch N vs Java one row at a time | **0 at every N** |

Unaffected by activation outliers up to 200x. Ordinary fp32 accumulation order.

The same differences after 32 layers, 24 tokens:

| comparison | max logit | mean | relative | argmax |
|---|---|---|---|---|
| Java batched vs Java stepped | 0.000e+00 | 0.000e+00 | 0 | same |
| Java one batch vs two batches | 0.000e+00 | 0.000e+00 | 0 | same |
| native one batch vs two batches | 0.000e+00 | 0.000e+00 | 0 | same |
| native batched vs native stepped | 3.255e-01 | 5.599e-02 | 2.5e-02 | same |
| native stepped vs Java stepped | 3.354e-01 | 5.940e-02 | 2.6e-02 | same |

So: **splitting a prefill changes nothing in either kernel**, and the Java kernel's
row result does not depend on how many rows share the call. What does differ is
crossing between the native kernel's one-row path and its many-row path -- the
`batch_size == 1` specialisations reduce one row straight to a scalar, while two or
more accumulate per-row vector lanes and reduce at the end -- and crossing between
the two kernels at all. 3e-7 on one projection becomes 3e-2 at the logits, about
five orders of amplification over the depth.

Grouped against the same question asked alone, four questions over 120 tokens of
shared evidence:

| kernel | result |
|---|---|
| pure Java | 0.000e+00 on every logit of all four |
| native | 0.28, 0.27, 0.28 max logit on three of four; the fourth exactly 0 |

That is the whole basis for `GroupedDecisionBackend.groupedDecisionsMatchSingleDecisions()`.
A lone question reads its answer out of a batch of one row and a group reads its out
of a batch of many.

A caller who needs the same bits twice gets them: the shipped path is fixed and
deterministic, and repeated identical calls agree exactly. What is not available is
agreement *across* implementations.

## 5. The noise floor, so the rest can be read (`prompt-arms/shipped-gdnnative.tsv`)

Swapping the recurrence kernel across all 24 Gated DeltaNet layers -- a real
implementation change, same prompt, same cohort:

| | shipped | recurrence kernel flipped |
|---|---|---|
| accuracy | 0.8917 | 0.8833 |
| Intelligence | 88.0 | 87.0 |
| winners differing | -- | 1 of 120 |

**One item, one point of Intelligence.** That is the unit everything below is
measured in. Logit-level divergence of 6e-2 sounds alarming and moves one decision
in 120: the probabilities move, the argmax mostly does not.

## 6. The published score was for a prompt the product could not build

> Superseded in part by section 8: the gap is real and the final number is 88.9.


JevBench hands every system a per-label rubric out of `question.criteria`.
`AnswerSpace` had two accessors, `question()` and `labels()`, so no caller could
supply one. The benchmark arm read the rubric; the shipped runtime did not.

Same model, same kernel, same 120 items, the only variable being the rubric and its
layout:

| arm | accuracy | Intelligence | Calibration | p50 |
|---|---|---|---|---|
| benchmark prompt, rubric block + lettered list | 0.8917 | 88.0 | 71.9 | 2.342 s |
| what shipped: no rubric at all | 0.7500 | **72.2** | 80.1 | 1.429 s |
| rubric inline, `A: label -- criterion` | 0.8250 | 80.8 | 78.1 | 2.036 s |
| rubric as a block above a bare lettered list | 0.8750 | **86.1** | 66.8 | 2.220 s |

Sixteen points of Intelligence between what was measured and what shipped, against a
one-point noise floor. Per family, no rubric against block: ordinal 0.2500 against
0.9167, routing 0.3333 against 0.7500. That is what `A: 3` means to a reader who was
never told what 3 is.

**Fixed:** the three answer spaces carry an optional rubric and the runtime renders
it. Section 8 supersedes the layout below and the final figure: the shipped rendering
puts the rubric before the criterion and scores **88.9**, not 86.1.

The rest of this section is the intermediate step and is kept because the ordering of
the four arrangements is the finding. The layout was measured rather than chosen -- inline was the obvious design and
scored five points worse. The shipped renderer was then re-run over the cohort and
reproduces the block arm's probabilities exactly (`runtime-renderer.tsv` against
`runtime-block.tsv`), so 86.1 is a number the product produces and not one a harness
produces.

Still 1.9 points below the benchmark prompt, which also carries a blank line and a
second answer cue. Within about two noise floors, and not chased.

Calibration moves the other way, 71.9 to 66.8: the rubric makes the model more
confident as well as more right. Temperature is the instrument for that and none was
fitted here.

## 7. The 1.76x that was not taken (`harness/Reorder`, `prompt-arms/options-first-fast.tsv`)

A decision's cost is the arithmetic of the tokens after the shared evidence, and for
a short question most of those tokens are the options -- identical across every
question with the same answer space. Moving them ahead of the criterion makes them
part of the shared prefix. Fifteen cases, warmed, per-question part only:

| | shipped order | options first |
|---|---|---|
| suffix tokens, total | 463 | 240 |
| mean seconds | 0.702 | 0.398 |

**1.76x on every decision, no kernel work.** Then scored against gold labels:
accuracy 0.8917 to 0.8083, Intelligence 88.0 to 80.1, with `fact` 1.0000 to 0.5000
and `adequacy` 0.7500 to 0.5000.

**Rejected.** Eight noise floors down, concentrated in two families rather than
spread, which is a prompt the model reads differently and not sampling. The arm stays
flagged and off, with its output and digest, so the next person to have the idea can
see it was tried.

Worth recording precisely: the Speed axis is *identical* in both arms, 64.8 either
way, because every JevBench item carries its own state and there is no shared prefix
for the options to move into. That cohort can price this change's cost and
structurally cannot price its benefit. The decision was made on the cost.

**Not re-asked after section 6.** A rubric makes the shared part of a prompt larger,
so the reordering is worth more now than when it was rejected. The accuracy objection
is unchanged and the question is open.

**Not run:** the hard tier, 111 items of about 3,700-token states, roughly 70 s an
item. It is where a 50-token options block matters least and would not have changed a
verdict already eight floors clear.

## 8. The performance release

The rubric of section 6 is worth 14 points of Intelligence and costs 53 tokens. A
decision's cost is the arithmetic of the tokens after the shared prefix, at 18.5 ms
each, so carried per question that rubric is **0.98 s of added latency on every
question** -- the fix in section 6 took a sub-second decision and made it a
second-and-a-half one.

The release is that it does not have to be per question. Which side of the shared/per
question split each part of the prompt falls on is a free parameter, and it turns out
to be the same parameter that sets accuracy.

Four arrangements of the same declarations, 120 items, warmup-clean:

| arrangement | accuracy | Intelligence |
|---|---|---|
| no rubric at all | 0.7500 | 72.2 |
| rubric and letters both **after** the criterion | 0.8750 | 86.1 |
| rubric and letters both **before** it | 0.8000 | 79.6 |
| **rubric before, letters after** | **0.9000** | **88.9** |

Moving both was a 6.5-point loss and moving only the rubric is a 2.8-point **gain** --
the best of the four, and the best number measured this week. Per family the reason is
visible: with the letters moved too, `fact` fell from 1.0000 to 0.3333, while
`ordinal` and `routing` rose to 1.0000. A model wants the letter-to-label mapping
after the question it answers, and is content to have read what the labels mean
beforehand. Two things had moved at once and only one of them was the problem.

So the shared prefix is evidence + rubric, and the per-question suffix is criterion +
letters + cue. Measured through the shipped API over one contract, 20 questions, warm
(`harness/Release`):

| rubric | shared tokens | suffix tokens | per question |
|---|---|---|---|
| none | 143 | 18 | 0.490 s |
| declared, 53 tokens | 196 | 18 | 0.486 s |

**A rubric now costs nothing per question.** The suffix is the same 18 tokens either
way, and the 53 tokens are read once for the batch instead of 20 times. Against the
0.98 s per question the naive placement would have cost, that is the release.

### The readout was a whole weight sweep for one token

A decision ended by prefilling all but its last token and then stepping that token
alone. A single token goes through all 2.55 GiB: 67 ms, and flat in thread count past
two because it is bandwidth. As one more row of a batch that is already compute bound
it is 18.5 ms.

Reading the answer off the final position of one prefill of the whole suffix:
**0.520 s to 0.485 s per question**, and it is a different kernel path so it was
scored rather than assumed harmless -- accuracy 0.9000 either way, Intelligence 88.9
either way, **0 of 120 winners differ** (`prompt-arms/F-shipped.tsv` against
`G-folded.tsv`). Free.

It does not unify the grouped path: forced on after this change, grouping still drifts
4.5e-02 and still runs 0.63x to 0.92x. The gate of section 3 stands.

### What was left on the table

The final projection computes all 248,320 logits when a typed decision needs two to
ten. That is 521 MB of the 2.55 GiB read once per decision, about 13.7 ms at this
box's measured 38 GB/s, or 2.8% of a question. It needs a new backend entry point to
project selected rows, and 2.8% does not pay for that surface. Written down rather
than done.

Per question now: 18 tokens x 18.5 ms = 0.333 s of irreducible arithmetic, plus about
0.15 s of resumption and bookkeeping, measured 0.486 s. The arithmetic is at the
machine's ceiling; the 0.15 s is where any further work belongs, and most of it is the
~100 MiB of recurrent state a resumption restores per question.

## 9. Release scoreboard

| change | measured | verdict |
|---|---|---|
| answer spaces carry a rubric | Intelligence 72.2 to 86.1 | kept |
| rubric shared, letters after the criterion | 86.1 to **88.9**, and free per question | kept |
| readout folded into the prefill | 0.520 s to 0.485 s, 0 of 120 answers moved | kept |
| project only the answer letters | 2.8% for a new entry point | not done |
| grouped recurrence in one kernel launch | 12.39 s to 10.79 s at n=20 | kept |
| group resumes its evidence | 10.79 s to 8.22 s at n=20 | kept |
| gate grouping on answer preservation | drift 7.15e-02 to 0.00e+00 | kept |
| grouping at all, native backend | 0.72x to 1.03x | off by default |
| grouping at all, no fast decode kernel | 1.25x at n=2, 1.69x at n=20 | on |
| options ahead of the criterion | 1.76x, minus 8 Intelligence | rejected |

## 10. The gap to a hosted System One service

A service answering in 5 to 10 ms of compute is not batching better. It is doing far
less arithmetic. Ours is at this machine's ceiling, so the levers are fewer tokens per
question, a smaller model, or more FLOPs -- not scheduling. Three days of scheduling
work to establish that, and the largest single win of the week came from noticing that
a label is a token and not an explanation.
