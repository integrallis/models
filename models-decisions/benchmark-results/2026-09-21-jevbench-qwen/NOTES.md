# JevBench on Qwen3-0.6B: measured position, and a mechanism that does not transfer down

All 231 public items (easy 48, hard 111, original 72), one dedicated ccx33, rust-ffm kernel,
`prefillBatchSize=512`, `maxContextLength=4096`. Every axis below comes from the same arm; the
scoring is JevBench's own `composite_v12` formulas applied to our measurements.

## Position

| axis | value | basis |
| --- | ---: | --- |
| Intelligence | 49.4 | accuracy 0.4935 over 231 items |
| Calibration | 75.7 | ECE 0.1218 after a temperature fitted on easy+original, applied to hard |
| Speed | 61.2 | p50 0.778 s, p95 21.99 s, their CPU adjustment x2 + 0.15 s |
| Cost | 53.9 | 4.691 s/decision at Laya's implied CPU rate, doubled for twice the threads |
| **JevBench Score** | **59.3** | Laya 70.1, jeff 66.9, Verdict 66.2, GLiNER2 53.0 |

Accuracy is not uniformly weak. Per family: tool_selection 0.917, intent 0.875, routing_hard 0.800,
extraction 0.667 -- against ambiguous 0.000, trap 0.125, temporal_numeric 0.200, routing 0.250. The
base is competent at reading and matching, and absent on reasoning-hard families.

## Letter-logit scoring: implemented, measured, rejected

SemIf sits at #2 with 74.7 using "one forward pass per decision, no generation; softmax over the
declared answer-letter logits" -- options rendered as a lettered list in the prompt, the answer read
off the logits for `A`, `B`, `C` at the answer position. It looked like it should move three axes at
once: one forward instead of 4.4, and the option text read from the prompt rather than learned.

Implemented as `LetterLogitScorer` (7 tests: normalises over the letters alone even when an
unrelated token dominates the vocabulary, order-equivariant under permutation, temperature
monotonic, refuses label/letter mismatches and out-of-vocabulary tokens). Run as a flagged arm over
the same 231 items:

| | candidate scoring | letter-logits |
| --- | ---: | ---: |
| Intelligence | **49.4** | 40.7 |
| Calibration | **75.7** | 57.8 |
| Speed | 61.2 | **62.4** |
| Cost | 53.9 | **54.7** |
| **Score** | **59.3** | **53.2** |

**Rejected.** It is 6 points worse. Two predictions were wrong and both are worth keeping:

*The speed gain was predicted "substantial" and measured 6%* (mean 4.688 -> 4.398 s). Candidate
forwards were never the cost: the run log reports 1014 forwards for 231 tasks because the first
token of each candidate is free from the prefix logits. Prefill dominates, and the letter arm
prefills the same state plus a longer prompt. That number was in the log an hour before the
prediction was made.

*Accuracy fell 0.4935 -> 0.4069.* Asking the model to emit a letter adds an indirection -- bind `B`
to "shipping", then produce `B` -- where scoring `" shipping"` against the context asks something it
does natively. SemIf runs this on Qwen3.5-**4B**. The mechanism that puts it at #2 does not transfer
down to 0.6B, which is a fact about scale rather than about the implementation.

The scorer stays in the tree behind `-Ddecisions.letterLogits`, off by default, because it is the
right mechanism at a larger base and the measurement should be repeatable when we have one.

## What actually moved the score

| change | effect |
| --- | ---: |
| held-out temperature scaling | Calibration 47.8 -> 75.7, score +7.1 |
| `prefillBatchSize=512` | p95 -17%, score +0.2 |
| letter-logit scoring | **-6.1, rejected** |

Temperature scaling is the whole gain so far, and it is post-processing over probabilities we had
already produced.

## Corrections made while measuring this

An estimate of 71.5 was published earlier from Granite 4.1 3B's accuracy combined with Qwen3-0.6B's
latency -- two different models in one figure. The Speed axis was first computed from mean latency
(5.51 s) where the formula specifies p50 and p95 (0.778 s), a 7x error. Cost was quoted at 81.9 from
a borrowed class estimate before being derived from our own throughput at 53.9.

A context-truncation arm was discarded: with `maxContextLength=1024` the runner throws
`prompt exceeds context length: 3832` rather than truncating, so it died at item 49 having written
48 easy items that read 0.8333.

## What the remaining gap is made of

Speed and Cost both derive from 4.691 s/decision against Laya's 0.79 s, and llama.cpp does
2.36 ms/token on this box where we do 5.18 -- a measured 2.2x of headroom that would move both axes.

Intelligence at 49.4 is the larger deficit and the one no setting reaches. The families we fail are
reasoning families, and an adapter trained on classification corpora will not install reasoning.
The honest routes are a larger base -- SemIf and open-alternative-jev both use Qwen3.5-4B at ranks
2 and 5 -- or a purpose-built decision model rather than an LM with a head.

## Scoring techniques from the 2021 calibration literature: one wins, four do not

The diagnosis that motivated all of these: on the 74 binary items we predicted **yes 71 times**
against a 35/39 gold split, and a content-free probe puts P(yes) at **0.8688** with no content at
all. That is textbook surface form competition (Holtzman et al., EMNLP 2021, arXiv 2104.08315):
`" yes"` is simply a likelier string than `" no"` after `Answer:`, and raw likelihood scoring lets
the prior swamp the evidence. Binary items are 74 of 231, so this is a third of the benchmark
answered by a token frequency.

| technique | held-out effect | verdict |
| --- | --- | --- |
| temperature scaling, fitted easy+original | ECE 0.2609 -> 0.1218, **score +7.1** | **kept** |
| `prefillBatchSize=512` | p95 -17%, score +0.2 | kept |
| PMI / content-free probe, alpha=1 | +0.009 on hard, +0.043 overall | marginal |
| letter-logit scoring (SemIf's mechanism) | score 53.2 against 59.3 | rejected |
| contextual calibration from item marginals | +0 of 111 held out | rejected, was in-sample |
| option-set prior shipped as a constant | -0.009 on hard | rejected, does not transfer |
| rubric-text candidates | -0.009 routed; 0.4026 overall | rejected |

### The pattern underneath every rejection

**Nothing fitted on easy+original transfers to hard.** Four independent attempts:

* PMI strength: the fit split selects alpha 1.00 and gets +0.009 on hard; alpha 1.75 would give
  +0.090 on hard but is only visible by looking at hard, so it cannot be claimed.
* A per-option-set bias learned on the fit split makes hard *worse*.
* The best scoring target by question type, decided on fit, makes hard worse.
* The temperature is the one exception, and calibration is a monotone transform that cannot move
  an argmax -- it adjusts confidence, not answers.

The two tiers differ in kind rather than in difficulty. Anything that adjusts *which* answer wins,
tuned on the easy tiers, is fitting noise with respect to the hard tier.

### Rubric scoring, the informative failure

Scoring the criterion text instead of the bare label did exactly what it was designed to do and
still lost overall:

| | bare labels | rubric text |
| --- | ---: | ---: |
| yes/no prediction balance | 71 / 3 | **56 / 18** (gold 35 / 39) |
| noul accuracy | 0.4865 | **0.5270** |
| hard tier | 0.3514 | **0.3874** |
| easy tier | **0.8333** | 0.5000 |
| score items | -- | 0.2222 |
| overall | **0.4935** | 0.4026 |

It defeats the token prior where the prior is the problem, and adds noise where the labels were
already discriminative. Routing by question type looked like the fix and does not survive
held-out evaluation, for the same reason as everything else above.

## Position, and what post-processing cannot reach

~59.6 with the temperature fix and cached probe. The remaining deficit is capability on the hard
tier, where we score **0.351** against Laya's 0.341, SemIf's 0.595 and Jev's 0.741.

The hard tier is the only one that discriminates: every ranked system scores 94-100% on easy and
73-99% on standard. It is also well defended -- 220 items, half authored by Claude Opus 5 and half
by GPT-5.6 Sol, cross-reviewed blind and against gold, "frozen and hashed before any benchmarked
system saw an item. No item was selected on any system's answers." The one caveat worth recording
is that the items are LLM-generated and LLM-adjudicated, which may favour systems that reason the
way frontier models do. That is a property of the instrument, not gaming by entrants.
