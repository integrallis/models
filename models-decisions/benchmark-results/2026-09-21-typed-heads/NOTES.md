# First Choice and Score heads: one works, one does not

Qwen3-0.6B Q4_K_M, frozen final hidden state, multinomial head fitted by LogisticHeadTrainer,
temperature on a held-out calibration split, sealed split read once through the released artifact.

## Choice works

AG News, four published labels. Measured at two sizes, and the larger one is the honest figure:

| n harvested | sealed | accuracy | floor | ECE |
| ---: | ---: | ---: | ---: | ---: |
| 400 | 80 | 0.9250 | 0.3125 | 0.0660 |
| **1000** | **200** | **0.8500** | 0.2650 | 0.0827 |

More data gave a *lower* number. Eighty sealed items carry roughly a +/-0.06 interval, so 0.9250
and 0.8500 sit about one standard error apart; the first was small-sample optimism rather than a
better model. **0.8500 on 200 sealed items is the number to quote**, and it still clears its floor
by 0.585.

The L2 sweep below is at n=400, 80 sealed.

| L2 | accuracy | ECE |
| ---: | ---: | ---: |
| 0.1 | 0.9250 | 0.0989 |
| 1.0 | 0.9250 | 0.0858 |
| **5.0** | **0.9250** | **0.0660** |
| 9.0 | 0.7750 | 0.2929 |
| 9.9 | 0.3875 | 0.1478 |

Majority floor 0.3125, so 0.9250 clears it by 0.61. Accuracy is flat in L2 across two decades while
calibration improves monotonically, then both collapse past 9.0 as the weight decay factor
`(1 - learningRate * l2)` approaches zero -- the regime the trainer's divergence guard bounds at 10.

**A frozen-state linear head does four-way classification well.** That is the primitive JevBench's
routing family needs, and routing has been scoring 0.333 -- chance -- for want of it.

## Score does not, and the reason is granularity

SST-5, five published ordinal levels, n=600 harvested, 120 sealed. Accuracy 0.3083 at L2=1.0 and
0.1917 at L2=5.0 against a 0.2667 floor. Neither beats the floor. Ordinal MAE about one whole level,
which is a head that is guessing.

Two explanations were available -- a bad harvest prompt, or the task being beyond a frozen probe.
Both were tested.

**The prompt was exonerated by re-harvesting.** A second harvest of the same 600 items replaced the
bare `\n\nSentiment:` suffix with an explicit instruction naming the scale: "Rate the sentiment of
this review from 0 (very negative) to 4 (very positive)." It scored **worse**:

| harvest prompt | accuracy | floor | ordinal MAE |
| --- | ---: | ---: | ---: |
| `\n\nSentiment:` | 0.3083 | 0.2667 | 0.9746 |
| explicit "rate 0 to 4" | **0.2333** | 0.2667 | 0.9988 |

A better instruction does not create resolution that is not in the state.

**Granularity is the cause.** The same harvested states were relabelled at coarser resolution.
Nothing but the number of levels changed.

| levels | sealed | floor | accuracy | margin over floor | ordinal MAE |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2 (neg / pos, neutral dropped) | 89 | 0.5169 | 0.7865 | **+0.270** | n/a |
| 3 (neg / neutral / pos) | 120 | 0.3833 | 0.4750 | **+0.092** | 0.7304 |
| 5 (as published) | 120 | 0.2667 | 0.3083 | +0.042 | 0.9746 |

The margin decays monotonically and crosses below the 0.05 sanity threshold between three and five
levels. Four experiments converge on this: two prompts, three granularities, and an L2 sweep at
each -- none of which moves five-level accuracy above its floor, while a binary head on the very
same states reaches 0.7865. So the state **does** carry sentiment: a binary head reads it at 0.7865 against a 0.5169
floor. What it does not carry is five resolvable ordered levels.

This is a property of the design rather than a defect. A linear read of a single frozen hidden state
resolves roughly two to three ordinal levels on this corpus, and asking for five returns noise.

**Consequence.** JevBench's score family uses four levels, which is past where the margin has
already collapsed. On this architecture Choice is viable and Score is not. Closing Score needs
something the frozen probe does not offer -- a trained adapter that reshapes the representation, an
ordinal-aware loss rather than plain multinomial, or a head reading several layers instead of one.

## What is not claimed

AG News topics and SST-5 sentiment are not JevBench's routing and adequacy tasks. These measure
whether the primitive can be fitted at all, on published data nobody here authored. Whether the
result transfers to the cohort's families is a separate question these corpora cannot answer, and
it stays open until it is measured there.

n=400 and n=600 are small, and 80 and 120 sealed items carry wide intervals. The larger harvest is
still running; these are the numbers available now, not final ones.

## Process

The `beatsFloor` condition fired on real data exactly as its synthetic test predicted. Without it,
a Score head at 0.3083 would read as a weak result rather than as no result -- which is how the
CUAD head's 0.5600 against a 0.7200 floor once read as progress.

## The fixed-label head cannot serve JevBench, and the alignment head did not rescue it

JevBench's 36 choice items use **three** distinct option sets, twelve items each:

| option set | size |
| --- | ---: |
| `cancel, refund, status, change_address, other` | 5 |
| `courier, pickup, post, unknown` | 4 |
| `math, coding, coding_agent, document, tools, general` | 6 |

Its twelve score items share one set, `0,1,2,3`.

A multinomial head learns *those* labels, so an AG News head has nothing to say about any of them,
and the benchmark's own items cannot be trained on. The head measured above therefore serves a
customer who owns a fixed taxonomy and has labelled data for it -- not a benchmark that supplies a
new option set per question.

`CandidateScorer` exists for exactly this and was built to be label-agnostic: score each candidate
independently against the state, normalise within the question, so "the ratio between two
candidates is unaffected by the presence of a third". It needed a trained `Alignment`.

One was trained. Features are the elementwise product and absolute difference between a state and a
candidate's encoded label text -- nothing in them names a label, so an unseen label is scored on the
same footing as a trained one. Trained on AG News interactions alone, 3,200 pairs of width 2,048:

| option set | | accuracy | floor |
| --- | --- | ---: | ---: |
| agnews | trained | 0.6100 | 0.2650 |
| jev_ship | held out | **0.1800** | 0.2650 |
| jev_intent | held out | **0.1750** | 0.2650 |
| jev_route | held out | **0.1800** | 0.2650 |

**All three held-out sets fall below the floor.** That is not a weak signal to be tuned up, it is no
signal. The head also reached only 0.6100 on the set it trained on, against 0.8500 for the
fixed-label head on the same items, so interaction features over a frozen state carry less than
direct class weights.

**A flaw in this experiment, stated rather than buried.** The held-out sets were scored against AG
News *items*, whose true label is a news topic with no counterpart in a shipping vocabulary. Asking
which of `courier / pickup / post / unknown` fits a sports report is incoherent by construction, so
this run cannot separate "the mechanism does not transfer" from "the question was meaningless".
What it does establish is that this head, trained this way, transfers nothing usable. A clean test
needs items whose true labels live in the held-out set.

**Where this leaves the architecture.** Our Choice head is real and works at 0.8500 for a fixed
taxonomy with training data. Jev reads its option set from the instructions and needs none. That is
a narrower product, and JevBench's choice family stays unaddressable until either per-option-set
training data exists or a head is built that genuinely generalises across option sets -- which the
OpenJev recipe achieves with a LoRA adapter reshaping the representation, not with a probe over a
frozen one.
