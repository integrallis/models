# JevBench v1.2 scores computed with the harness's own functions

`jevscore.py` imports `jevbench.composite_v12` and `jevbench.metrics` from the MIT harness rather
than retyping any formula. Three numbers in this project were assembled by hand and were wrong
(Speed from a mean instead of p50/p95, a Cost borrowed from Jev's tariff, an Intelligence taken
from one model with another's latency). This removes that class of error.

Run `python3 jevscore.py RESULTS.tsv TOKENS_PER_DECISION USD_PER_M_IN cpu --calibrate`.

## Results, public tiers, 231 items

| arm | tok/decision | Intelligence | Calibration | Speed | Cost | **Score** |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen3-0.6B Q4_K_M, letter logits | 618.4 | 40.8 | 77.6 | 79.0 | 76.3 | **66.1** |
| Qwen3-0.6B Q4_K_M, candidate scoring | 605.4 | 49.7 | 73.0 | 79.7 | 76.5 | **68.6** |
| Granite 4.1 3B Q4_K_M, candidate scoring | 567.3 | 66.4 | 70.2 | 66.2 | 63.1 | **66.4** |
| Qwen3.5-4B Q4_K_M, letter logits | 632.1 | **73.3** | 71.8 | 63.0 | 61.7 | **67.3** |

Per-tier accuracy, Qwen3.5-4B: easy 1.0000, standard 0.8056, hard 0.5405. Its natural temperature
is 1.64, against 6.90 for Granite and 6.27 for the 0.6B, so it is far better calibrated before any
fitting.

All four land between 66 and 69 by different routes. The 4B has the best Intelligence by 6.9
points and the worst Speed and Cost; the 0.6B is the mirror image and wins the composite. On a
geometric mean of four axes that is not a coincidence worth ignoring: the best arm here is the
smallest model.

## What each axis actually rests on — they are not equally established

* **Intelligence — measured.** Accuracy on all 231 public items, weighted easy 0.14 / standard
  0.28 / hard 0.30. **The judge tier carries 0.28 and is not in the public release**, so the
  harness renormalises it away. Checked against the 25 published systems that ran both: judge runs
  0.033 *below* standard on average, but that is driven by strong systems. For the six with
  standard below 0.70 — our regime — judge runs a median 0.037 *above* standard. Substituting that
  moves our Intelligence by +0.1 to +1.1. So renormalising judge away is mildly pessimistic for us,
  not optimistic; an earlier note in this file said the opposite and was wrong.
* **Calibration — measured and held out, but half of it is missing.** Temperature fitted on 56
  hard items by NLL, ECE reported on the other 55. Temperature is monotone, so accuracy and
  therefore Intelligence are provably untouched. The axis is defined as the mean of the ECE term
  and a TVD-to-gold-distribution term; **the public sets carry no gold distributions**, so only
  the ECE half is computed. Uncalibrated, these arms score 0.0 / 37.5 / 35.6.
* **Speed — measured, with the harness's stated assumption.** p50/p95 over the standard tier,
  adjusted x2 +0.15 s as `composite_v12.adjusted_latency` does for our own CPU. The spec says
  p50/p95 come from the standard **+ judge** run; we have standard only.
* **Cost — measured tokens, reference price.** Input tokens per decision are now counted with
  each model's own tokenizer over the text that arm actually feeds (`TokenCount.java`); the
  earlier characters/4 estimate of 608 was within 2-5% of every exact count, which was luck rather
  than method. The price remains a size-class reference, $0.01/M in for the <=0.6B class and
  $0.03/M in for the 3-4B class, which is what the published rows do and label `kind: estimate`.

## The Cost rule matters more than any measurement feeding it

Pricing our runs the obvious way — host hourly rate times wall time, $0.266/hr on a Hetzner
ccx33 — gives Granite $1.246 per 1,000 decisions and **Cost 7.1**, which drags its composite to
32.5. That is not how JevBench prices self-hosted entrants. Every self-hosted row in
`results/v1.2/jevbench-v1.2-results.json` carries a cost of kind `estimate` with a basis like:

> ESTIMATE: hosted-provider price, DeepInfra Qwen3-Embedding-0.6B size-class reference list price
> $0.01/M in, $0.0/M out (a <=0.6B one-pass model with no generated output) x 279 input and 0
> output tokens per decision

On that rule the same Granite run is Cost 62.2 and scores 66.2. The 55-point swing comes entirely
from which rule is applied, so the rule is recorded here and the hourly-rate figure is kept beside
it rather than discarded.

## Where the time actually goes

| tier | n | share of total compute | mean latency | mean state |
| --- | ---: | ---: | ---: | ---: |
| easy | 48 | 2.0% | 0.520 s | 50 chars |
| standard | 72 | 2.1% | 0.366 s | 75 chars |
| hard | 111 | **96.0%** | 11.005 s | 4,322 chars |

Twenty-five items are 58.5% of all compute. Speed and Cost are a long-prompt prefill problem, not
a model-size problem.

## prefillBatchSize, settled

Retracted twice before (reported 1.56x, then 1.70x, then a 9% regression on dedicated hardware).
Per tier, same model and host, 231 items:

| tier | sequential | batch 512 | speedup |
| --- | ---: | ---: | ---: |
| easy | 25.0 s | 25.2 s | 0.99x |
| standard | 26.4 s | 27.7 s | 0.95x |
| hard | 1221.5 s | 1029.9 s | **1.19x** |
| all | 1272.9 s | 1082.8 s | 1.18x |

It is a long-prompt-only win and a slight regression on short prompts. Both earlier readings were
real; they were measured on different tier mixes.

## Next

Speed is the one axis with an unexplained gap. The 4B letter arm posts p50 2.993 s on the standard
tier, where prompts average 386 characters. One prefill of roughly a hundred tokens plus one
forward pass over 2.1 GB of Q4_K weights should be a few hundred milliseconds against memory
bandwidth, so this is about an order of magnitude off its roofline and is worth measuring before
anything else. Nothing here has profiled it.
