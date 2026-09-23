# Held-out temperature scaling — result

Measured here, 2026-09-21. Preflight and its recorded expectation: `PREFLIGHT-CALIBRATION.md`.
Diagnosis that motivated the run: `CALIBRATION-DIAGNOSIS.md`.

## The number

One temperature, fitted on 159 held-out items (JevBench public `easy` 48 + `hard` 111), applied to
the stored 72-item `original` cohort. Disjointness re-verified against the exact files used:
`0` shared ids, all 159 scored.

| | raw (T=1) | held-out T=3.2794 |
| --- | ---: | ---: |
| accuracy | 0.6667 | 0.6667 |
| **ECE** | **0.2062** | **0.1757** |
| Brier | 0.5034 | 0.4823 |
| strict schema validity | 1.0 | 1.0 |

Accuracy is unchanged by construction — temperature is monotonic and cannot move an argmax. The
in-sample bound recorded earlier was **0.1087**; the honest held-out figure is **0.1757**. The
preflight predicted "worse than 0.1087" before the run, and that is what happened. Jev 1.13.0 on
the same harness and cohort is 0.0333, so this closes roughly 18% of the gap and leaves the
calibration blocker open.

Fit cost: 159 items, 3420.1 s wall on ccx33, prefix sharing proven 159/159.

## Per family — the global temperature is not a uniform win

| family | acc | raw ECE | tempered ECE | |
| --- | ---: | ---: | ---: | --- |
| adequacy | 0.4167 | 0.4584 | 0.3112 | better |
| routing | 0.3333 | 0.4170 | 0.2233 | better |
| policy | 0.6667 | 0.2605 | 0.1861 | better |
| intent | 0.7500 | 0.1914 | 0.1880 | flat |
| extraction | 0.9167 | 0.1250 | 0.2089 | **worse** |
| ordinal | 0.9167 | 0.1312 | 0.4535 | **much worse** |

The pattern is not noise: softening helps every family the model is bad at and hurts both families
it is good at. Where the model is accurate it was already near-calibrated, so a global scalar
converts calibrated confidence into underconfidence.

## Why the transfer is imperfect — a protocol limitation, recorded

The fit set and the scored set are disjoint in items, but they are **not drawn from the same
distribution**. Only 2 of the 6 cohort families occur in the fit set at all:

- cohort: `adequacy, extraction, intent, ordinal, policy, routing`
- fit set: `adversarial, ambiguous, extraction, fact, intent, judge_hard, long_policy, multi_hop,
  probability, routing_hard, temporal_numeric, tool_selection, tradeoff, trap`
- shared: `extraction, intent`

By question type the skew is sharper, and it names the mechanism directly: the `score` type is
**6 of 159** fit items but **12 of 72** cohort items. `score` is the type behind the `ordinal`
family, which is the family that regressed worst. The temperature was fitted almost entirely on
`choice` (103) and `noul` (50) and then applied to a type it barely saw. The `hard` split is also
harder than `original`, so the fit sees more confident-wrong items and pulls T upward.

So `0.1757` is a fit transferred across a distribution shift, not a same-distribution fit. That is
the harder test and arguably the deployment-relevant one, but it means the true same-distribution
value is bracketed between the in-sample `0.1087` and this `0.1757`, and neither endpoint should be
quoted alone.

## Rejected arm: per-question-type temperature (exploratory)

Labelled exploratory, not pre-registered: the decision to try it was taken after seeing the
per-family breakdown. Question type is a structural property known in advance, so fitting one T per
type is legitimate in shape — the objection is only to when it was chosen.

Fitted on the same 159 held-out items: `choice` T=2.2848 (n=103), `noul` T=8.3419 (n=50),
`score` T=17.8112 (n=6).

| | global T | per-type T |
| --- | ---: | ---: |
| ECE | 0.1757 | **0.1596** |
| Brier | **0.4823** | 0.5114 |
| ordinal ECE | 0.4535 | 0.6313 |

**Rejected.** ECE improved while Brier got worse. ECE is a binned statistic and rewards pushing
confidence toward the base rate whether or not the ranking improves; Brier is a proper scoring rule
and disagrees. When a binned metric and a proper scoring rule disagree, the proper scoring rule
wins. The mechanism is visible: `T=17.8112` fitted on 6 items is a near-uniform distribution, and
applying it flattens a family that is 91.7% accurate.

Kept as the standing rule for this work: **report Brier beside every ECE**, because ECE alone would
have accepted this.

## What this does and does not license

- Does: a real held-out calibration number exists and the direction of the fix is confirmed.
- Does not: close blocker 1. 0.1757 vs 0.0333 is still a different regime, and the per-family split
  shows a single scalar cannot get there.
- Next: the trained scalar head, which is the same artifact blocker 2 needs. A head trained with a
  proper loss calibrates as it fits, rather than correcting a frozen base after the fact.

Jev comparison numbers remain internal pending counsel on MCA 2.3(b) and 14.1.
