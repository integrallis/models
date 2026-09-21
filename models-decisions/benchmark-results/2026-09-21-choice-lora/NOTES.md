# A LoRA beats the frozen probe on trained sets; transfer is NOT established

Qwen3-0.6B (`c1899de2`), rank 32, alpha 64, seven projections, 20,185,088 trainable parameters
(3.28% of 616M). Trained on three option sets and evaluated on a fourth held out entirely. Options
are rendered into every prompt, so nothing can be answered by memorising a label position. Scoring
ranks the options by their own likelihood rather than decoding text, because decoding measures
whether the adapter emits a well-formed label -- a property of the training objective, not of the
decision.

A40, torch 2.8.0+cu128, transformers 4.53.3, peft 0.18.1, 24 optimizer steps over 2,364 rows.

## The held-out result: two runs, opposite signs, both inside the noise

| run | trained sets | rows | held-out `emotion` | delta |
| --- | ---: | ---: | --- | ---: |
| 1 | 3 | 2,364 | 0.5250 -> 0.5700 | **+0.045** |
| 2 | 6 | 4,759 | 0.4850 -> 0.4550 | **-0.030** |

Binomial standard error at n=200 and p about 0.5 is **+/-0.035**. Both deltas sit within roughly one
standard error of zero, so **transfer to an unseen option set is not established in either
direction**.

This corrects a claim made here after run 1 alone, which called the path "validated" on a single arm
inside the noise band. The second arm should have been run before the claim.

The two runs' *absolute* baselines are also not comparable: the corpus builder's seeded RNG consumes
different entropy depending on how many option sets precede `emotion` in the dict -- second in the
small corpus, last in the large -- so the runs sampled different rows. Only within-run deltas mean
anything.

Doubling option-set diversity made trained sets better and held-out drift down, which is the shape of
overfitting to the trained sets: the adapter learning those option sets rather than learning to read
options. Consistent with, not proven by, n=200.

## Reshaping beats reading on trained sets -- this part is solid

| option set | frozen probe | LoRA run 1 | LoRA run 2 |
| --- | ---: | ---: | ---: |
| agnews (4) | 0.8500 | **0.9107** | **0.9048** |
| spam (2) | not measured | **0.9221** | **1.0000** |
| subj (2) | not measured | not trained | **1.0000** |
| sst2 (2) | not measured | not trained | **0.9000** |
| cr (2) | not measured | not trained | **0.8929** |
| sst5 (5 ordinal) | 0.3083 *(below floor)* | **0.2687** *(below floor)* | **0.2414** *(below floor)* |

Run 2 validation across six trained sets: **0.8400**. AG News beats the frozen probe in both runs --
the one comparison repeated often enough to rely on.

## The finding that reframes the day

**The base model already ranks options zero-shot at 0.5250 against a 0.3356 floor, untrained.**

That is the same task on which a frozen linear probe scored exactly chance (0.4944 against 0.5169).
The two are different mechanisms, and the difference is the point: likelihood ranking runs through
the vocabulary projection, which was trained across all of pretraining, while a probe over the final
hidden state has to learn the state-to-candidate relation from a few hundred examples.

So the probe was not a weak version of the right approach. It was the wrong mechanism for arbitrary
option sets, and `SharedPrefixCandidateEvaluator` -- which already ranks candidates by likelihood --
was the right one all along. It is also why the untrained base reached 0.6667 on the JevBench
cohort while the trained probe could not transfer at all.

## The ordinal limit survives representation learning

`sst5` at five levels has now failed under two harvest prompts, three label granularities, two loss
formulations, an L2 sweep on each, **and** a LoRA: 0.2687 against a 0.2711 floor. Four frozen-probe
attempts and one adapter attempt, all below floor.

That is no longer a property of reading a frozen representation. Five-level ordinal resolution is
absent from what this 0.6B base makes available at all, and an adapter trained on 2,364 rows does
not install it. JevBench's score family uses four levels.

## What is not claimed

24 optimizer steps on 2,364 rows is a small run. The **direction** is established -- transfer to an
unseen option set is positive where the probe was flat -- but the **magnitude** is not. A larger
corpus with more option sets is the obvious next move, and the held-out design makes it measurable.

These are also not JevBench's families. The adapter has not yet been served through the Java runtime
(`loadActivatedAdapter` reads exactly this safetensors layout) nor scored on the cohort.
