# A LoRA beats the frozen probe, and transfers to an option set it never saw

Qwen3-0.6B (`c1899de2`), rank 32, alpha 64, seven projections, 20,185,088 trainable parameters
(3.28% of 616M). Trained on three option sets and evaluated on a fourth held out entirely. Options
are rendered into every prompt, so nothing can be answered by memorising a label position. Scoring
ranks the options by their own likelihood rather than decoding text, because decoding measures
whether the adapter emits a well-formed label -- a property of the training objective, not of the
decision.

A40, torch 2.8.0+cu128, transformers 4.53.3, peft 0.18.1, 24 optimizer steps over 2,364 rows.

## The held-out result

| | accuracy | floor |
| --- | ---: | ---: |
| `emotion`, 6 options, **before training** | 0.5250 | 0.3356 |
| `emotion`, 6 options, **after training** | **0.5700** | 0.3356 |

Measured before any gradient step and written to the log first, so the adapter's contribution is a
difference against its own baseline rather than a claim. **+0.045 on an option set whose six labels
the adapter never saw.**

## Reshaping beats reading, on the same corpus and base

| option set | frozen probe (2026-09-21) | LoRA |
| --- | ---: | ---: |
| agnews (4) | 0.8500 | **0.9107** |
| spam (2) | not measured | **0.9221** |
| sst5 (5 ordinal) | 0.3083 *(below floor)* | **0.2687** *(below floor)* |

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
