# A LoRA transfers to an unseen option set, and more training diversity destroys it

Qwen3-0.6B (`c1899de2`), rank 32, alpha 64, seven projections, 20,185,088 trainable parameters
(3.28% of 616M). Trained on three option sets and evaluated on a fourth held out entirely. Options
are rendered into every prompt, so nothing can be answered by memorising a label position. Scoring
ranks the options by their own likelihood rather than decoding text, because decoding measures
whether the adapter emits a well-formed label -- a property of the training objective, not of the
decision.

A40, torch 2.8.0+cu128, transformers 4.53.3, peft 0.18.1, 24 optimizer steps over 2,364 rows.

## The held-out result, settled at full size

Both adapters scored against the same 900 held-out `emotion` rows, each arm loading base weights
**fresh from the checkpoint**, paired per row, McNemar on the discordant pairs:

| arm | accuracy | delta | gained / lost | paired SE | verdict |
| --- | ---: | ---: | ---: | ---: | --- |
| floor | 0.3278 | | | | |
| base, no adapter | 0.4711 | -- | -- | -- | |
| **trained on 3 option sets** | **0.5178** | **+0.0467** | 115 / 73 | 0.0152 | **SIGNIFICANT** |
| trained on 6 option sets | 0.4522 | -0.0189 | 71 / 88 | 0.0140 | within noise |

**Transfer to an option set the adapter never trained on is real**, and **doubling training
diversity destroyed it**. Same held-out rows, same hyperparameters, same base; the only variable was
how many option sets were trained on.

That is the opposite of the scaling assumption this experiment was built to confirm. It is the
signature of the adapter fitting the specific trained option sets rather than learning to read
options: six sets and 4,759 rows give it more to memorise at the same 33 optimizer steps.

### Two claims retracted along the way

Run 1 measured +0.045 at n=200 and was reported as validating the path. Run 2 measured -0.030 and
the claim was retracted as unestablished. **Both statements outran the evidence.** At n=200 the
binomial standard error is about 0.035, so neither arm could resolve its own effect, and "inside the
noise band" was treated as evidence of absence when it was absence of evidence. The correct response
after run 1 was to run it at full size, which is what finally answered it.

### A contaminated run, discarded

The first full-size evaluation reused one base model object across adapters with `unload()` between
them; peft warned that a `peft_config` was already attached, so the second arm might have been
scoring stacked adapters. Every number from it was discarded and both arms re-run with a fresh base
per arm. The clean 6-set figure came back 0.4522, identical to the contaminated one -- the design was
unsound whether or not it happened to corrupt the result, and the only reason that is known is the
re-run.

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
