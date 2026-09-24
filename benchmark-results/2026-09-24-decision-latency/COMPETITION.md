# What the competition actually does

Read from source, not from README prose. SemIf (formerly OpenJev) is the second-placed
system on JevBench v1.2 at 74.7 composite and runs the same frozen Qwen3.5-4B we do, so
it is the only comparison on the board where the model is held constant.

Cloned at `references/openjev/semif`. Files read: `src/semif_phase1/core.py`,
`direct.py`, `shared.py`, `llamacpp_backend.py`, `reranker.py`, and
`results/phase1-summary.json`.

## They have no technique we lack

| | SemIf | us |
|---|---|---|
| model | Qwen3.5-4B | same |
| readout | option logits at the final position, no generation | same |
| answer slots | bare letter must be one exact round-trip token, verified against the prompt | same |
| state reuse | prefill the prefix once, save it, restore per branch | same |
| vocabulary projection | logits flagged on the last token only | same |
| grouped path | replicate the prefix cache across rows, ragged suffixes by position ids | same |
| batched vs looped | measured per device: CUDA batches, MPS loops | same finding |

`llamacpp_backend.py` is the closest analogue to our stack and is functionally identical
to what we now ship: `prefill`, `llama_state_seq_get_data`, restore per branch, and
`batch.logits[i] = (i == last)` so the vocabulary projection happens once.

**Their Speed and Cost advantage is hardware.** `results/` metadata: CUDA throughout,
RTX 3090 and 4090, and "native BF16 exact final-position option logits". Ours is four
physical CPU cores at Q4_K_M. That is not a code comparison.

## Their reranker is a dead end and they measured it

`reranker.py` scores one forward pass **per option** with a Qwen3-reranker yes/no
readout. `results/phase1-summary.json`, `paired_reranker_minus_direct`:

| fixture | reranker minus direct | bootstrap 95% |
|---|---|---|
| authored | **-0.1882** | [-0.2557, -0.1200] |
| wanli | **-0.1145** | [-0.1844, -0.0444] |

Balanced accuracy, both intervals entirely below zero. It is also N forwards per decision
where ours is one. Not pursued; noted as *their* measurement, not ours.

## Their prompt is better than ours, and not separably so

Transcribed verbatim from `core.py` and `direct.py` -- a chat template with a system
turn, a user turn carrying one JSON object of evidence, criterion and options, each
option a letter and a description with the label name dropped -- and run on our model,
our host, our readout, with the prompt as the only variable. Python's `json.dumps`
escaping reproduced exactly, since the bytes are the comparison.

| prompt | accuracy | Intelligence | Calibration | p50 |
|---|---|---|---|---|
| ours | 0.8917 | 88.0 | 73.0 (ECE 0.1348) | 1.913 s |
| **SemIf, verbatim** | **0.9083** | **90.0** | **82.1 (ECE 0.0894)** | 4.236 s |
| their template and system turn, our layout | 0.8583 | 84.3 | 77.3 (ECE 0.1136) | 3.187 s |

**+2.0 Intelligence and +9.1 Calibration, at 2.2x the latency.**

The third row is the important one. The chat template and the system turn are identical
for every question about one piece of evidence, so they belong in the shared prefix and
would cost nothing per question -- which made them the obvious active ingredient. They
are not: with our layout they score **84.3**, worse than either, and `ordinal` collapses
from 1.0000 to 0.5833. Their number comes from the combination. The factor does not
decompose, and assuming it did would have been the fourth wrong extrapolation in this
directory.

It also does not port cleanly to the regime Harriet is for. Their JSON places the
criterion between the evidence and the options, so the options and their descriptions
land after the per-question boundary and cannot be shared; ours are shared. Over a batch
of questions about one document their prompt costs more than 2.2x, not less.

**Tabled.** It buys accuracy with latency, which is the wrong direction under the current
priority, and the cheap version of it does not work.

## The one calibration worth keeping

Their browser model ladder, authored balanced accuracy on 144 items: Qwen3-0.6B 0.440,
MiniCPM5-2B 0.686, Qwen3.5-4B 0.813. Quality tracks size, which is the same thing our own
0.6B arm said. Nothing in a prompt or a kernel closes a gap that size.
