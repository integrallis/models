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

---

# CLM: the second competitor, and the first one we can actually run

Read 2026-09-24 from `github.com/Contrastive-LM/CLM` at commit-of-clone (Apache-2.0,
797 stars, last push 2026-09-24). Announced the same day on X claiming "up to 9x faster
than Jev" with comparable quality. Everything in this section is either *read from their
source* or *their own committed numbers*, clearly marked. Nothing here is measured by us
yet, and nothing here should be repeated as though it were.

## Architecture, read from `src/clm/heads.py` and `src/clm/schema.py`

CLM-8B is a **bi-encoder**, not a decoder readout:

- encoder: **frozen Qwen3-8B with last-token pooling**, 4096-d, served by vLLM
  `--runner pooling`
- two MLP projection heads, `4096 -> width -> 512`, depth 2 by default
- score is `exp(logit_scale) * cos(state_head(s), action_head(c))`, InfoNCE-trained
- the trained artifact is **75 MB of MLP heads** over an off-the-shelf encoder

Training, per their README: 60M Nemotron Q&A pairs, 30M synthetic hard negatives, 1M
agentic trajectories.

The action head embeds **the option's own text**. That is how it escapes the failure this
project measured on 2026-09-21 and wrote into `LetterLogitScorer`: a head fitted over a
frozen hidden state learns the labels it was trained on and scores chance on any other
set. CLM's labels are not classes, they are strings to embed, so an unseen option is just
unseen text. **This is a real architecture we have not tried and cannot dismiss on the
grounds we dismissed the old one.**

## The structural weakness, visible in their own results

A bi-encoder embeds state and option **independently**, so the option text never attends
to the state. Their committed `examples/t_rex/results/*.json`:

| | CLM | Jev |
|---|---|---|
| model latency p50 | 2.6 ms | 131.9 ms |
| agreement with the planner's correct move | **0.658** | **0.987** |
| shield interventions | **4883** | **28** |
| mean best score / deaths | 697 / 0 | 697 / 0 |

Both arms score 697 with zero deaths, which is what the README's "on par" rests on. The
equal score is the harness's safety net: across 3,342 decisions the shield replaced CLM's
answer 4,883 times, against 28 for Jev.

And the task is not subtle. The prompt's options read:

    jump: Safe. Clears the 2 large cacti. Best.
    duck: Unsafe. Hits the 2 large cacti. Collision.
    run:  Unsafe. Hits the 2 large cacti. Collision.

The answer is written in the option text and CLM picks it 66% of the time. That is the
bi-encoder's limit arriving as a number: "Safe ... Best" and "Unsafe ... Collision" are
similar strings, and nothing lets either one look at *this* state. A cross-encoder gets
98.7%.

**Our letter-logit readout is a cross-encoder.** The options sit in the prompt and attend
to the evidence. That is the axis on which we should expect to beat CLM, and it is now the
obvious thing to measure.

## Two things they do not have that we do

**They re-encode the state per question.** `schema.state_text` appends each question's
instructions to the state, so N questions about one document cost N full encoder passes
over the whole document. Our shared-evidence prefix -- prefill the evidence once, resume
per question -- has no counterpart in their design. Measured here, that is the difference
between 1.38 s and 0.61 s per question.

**They need a GPU.** vLLM serving Qwen3-8B, benchmarked on a 4090 and an H100. We are a
4B on CPU.

## Where their 9x is real, and where it does not reach us

Their speedup grows "when the number of candidate actions is large or when actions are
reused across states" -- both are bi-encoder sweet spots, because cached action
embeddings make extra candidates nearly free. Neither describes our workload: we already
answer a question of any option count in **one** forward pass, so we are already O(1) in
options. Their 9x is measured against Jev, not against us, on an 8B against our 4B. It
does not transfer, and per section 1 of this file the rule stands: we re-run every arm
ourselves or we make no comparison.

Convergent detail worth noting: `schema.state_text` is commented "Context first, question
last -- the layout the heads were trained on." We measured our way to the same layout
independently (`ACCURACY.md`: rubric before the criterion, letters after, 88.9 against
79.6 for the alternative).

## What to measure, and what it costs

CLM-v0.1-8B weights are Apache-2.0 and on the Hub, so **this is the first competitor we
can run on our own harness** rather than read about -- no MCA constraint, no self-reported
number to take on faith. The arms that matter:

1. CLM-8B on our 120-item JevBench cohort, their heads, their encoder, our scoring -- does
   the cross-encoder advantage show up where the answer depends on the state?
2. `clm-raw`, their own no-head ablation, as the floor.
3. Harriet on the same items, already at Intelligence 88.0.

Needs a GPU box for the vLLM encoder, which is a RunPod pod and not the CPU bench host.
