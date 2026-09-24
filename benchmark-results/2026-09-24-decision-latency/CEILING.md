# What is left in a decision's latency, and where it is

Measured 2026-09-24, after the 0.3.46 release brought a warm decision to 0.486 s per
question. The question this answers: is that number close to what the machine can do,
or is there a lot left?

Short version: **there is about 1.3x to 1.6x left, all of it in one place, and
everything else is measured out.**

## 1. There is no bookkeeping to remove (`harness/Where`)

Each stage of a warm decision timed on its own, in the order the runtime performs them,
32 decisions, 18-token suffix, rubric shared:

| stage | seconds | share |
|---|---|---|
| tokenize the prompt | 0.0012 | 0.3% |
| resume the shared prefix | 0.0046 | 1.2% |
| **prefill the suffix and read the logits** | **0.3854** | **98.5%** |
| score the letters | 0.0002 | 0.0% |

Restoring roughly 100 MiB of recurrent state costs 4.6 ms. An earlier estimate that a
third of a decision was overhead was arithmetic on a difference, and it was wrong: the
parts add up, and the prefill is essentially all of it.

**So the only levers are fewer tokens, or faster tokens.**

## 2. Fewer tokens: both attempts cost more than they saved

The 18-token suffix is about 7 tokens of criterion and 11 of lettered list.

**Move the lettered list into the shared prefix.** Already measured for the release:
Intelligence 88.9 with the list after the criterion, 79.6 with it before. A model wants
the letter-to-label mapping after the question it answers. Rejected, 6.5 noise floors.

**Drop the lettered list and read the label tokens directly.** When every label is a
single token -- `true` and `false` are, and 53 of the 120 cohort items qualify -- the
indirection looks like pure overhead: the prompt becomes evidence, rubric, criterion,
`Answer:`, and the logits for the labels are read at that position. Nine tokens instead
of eighteen.

| | lettered | label tokens |
|---|---|---|
| accuracy | 0.9000 | 0.8049 |
| Intelligence | 88.9 | **78.6** |
| qualifying items | -- | 53 of 120, rest fell back to letters |

Worse on exactly the families that qualified: `fact` 1.0000 to 0.7500, `policy` 0.7500
to 0.5000, `adequacy` 0.5000 to 0.4167. Rejected, 10 noise floors.

That is worth stating plainly, because the lettered list looks like packaging: **it is
not. A lettered multiple choice constrains the model in a way that "Answer:" followed by
a free token does not, and it is worth ten points even where it is not needed to
represent the labels.** The 11 tokens are load-bearing.

## 3. Grouping now has nothing left to offer

Grouping's only benefit was replacing one bandwidth-bound readout step per question
with one per group. The release folded that step into the prefill, where it is one more
row of an already compute-bound batch. N questions are N questions' arithmetic either
way, so there is no longer anything for grouping to amortise -- which is what the
forced-on measurement shows: 0.63x at two questions to 0.92x at twenty, and answers
0.045 apart.

## 4. Faster tokens: the kernel is at about two thirds of its issue ceiling

`harness/Ceiling` times a real Q4_K and Q6_K weight from the shipped model at the batch
width a decision actually uses, and compares against this core's issue limit. Zen has
AVX2 and no AVX-512, and an int8 dot product goes through `VPMADDUBSW` into
`VPMADDWD`, so the sustained ceiling is roughly 32 int8 multiply-accumulates per cycle
per core.

Measured on a 16-vCPU shared host (EPYC Rome), which adds about 10% run-to-run spread:

| tensor | type | batch | G-MAC/s | of ceiling |
|---|---|---|---|---|
| `blk.0.ffn_up` | Q4_K | 1 | 89-103 | 17-20% |
| `blk.0.ffn_up` | Q4_K | 18 | 298-330 | 58-65% |
| `blk.0.ffn_down` | Q6_K | 1 | 88-92 | 17-18% |
| `blk.0.ffn_down` | Q6_K | 18 | 257-277 | 50-54% |

Batch 1 at 17% is the bandwidth-bound decode step and is expected. **Batch 18 at 50-65%
is the number that matters**, because that is what a decision does.

### It is the inner loop, not the work splitting

Thread scaling of the same batch-18 matmul:

| threads | G-MAC/s | gain |
|---|---|---|
| 1 | 42.4 | -- |
| 2 | 78.4 | 1.85x |
| 4 | 120.1 | 1.53x |
| 8 | 216.1 | 1.80x |
| 16 | 330.1 | 1.53x |

It scales cleanly to 16 and is still gaining, so the partitioning is not the problem.
One thread at 42.4 G-MAC/s against a 64 G-MAC/s per-core ceiling is **66%**, and that
is where the gap lives.

### Nothing free is available

`-C target-cpu=native` makes no difference at all -- 292 against 298 G-MAC/s at batch
18, inside the noise, and slightly worse at batch 1. The hot paths already carry
explicit `#[target_feature(enable = "avx2,fma,f16c")]`, so the compiler has nothing to
add. The weight block is also already decoded once per row and reused across the whole
batch, which is the big amortisation and is done.

### What a real attempt would look like

Per Q4_K block of 256 weights the kernel spends roughly 30 instructions per batch row
on 256 multiply-accumulates, which is about 34 MAC/cycle structurally; measured is 21.
Closing that is instruction-level work on the K-quant inner loop -- the accumulation
dependency chains and the horizontal reductions -- worth perhaps 1.3x to 1.6x.

**It was not attempted here**, for two reasons worth recording rather than hiding: it is
a serious rewrite of the hottest code in the project, and it cannot be honestly
validated on a shared-vCPU host, which is the only host available while the dedicated
one is scoring the release. Q6_K is the weaker of the two at 50-54% and `ffn_down` and
`output` are Q6_K, so that is where to start.

## 5. The floor, stated plainly

A warm decision is 18 tokens of arithmetic. On 4 physical Milan cores that is 0.333 s at
the measured 18.5 ms a token, and 0.486 s end to end including a cold-ish first
resumption. Perfect kernel work would take the arithmetic to roughly 0.21-0.25 s.

Below that needs fewer tokens, which costs accuracy, or a smaller model, or a machine
with AVX-512 -- Zen 4 or later roughly doubles int8 throughput per core, and this host
is Zen 3. That is a deployment choice rather than a code change, and it is the single
largest available factor.

The other number worth attacking is not the warm one: the **first** decision over new
evidence costs 5.5 s, because 196 tokens of evidence and rubric are prefilled at
18.5 ms each. Nothing about that is wasted work, but it is paid again on every process
start for the same document, and a persisted prefix state would remove it entirely. That
is a feature rather than a tune, and it is the largest user-visible number left.
