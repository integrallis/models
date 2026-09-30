# What is left in a decision's latency, and where it is

Measured 2026-09-24, after the 0.3.46 release brought a warm decision to 0.486 s per
question. The question this answers: is that number close to what the machine can do,
or is there a lot left?

Short version: **a kernel that had shipped switched off was most of it, the release measures 1.54x
per decision for a demo-shaped workload, and what remains is at the instruction set's ceiling.**

An earlier version of this file concluded the opposite -- that the headroom was in the matmul inner
loop. Section 4 records why that was wrong, because the reasoning was a plausible ceiling estimate
that ignored what the quantisation actually costs, and it pointed at days of work in the wrong
place.

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

## 4. Faster tokens: the inner loop is NOT at its ceiling -- see the correction below

`harness/Ceiling` times real Q4_K and Q6_K weights from the shipped model at the batch width a
decision uses. At batch 18 they reach 58-65% and 50-54% of roughly 32 int8 multiply-accumulates
per cycle per core; batch 1 sits at 17%, which is the bandwidth-bound decode step and expected.
Thread scaling is clean from 1 to 16 and still gaining, so the partitioning is fine.

**An earlier version of this file read that as 1.3x to 1.6x sitting in the inner loop. That was
wrong**, and the correction matters more than the original claim.

That ceiling ignored what Q4_K actually costs. It carries a scale per 32-weight group, so applying
it is a second vector multiply for every one that does useful work: per 256-weight block the kernel
spends about 30 vector operations on 256 multiply-accumulates, which is roughly 17-25 MAC/cycle
structurally. Measured is 21. **The inner loop is at its algorithmic ceiling for AVX2.**

> **RETRACTED 2026-09-24, later the same day.** This conclusion is wrong, and the two
> failed experiments that produced it were not enough evidence for it. On a dedicated
> CCX33 whose cores actually run at 2400 MHz -- not the 2.0 GHz the BIOS string claims,
> which is where the "21" came from -- the same kernel reaches **25.6 MAC/cycle/core
> when its weights are cache resident** and only **17.1** when they stream from DRAM.
> Identical instructions, 50% apart. The arithmetic is not the wall; load latency is.
> Neither arm is bandwidth bound either: both use under a fifth of the 38 GB/s this box
> can sustain. Full measurements in `q4k-row-tile/NOTES.md`.

Tested rather than argued. The one visibly redundant operation was the per-group scale broadcast,
recomputed inside the batch loop -- 144 times per block instead of 8. Hoisting it is bit-exact,
because the accumulation it feeds is integer. Alternating runs, stock against hoisted: 305/344/300
against 296/322/150 G-MAC/s. **No change**; the compiler was already hoisting it. `-C
target-cpu=native` likewise does nothing, because the hot paths already carry explicit
`#[target_feature(enable = "avx2,fma,f16c")]`.

Past this needs `vpdpbusd`, which folds the multiply, the widening and the accumulate into one
instruction and roughly halves the operation count. It is AVX512-VNNI or AVX-VNNI: absent on Zen 3,
present on Zen 4 and later and on Intel from Ice Lake. That is a deployment choice or a new code
path, not a tune.

## 5. The gap was never in the matmuls (`harness/Achieved`, `harness/PerShape`)

The right question is not how fast one matmul goes; it is whether a forward pass -- about two
hundred matmuls of a dozen shapes, plus norms, a recurrence, attention and a vocabulary projection
-- gets anywhere near that rate. Both numbers from the same host:

| | G-MAC/s |
|---|---|
| best single matmul at batch 18 | 407 |
| whole forward pass, 18 tokens | 211 |

**52%.** Timing every shape in the model separately and summing gives a matmul floor of 0.205 s
against a measured forward pass of 0.357 s, so **43% of a forward pass was not matmul at all.**

> **CORRECTED 2026-09-24, later the same day.** That 43% is an artifact of how the floor
> was measured, not a property of the forward pass. `PerShape` times one exemplar tensor
> per shape, eight warmups and ten rounds deep; a 9216x2560 Q4_K tensor is ~13 MB and this
> CCX has 32 MB of L3, so it is resident by the time it is timed. A forward pass never sees
> a weight twice. Re-measured with `q4k-row-tile/ColdShape.java`, which touches every
> `blk.*` weight exactly once in layer order, the matmul floor is **0.392 s against a
> 0.498 s forward pass -- 79% matmul, not 57%**. The per-shape numbers in the table below,
> including the 445-478 against 218-244 spread, are resident-tensor rates and do not
> describe what the model does. Judge kernel changes on `ColdShape` and `Achieved`, never
> on `PerShape` alone.

Two things are visible in the per-shape table. Narrow outputs run at half the rate of wide ones --
9216-row `ffn_gate` and `ffn_up` at 445-478 G-MAC/s against 218-244 for the 2560- and 1024-row
tensors, because a 9216-column activation no longer fits in L2 and gets re-streamed. And three
quantisation types are in play, with Q5_K and Q6_K decoding more per weight than Q4_K.

But neither was the 43%.

## 6. What it was: a kernel that shipped switched off

`models.native.gatedDeltaNet` defaults to false. The Gated DeltaNet recurrence therefore ran in
Java, token by token, reading and writing each layer's 4 MiB of recurrent state once **per token**
instead of once per batch. Twenty-four layers over eighteen tokens is about 3.4 GiB of state
traffic that the chunked native scan does in 192 MiB.

The kernel was written, tested and shipped. Nothing turned it on, because it had been evaluated for
generation, where a decode step advances one token and a chunked scan has nothing to chunk. A
decision is the opposite shape: it prefills its whole question in one batch.

| | ms/token | forward pass |
|---|---|---|
| `gatedDeltaNet=false`, as shipped | 19.9 | 0.359 s |
| `gatedDeltaNet=true` | **14.5** | **0.261 s** |

**The gain depends on how many tokens are being prefilled, and not monotonically.** On the
dedicated 4-core host, a forward pass:

| tokens | false | true | gain |
|---|---|---|---|
| 18 | 0.500 s | 0.443 s | 1.13x |
| 128 | 2.863 s | 2.374 s | **1.21x** |
| 512 | 11.903 s | 11.645 s | 1.02x |

Small batches have little state traffic to save; long ones are increasingly dominated by attention,
which grows with the square of the position count and dilutes the saving. The middle is where
decisions live. It is never slower in anything measured here.

At decision level on the same host:

| shape | false | true | gain |
|---|---|---|---|
| one document per case, no rubric | 1.483 s | **1.107 s** | **1.34x** |
| one document per case, with rubric | 2.201 s | 1.604 s | 1.37x |
| one warm question over shared evidence | 0.444 s | 0.408 s | 1.09x |

A warm question prefills only its own 18 tokens, so it gains least. A decision that brings its own
document gains most, and that is the shape the demo videos use.

**It costs one item in 120.** Accuracy 0.9000 to 0.8917, Intelligence 88.9 to 88.0 -- this cohort's
noise floor, measured in section 5 of NOTES.md -- while Calibration improves from 71.7 to 73.0 and
the Speed axis from 66.7 to 69.6.

So the decision plan now recommends it, alongside the quantized decode kernel it already
recommended. A deployment setting still wins over the recommendation, and it is never slower in any
shape measured, so recommending it does not need a caveat.

## 7. Where that leaves it

**The before state was measured, not computed.** An earlier version of this file quoted "~1.53 s"
for it, arrived at by taking a measured after-number and adding back an assumed readout saving. That
is the habit this file exists to avoid, so the published `modeljars 0.1.52` jars were fetched from
Central and run against the same harness on the same host, with `models` held at 0.3.46 so that only
the release's own changes move.

Alternating, on an idle host, fifteen decisions each bringing its own document (`harness/Before`):

| pair | before (0.1.52) | after | |
|---|---|---|---|
| 1 | 1.665 s | 0.939 s | |
| 2 | 1.670 s | 1.085 s | |
| 3 | 1.829 s | 1.110 s | |
| **median** | **1.67 s** | **1.09 s** | **1.54x** |

**1.54x, or 35% less time per decision**, in the shape the demo videos use. The estimate it replaces
was 1.38x, so the estimate was low as well as unearned.

A warm question over already-prefilled evidence gains much less, because it only prefills its own 18
tokens: 0.444 s to 0.408 s, measured as an A/B of the recurrence change alone on the same build.

All of the numbers in sections 4 and 5 above were taken on a 16-vCPU shared host, which has more
cores and about 10% run-to-run spread; the decision-level and per-token figures here are from the
dedicated 4-core host every other published number uses. Where the two disagree, this host wins.

The arithmetic that remains is at the inner loop's ceiling for this instruction set. What is left,
in order of size:

1. **A machine with VNNI.** Zen 4 or later, or Intel from Ice Lake, roughly halves the operation
   count per multiply-accumulate. Larger than everything below put together, and it is a purchase
   rather than a change.
2. **Cache blocking for narrow outputs.** `ffn_down` and `ssm_out` run at half the rate of the fat
   FFN tensors because their activations do not fit in L2. Worth perhaps 15% of a forward pass.
3. **Whatever is left of the non-matmul 43%** after the recurrence fix. Not re-measured, and it
   should be: the split between norms, attention and the recurrence has moved.
4. **A persisted prefix state.** The first decision over new evidence is still the largest
   user-visible number, and it is paid again on every process start for the same document.
