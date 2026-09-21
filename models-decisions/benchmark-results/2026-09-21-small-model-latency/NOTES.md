# Why we do not get faster when the model gets smaller

Measured on one 6-core i7-9750H (2019, AVX2, no AVX-512), 404-418 token document, rust-ffm kernel.
llama.cpp b9960 with BLAS/Accelerate is the reference, run at `-t 6` on the same files.

**Caveat that governs every number here.** The same default configuration read 17.80 ms/token in one
run and 12.76-13.38 ms/token in later runs. That is roughly 40% drift on a thermally throttling
laptop, and it is larger than several of the effects below. Only same-session back-to-back
comparisons are trustworthy; nothing cross-session is quoted as a result.

## The finding

| params | ours | llama.cpp | |
| ---: | ---: | ---: | --- |
| 3.40 B (Granite 4.1, Q4_K_M) | 36.90 ms/tok | 40.05 ms/tok | we are 1.08x faster |
| 0.596 B (Qwen3-0.6B, Q4_K_M) | 17.80 ms/tok | 3.46 ms/tok | we are 5.1x slower |

5.7x fewer parameters bought 2.1x. Fitting those two points leaves roughly 11 ms/token that does
not shrink with the model. At 3B that overhead is about 30% of the bill and we look competitive; at
0.6B it is about 63% and we do not.

This is load-bearing for the decisions work. A System One model is supposed to be small -- Jev's
class, Laya at 421M, OpenJev at 0.6B -- so a runtime with a floor independent of model size cannot
compete in that class at all. The base swap that was queued would have delivered about 2x and
stopped.

## Ruled out, each by measurement

| hypothesis | result |
| --- | --- |
| the prefix-sharing witness fork inflates marginal cost | -3.5%, i.e. nothing |
| quantization (Q4_0 against Q4_K_M, same weights) | 16.76 against 17.80 -- flat for us |
| thread count | 2 -> 12 threads buys 1.23x (16.75 -> 13.64) |
| parallel work threshold | 1048576 -> 4096 buys about 18% |

Flat thread scaling is the tell. A compute-bound GEMM should scale close to linearly across six
cores; flat scaling means memory-bound, and memory-bound means low arithmetic intensity.

Note the quantization result corrects an earlier claim of a "1.69x implementation gap against
llama.cpp". That gap is not general: on Q4_K_M at 3B we are faster. The Q4_0 difference belonged to
llama.cpp, not to us.

## RETRACTED: the prefill-batch "win"

Reported here first as 1.56x, then "confirmed" at 1.70x in a controlled laptop run. **Both were
artifacts of a thermally throttling laptop.** Re-measured on a dedicated-vCPU box (Hetzner ccx33,
8 dedicated cores, ash), three interleaved repeats of each arm:

| config | prefill ms/tok | tail ms/tok | ten questions |
| --- | --- | --- | --- |
| default | 5.18 / 5.31 / 5.31 | 9.41 / 10.37 / 10.27 | **4.329 / 4.604 / 4.580 s** |
| prefillBatchSize=512 | 4.41 / 4.12 / 4.61 | 13.62 / 13.24 / 13.63 | 4.974 / 4.766 / 5.063 s |

Repeat spread on this box is about 2.5% against the laptop's roughly 40%, and the two arms do not
overlap on any measure: **every** batch512 run is worse on total than **every** default run.

`prefillBatchSize=512` is a **net regression of about 9%** on this workload. Prefill improves 1.22x
and the 23-token tail degrades 1.33x, and there are ten tails against one prefill. Sizing scratch
for 512 tokens and pushing 23 through it costs more than the better GEMM shape saves.

This was one step from being filed as a default change in `models`, which would have shipped a
regression to every small-model caller. The lesson is not subtle and is now a rule for this work:
**latency A/Bs run on dedicated-CPU hosts, interleaved, with the baseline measured on both ends.**
A 40% measurement floor cannot resolve a 9% effect, let alone tell its sign.

## What the stable box says

Qwen3-0.6B Q4_K_M, 418-token document, 23-token tails, rust-ffm, 8 dedicated cores:

| | |
| --- | ---: |
| prefill | ~5.3 ms/token, about 2.2 s |
| question tail | ~10.3 ms/token, about 0.22 s each |
| ten questions, total | **4.33-4.60 s** |
| Jev 1.13.0, same ten prompts | 2.75 s |

We are **1.57x** off, not the 4x the laptop implied. Prefill and tails are each about half the bill.

The tail costs roughly twice as much per token as the prefill, which is the signature of work that
is too small to fill the machine. Ten tails of 23 tokens are ten separate passes; batching them
across forked branches into one ragged pass is the only lever aimed at the dominant cost, and
unlike a buffer-size knob it changes the shape of the work rather than its padding. It needs a
batched hidden-state API, which does not exist -- `prefillBatch` returns logits.

## Confirmed win: batching the question tails

`prefillBatchHiddenStates` forks N branches off the frozen prefix and prefills them in one ragged
pass, returning one final normalized hidden state per branch. It also skips the vocabulary
projection, which the old path computed and discarded -- on a 1024-wide model that is a
151,936 x 1024 matmul per question, about the cost of a transformer layer.

Dedicated ccx33, Qwen3-0.6B Q4_K_M, 418-token document, ten 23-token tails, five rounds with two
discarded as warm-up:

| | best | round spread |
| --- | ---: | ---: |
| 10 tails sequential | 2.467 s | 3.7% |
| 10 tails batched | **1.716 s** | 7% |
| | **1.44x** | |

| | |
| --- | ---: |
| ten questions, sequential | 4.631 s |
| ten questions, batched | **3.879 s** |
| Jev 1.13.0, same ten prompts | 2.75 s |

Every batched round beats every sequential round, so the arms separate cleanly.

**Correctness first.** Two integration tests against a real Qwen3-0.6B fixture assert the batched
hidden states are bit-identical to the sequential ones, including the batch-of-one case. A batched
path that was merely close would make a decision depend on how many questions happened to be asked
together, which is a faster wrong answer. The ragged loop is shared with the logits path
(`runSessionPrefillRows`) so a batching bug cannot live in one and be invisible to the other's tests.

Prefill is now 2.163 s of the 3.879 s, so it is the next target rather than the tails.

**The base-digest check earned its keep here.** Pointing the Granite-fitted head at Qwen3-0.6B was
refused outright. The timings would have looked perfectly reasonable while every probability was
meaningless -- exactly the failure that produces a confident published number.

## Carried over from the laptop, structural rather than numeric

These do not depend on timing and still stand:

## The parallel threshold, unresolved

`parallelThreshold=1048576` looks tuned for 3B-class hidden widths; `prefillBatchSize=32` is
**correct** for this workload and must not be changed (see the retraction above).
A 1024-wide model's projections land at 1024x1024 = 1048576 elements, exactly on the threshold, so
a narrower model silently serialises. Neither is a decisions problem; both belong in `models`,
because they affect every small model the runtime serves.

## Constraint discovered while choosing a base

Prefix sharing is implemented only in `LlamaDecoder`. Qwen3.5 has its own forward pass and throws
`model family qwen35 cannot share KV-cache prefixes`, so it cannot host a decision head on a shared
prefix whatever its speed. Any candidate base must be llama-architecture. Qwen3-0.6B qualifies.

Separately, Qwen3.5 allocates session state for its whole declared context window, so the 0.8B model
could not open a session at 12 GB of heap while the 3.4B model was fine. The probe now caps
`models.purejava.maxContextLength`, because otherwise which bases are measurable is decided by
declared context length rather than by speed.
