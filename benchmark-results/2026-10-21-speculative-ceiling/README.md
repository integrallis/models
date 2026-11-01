# Is a model-based speculative drafter worth a new module in Models?

**Status: protocol frozen, no data collected.** Written and committed before the first measurement.

## Why this question, now

Models ships speculative decoding already: `SpeculativeInferenceBackend.verify(int[], int)` with a
`LogitBatch` return, `RewindableInferenceBackend` for the KV rollback, the speculative path in
`GenerationLoop`, and `SpeculativeGenerationOptions`. What it ships as a **drafter** is
`NgramDraftStrategy`, described in its own javadoc as "modeled after llama.cpp's ngram-simple".
There is no model drafter.

[FlashMap](https://arxiv.org/abs/2610.24365) (Shen et al., 17 Oct 2026, MIT-licensed code) reports
up to **6.14x lossless** speedup on Qwen3-8B with a 0.5B flow-map drafter conditioned on the target's
hidden states, drafting a whole K=16 block in one network evaluation. Those numbers are **read, not
measured here**, and they were measured on an NVIDIA H200 with SGLang at concurrency 1.

A new module is a long-lived commitment: a DiT architecture the Java backend does not have, hidden
states exposed from five layers in the decode hot path, a drafter checkpoint format, and a
conversion and quantisation story. Before any of that, one property of **our** runtime decides
whether the idea can pay at all.

## The hypothesis

> **H.** On Models' own kernels, verifying a block of K proposed tokens in one pass costs materially
> less than K sequential decode steps -- enough that a drafter with a published acceptance length
> would beat the n-gram drafter that ships today by a margin worth a module.

The arithmetic this rests on, which is the paper's and Leviathan et al.'s: a speculative round
commits `gamma` tokens for `T_draft + T_verify(K)`, against `gamma x T_decode(1)` for ordinary
decoding. Write `c(K) = T_verify(K) / T_decode(1)`, the block cost in units of single steps, and `d`
for the drafter's cost in the same units. Then

```
speedup(gamma, K) = gamma / (d + c(K))
```

Everything a new module can change is on the left: `gamma` and `d`. `c(K)` is a property of this
runtime and is what this experiment measures. **If `c(16)` is near 16, no drafter can win and the
module is dead whatever FlashMap achieves on an H200.**

## What is measured

`models-bench`'s committed `SpeculativeVerificationBenchmark`, which already measures exactly this
pair and asserts in setup that the two paths produce identical tokens:

- `sequential()` -- K separate `forwardTransient` calls, the ordinary decode path.
- `batched()` -- one `verifyTransient` over the same K tokens, the speculative path.

Swept over `draftLength` in **1, 2, 4, 8, 16, 24, 32** at two context lengths (**256** and **2048**)
and on two models, so the curve is not a property of one size or one prompt length:

| model | why |
| --- | --- |
| `Meta-Llama-3.1-8B-Instruct-Q4_K_M` | the arm this fleet's quality decisions chose; 4.92 GB, the decode this would accelerate |
| `Llama-3.2-1B-Instruct-Q4_K_M` | a size where weight traffic per step is 6x smaller, which is where batching should pay least |

Two knobs were added to make that sweep possible, both in benchmark code and neither in the runtime:
`-Pjmh.params` for benchmark parameters, and the benchmark now honours a context length the caller
already set instead of overwriting it.

## Decision rule, fixed now

From the measured `c(16)`, with `d = 0.06` -- a 0.5B drafter against an 8B target, both
bandwidth-bound, so the parameter ratio is the estimate, and it is an estimate and labelled one --
and with `gamma` taken from FlashMap's Table 1 at K=16 on Qwen3-8B: **6.00** (its weakest, MT-Bench
chat, 1 step) and **8.31** (its strongest, GSM8K, 2 steps).

- **PASS, build the module**: projected `speedup(6.00, 16) >= 2.0x` **and** at least 1.5x the
  end-to-end speedup the shipped n-gram drafter achieves on the same host. A drafter that doubles
  decode on the weakest domain, and clearly beats what is already there, earns the work.
- **NULLIFY, do not build it**: projected `speedup(8.31, 16) < 1.3x`, or `c(16) >= 12`. Either means
  the batching this runtime can do is too weak for any drafter to exploit, and FlashMap's gains do
  not transfer from an H200 to these kernels.
- **INCONCLUSIVE** between, in which case the curve and the implied speedup as a function of `gamma`
  are reported and the module waits for the second stage rather than being argued for.

The n-gram baseline is measured on the same host in the same session, because "better than nothing"
is not the bar -- "better than what Models already does" is.

## What this cannot settle, by construction

- **It does not measure `gamma` for a FlashMap drafter against a quantised target.** The published
  acceptance lengths are for an unquantised Qwen3-8B. Q4_K_M moves the target's argmax, and
  acceptance is exactly what that erodes. This experiment projects with the paper's `gamma` and says
  so; stage two measures ours on a GPU and is not started until this stage passes.
- **It does not measure `d` on our kernels.** 0.06 is a parameter-ratio estimate. A DiT with
  cross-attention is not a transformer decoder and may be worse per parameter.
- **CPU only.** Models' GPU path is a separate decode regime with its own transfer costs, measured
  elsewhere in this journal.
- **Greedy proposals.** The benchmark drafts with the target's own argmax, so it measures the cost
  of verification, not the quality of a draft.

## Host

One Hetzner `ccx33`, 8 dedicated vCPU, idle, provisioned and torn down by
`articles/models-deep-dive/tools/isolated_host.sh` -- the same helper the article's experiments use,
with its teardown trap and firewall janitor. Recorded here rather than copied into this repository:
a second copy of provisioning logic is how an orphan host happens, and this journal has a page about
that.

**Ceiling**: 2 hours on one host. Stops and reports what finished.

---

# Result: NULLIFIED. No drafter module.

Measured 2026-10-21 on one idle Hetzner `ccx33` (8 dedicated vCPU, AMD EPYC), JDK 25, the benchmark
jar built on the workstation and shipped so the measured host ran nothing but the measurement. Model
digests in [`model-digests.txt`](model-digests.txt); the 8B's `7b064f58…` is the artifact the
ModelJars catalog pins. Raw JMH JSON per cell, derived numbers in [`RESULTS.json`](RESULTS.json) by
[`analyse.py`](analyse.py).

## The number the rule reads

`c(K)` is the cost of verifying a K-token block in units of one decode step.

| cell | one decode step | c(4) | c(8) | **c(16)** | c(32) | marginal cost per extra token |
| --- | --- | --- | --- | --- | --- | --- |
| 8B, ctx 256 | 559.6 ms | 3.23 | 6.39 | **14.59** | 25.43 | ~0.80 steps |
| 8B, ctx 2048 | 560.6 ms | 3.38 | 6.70 | **12.81** | 26.73 | ~0.82 steps |
| 1B, ctx 256 | 101.1 ms | 3.33 | 6.67 | **13.14** | 27.14 | ~0.84 steps |
| 1B, ctx 2048 | 111.4 ms | 3.11 | 6.16 | **12.41** | 23.56 | ~0.73 steps |

`c(16)` is **12.4 to 14.6**, against a nullify threshold of 12. Projected end-to-end speedup with
FlashMap's published acceptance lengths at K=16 and a drafter costing 0.06 of a target step:

| | gamma = 6.00 (its weakest, chat) | gamma = 8.31 (its strongest, GSM8K) |
| --- | --- | --- |
| projected speedup | **0.41x to 0.48x** | **0.57x to 0.67x** |

Both below 1.0. A perfect drafter with the best acceptance length in the paper would make this
runtime **slower** than ordinary decoding. Both nullify conditions fire in all four cells, so the
n-gram baseline the rule also asks for was not collected: nothing it could have said changes a
projection that never reaches 1.0x.

## The batched path really was running

The whole verdict rests on `batched()` not secretly being the `batchSize = 1` fallback inside
`LlamaForwardPass.verifyTransient`, so that was measured rather than assumed -- the default plan
against `-Dmodels.purejava.prefillBatchSize=1`:

| mode | K | default plan | forced unbatched | ratio |
| --- | --- | --- | --- | --- |
| `batched` | 8 | 3,541.8 ms | 4,403.9 ms | **1.24x** |
| `batched` | 16 | 7,500.3 ms | 8,751.1 ms | **1.17x** |
| `sequential` | 8 | 4,420.0 ms | 4,408.7 ms | 1.00x |
| `sequential` | 16 | 8,971.0 ms | 8,892.4 ms | 0.99x |

The knob moves `batched` by 17-24% and leaves `sequential` alone to within noise, which is the
control: a setting that changed both, or neither, would mean the measurement was of something else.
So the batched kernel is active, and what it buys over a plain loop is **about 20%**.

## What this says, and what it does not

**Says.** Speculative decoding's premise is that verification is nearly free because decode is
memory-bound and K positions re-read the weights once. On Models' pure-Java CPU backend that premise
does not hold: each extra token in a verified block costs ~0.8 of a decode step, flat from K=4 to
K=32, and **the same on a 1B as on an 8B** despite six times less weight traffic. A drafter cannot
fix that; it is upstream of anything a drafter module would change.

**Does not say.** It does not say FlashMap is a weak technique -- its H200 numbers are not in
dispute and were not re-measured here. It does not say our batched kernel is deficient: a flat
marginal cost is equally consistent with the pass being **compute-bound** at these block sizes on
eight cores, which would be physics rather than a defect. This experiment cannot tell those apart,
and the claim is not made.

## Disposition

**Discarded**: no FlashMap-class drafter module in Models. Stage two -- measuring acceptance length
against a quantised target on a GPU -- is not started, because its result could not change this one.

**Open, and not taken here**: whether ~20% is all that batched verification can buy on this
hardware, or whether the kernel leaves amortisation on the table. The measurement that separates
them is the one this journal already uses for kernel questions -- achieved throughput at K=16
against llama.cpp's batched prefill on the same host, as an independent oracle. That is a question
about the prefill path, which serves every prompt this library processes, not about speculative
decoding, which serves none of them today.
