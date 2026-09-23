# Side by side — Integrallis Decisions vs TypeSafe Jev

Video: `sidebyside.mp4` (24.5 s), `sidebyside.gif`, raw terminal capture `sidebyside.cast`.
Re-run it with `run_demo.sh`. Both arms answer the same ten questions about the same 404-token
master services agreement, and each timing is taken inside the process that answers, so neither
side is credited or charged with the other's network path.

## Result

| | TypeSafe Jev | Integrallis |
| --- | ---: | ---: |
| correct of 10 | **10** | **7** |
| total wall time | **2.75 s** | 18.86 s |
| one-off prefill | n/a | 8.98 s |
| marginal per question | **0.275 s** | 0.989 s |
| document sent to a server | 10x | 0 |
| input tokens billed | 7,292 | 0 |
| prefix physically shared | n/a | yes, 10/10 |
| runs offline | no | yes |

Jev `jev-1.13.0` over its hosted `/v1/systemone`. Ours is Granite 4.1 3B Q4_K_M plus an 82 KB
head, on one 8-core AMD EPYC-Milan (Hetzner ccx33), rust-ffm kernel, base pinned by sha256
`662b0626cd58f443…`.

We lose on both axes measured here. The gap is 6.9x on wall time and three questions on accuracy.

## The three misses

Q3 may Acme use customer data to train models, Q5 does the liability cap apply to data breaches,
Q7 what is the early termination fee. The contract answers all three — by *exception or negation*
in each case ("shall not use", "this cap does not apply to Section 4", a fee stated as a condition
of terminating for convenience). The head was fitted on squad2, which teaches whether a matching
span is present, so a clause that settles a question by excluding something reads as absence.
No amount of hardware moves this; it needs training data whose labels turn on entailment.

A CUAD-fitted head was cut to attack exactly that and is **not shipped**: 0.5600 accuracy against
a 0.7200 majority floor, 68% of sealed items truncated, temperature pinned at the fitter's 64.0
ceiling. That is the known CUAD defect — gold spans outside the 4,000-token window mean the labels
describe text the model was never shown.

## Speed, and what actually moved it

| kernel | prefill | marginal/question | total |
| --- | ---: | ---: | ---: |
| pure-java (2019 Intel Mac) | 126.28 s | 8.558 s | 211.87 s |
| pure-java (8-core EPYC) | 85.44 s | 5.504 s | 140.47 s |
| **rust-ffm (8-core EPYC)** | **7.69 s** | **0.948 s** | **17.17 s** |

The native kernel is 8.2x on total wall time and 11x on prefill. Both earlier runs were the
fallback path: no native kernel had ever been bundled for any platform, because
`prepareNativePlatformResources` depends on a cargo build that had not run. The kernel used here
was compiled from our own crate on the target box.

Two things this cost, recorded so they are not repeated:

- A prediction was published from the fallback path ("the Linux box will give us ~2.26 s/item")
  without first checking which kernel had loaded. The demo now prints the kernel on every run.
- `native.properties` was hand-written with `abi=1`; the loader requires `abi=5`. It failed closed
  with a clear message, which is the correct behaviour, but the run before that silently used the
  fallback and produced timings within 1% of the previous one — which is exactly what a
  no-op change looks like when a switch is not observable.

## The trap this run nearly walked into

The HuggingFace copy of `granite-4.1-3b-Q4_K_M.gguf` is **byte-for-byte the same size** as ours,
2,099,501,664, with a different sha256 (`87320650…` against `662b0626…`). Different weights, same
name, same size. A head reads hidden states from specific weights; fed the wrong ones it returns
confident nonsense and nothing errors.

The artifact recorded `baseModel` as a *name*, which would not have caught it. As of v0.2 it
records the base's sha256 and `requireBase()` refuses to run against a file that disagrees, before
any inference is spent. Two tests cover it.

## Honest scope

These ten questions are a demo, not a benchmark: one document, one family, gold labels written by
us. The measured benchmark position is in `../2026-09-21-jevbench-baseline/`. This artifact is a
Noul head and answers 24 of JevBench's 72 public items; choice and score have no trained head.
