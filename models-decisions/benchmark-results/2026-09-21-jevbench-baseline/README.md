# JevBench public cohort — first measurement of our stack

Run 2026-09-21 on Hetzner `decisions-bench-noul-20260920` (ccx33, 8 vCPU AMD EPYC-Milan, AVX2,
30 GiB, Ubuntu 24.04.4), Temurin 25.0.4.1, `vectors-core` 0.1.22 Panama path, rust-ffm kernels.

Harness: `fstandhartinger/jevbench` (MIT), `datasets/public/original.jsonl`, 72 decisions —
24 noul, 36 choice, 12 score. Scored by the harness's own `jevbench.cli summarize`.

## Two arms, both run here

| | Jev 1.13.0 | Granite 4.1 3B Q4_K_M + shared-prefix candidate scoring |
| --- | ---: | ---: |
| accuracy | **0.9861** (71/72) | **0.6667** (48/72) |
| ECE | **0.0333** | **0.2062** |
| Brier | — | 0.5034 |
| ordinal MAE | 0.005 | 0.2722 |
| latency p50 / p95 | 0.267 s / 0.341 s | 2.264 s / 4.323 s |
| schema validity (strict) | 1.0 | **1.0** |
| operational success | 1.0 | **1.0** |
| renormalized | 0 | **0** |
| probability source | native | **native** |
| price / 1,000 decisions | $0.0154 | no provider tariff (local) |
| prefix sharing proven | n/a | **72/72** |

Jev was reached over its production API with our own key; cost of that run was USD 0.0011. Our arm
ran locally, so cost is reported as absent rather than zero: compute is not free, only unbilled.

## Per family, our arm

| family | n | accuracy |
| --- | ---: | ---: |
| extraction | 12 | 0.9167 |
| ordinal | 12 | 0.9167 |
| intent | 12 | 0.7500 |
| policy | 12 | 0.6667 |
| adequacy | 12 | 0.4167 |
| routing | 12 | 0.3333 |

## What this establishes

**The base carries real signal untrained.** 0.6667 with no LoRA, no scalar head and nothing fitted,
against Laya's published Intelligence of 63.2 at 421M. A prediction of "40s to 50s" was recorded
before the run and was wrong on the low side.

**The architecture's claims held and were proven, not asserted.** Prefix sharing confirmed by the
backend on all 72 tasks. Strict schema validity 1.0 with zero renormalisation, because the answer
space is a type rather than a validation step. Only 72 candidate-token forward passes across all 72
tasks: a single-token candidate is free once the state is evaluated, since the state's own logits
already hold its probability.

## What this does not establish, and where the work is

**Calibration is the blocker, not accuracy.** ECE 0.2062 against Jev's 0.0333. The bin table shows
why: 36 of 72 predictions land in the 0.9-1.0 confidence bin and are right 77.8% of the time. The
model is confidently wrong at scale, which is exactly what a trained scalar head and a properly
fitted temperature exist to correct.

**Routing at 0.3333 is chance**, and adequacy at 0.4167 is close to it. The base reads things out
well (extraction, ordinal both 0.9167) and judges between options poorly -- which is the thing a
decision model is for.

**Speed is the weak axis.** 2.264 s p50 against Jev's 0.267 s. A 3B is large for this task; the
400M-600M class is where the cost and speed axes are won.

**n = 72.** The 95% interval on 0.6667 is roughly plus or minus 11 points. Treat it as "mid-sixties",
never as a precise figure, and never compare it to another system's number across a margin this
sample cannot resolve.
