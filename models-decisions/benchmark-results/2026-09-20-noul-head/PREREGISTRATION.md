# Pre-registration — Noul head vs decode verdict

**Written and committed 2026-09-20, before any split was opened and before any hidden state was
harvested.** Nothing below may be changed once the sealed split is read. If the gate fails, the
failure is the result.

## Hypothesis

A linear head reading the base's final normalized hidden state reproduces the verdict that the same
base reaches through an autoregressive decode loop, closely enough to replace it, and its
probabilities are calibrated well enough to be worth reading.

## Artifact and host, both fixed

| | |
| --- | --- |
| Base | `granite-4.1-3b-Q4_K_M.gguf`, SHA-256 `662b0626cd58f443baea23559b469df6576a81d349649c59413b36a9fb32eb29`, verified equal to the ModelJars catalog |
| Hidden width | 2560 |
| Backend | `rust-ffm` (`backend-native`, `jmodels-kernels` 0.3.42 built on this host), because the qualified answerability evidence is on `rust-ffm` |
| Host | Hetzner ccx33, 8 vCPU AMD EPYC-Milan, AVX2, 30 GiB, Ubuntu 24.04.4, Temurin 25.0.4.1 |
| Decode | greedy, temperature 0. Deterministic by construction |

## Corpora — two, in different genres, both published and neither authored here

| | SQuAD v2 dev | MS MARCO v2.1 validation |
| --- | --- | --- |
| SHA-256 | `80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8` | `a07a87f483e602f5812573bff45109e5cbd934773c1c364366f176dd791e44d7` |
| Items | 11,873 | 101,093 |
| Negative class | 5,945 unanswerable (0.5007) | 45,457 no-answer (0.4497) |
| **Majority-class floor** | **0.5007** | **0.5503** |
| Genre | Wikipedia, crowd-written questions | real web search queries and retrieved passages |

The two-dataset requirement is not decoration. One dataset measured carefully has already produced a
confident wrong answer here. **The gate must hold on both.** A pass on one and a failure on the
other is a failure, and is to be reported as the disagreement it is.

## Sampling and splits, fixed before harvest

- N = 1000 per dataset, drawn uniformly at random with seed **20260920**, preserving the natural
  base rate rather than rebalancing, so the realized floor is reported as it falls.
- Split 500 train / 200 calibration / **300 sealed test**, disjoint, same seed.
- Splits are grouped by source passage: no passage may appear in two splits. This forecloses the
  leakage that would otherwise inflate agreement.
- The sealed test split is read **once**, after the head and the temperature are frozen.

## Procedure

1. For each item, build the prompt from a template fixed now and recorded in the run manifest.
2. Harvest, in one pass per item: the decode verdict from the base, and the final normalized hidden
   state at the last prompt position.
3. Fit the linear head on the train split only, by deterministic logistic regression with fixed
   iteration count and fixed L2, no early stopping of any kind.
4. Fit the calibration temperature on the calibration split only, by minimizing negative log
   likelihood.
5. Open the sealed test split once and report.

## Gates

**Primary.** Agreement between head verdict and decode verdict on the sealed split **>= 0.95**.

**Secondary.** Expected calibration error of the head on the sealed split **<= 0.05**, 15 equal-width
bins.

**Performance.** Median per-decision cost strictly below one decode turn, same host, same process,
**n = 3**.

**Both gates must hold on both datasets.**

## Sanity condition, declared in advance

If the decode verdict's own accuracy against the published labels does not exceed that dataset's
majority-class floor by at least **0.10**, then the base cannot do this task, agreement is agreement
with a coin flip, and the arm is reported **UNINFORMATIVE** — neither pass nor fail. This is written
down now precisely so a high agreement with a useless teacher cannot later be read as success.

## On sample size

Given a fixed artifact, greedy decode, and fixed seeds, agreement and calibration error are
deterministic functions of the inputs rather than draws from a distribution: repeating the harvest
returns the identical number, so n = 1 is exact for them and reporting error bars would be
theatre. Latency is the stochastic quantity, and carries n = 3.

## What this design cannot test

- Whether the result transfers to any base other than Granite 4.1 3B, or to any task other than
  answerability. It cannot, and no claim of that kind may be drawn from it.
- Whether a from-scratch encoder (arm C) would do better. That is a separate arm.
- Anything whatever about Jev, which we have not run and do not have access to.
- Whether calibration measured on these two corpora holds on a customer's data. It is reported per
  dataset and never pooled, for exactly this reason.

## Stopping rule

The sealed split is opened once. No hyperparameter may be changed and the split re-read. If the
gate fails, the recorded outcome is a failure, and any subsequent attempt is a new experiment with
a new pre-registration and a fresh split.
