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

---

# Amendment 1 — enterprise corpus, energy, and a hardware sweep

**Written 2026-09-20. No split has been opened and not one hidden state has been harvested**, so
this is still pre-registration rather than a revision made after seeing a result. The original gates
are unchanged; this adds a corpus, an axis, and two hardware arms.

## A1.1 Third corpus — CUAD, because enterprise usage is the case worth making

SQuAD and MS MARCO are academic and web genres. The buyer this tier is aimed at reads contracts,
policies and support tickets, so a third corpus is added in that genre.

| | CUAD test split |
| --- | --- |
| Source | `github.com/TheAtticusProject/cuad` `data.zip`, SHA-256 `f8161d18bea4e9c05e78fa6dda61c19c846fb8087ea969c172753bc2f45b999a` |
| Contents | 102 commercial contracts, 4,182 clause questions |
| Negative class | 2,938 unanswerable (0.7025) |
| **Majority-class floor** | **0.7025** |
| Licence | Apache-2.0, published by the Atticus Project |

CUAD is the most awkward of the three and that is why it is in. Its base rate is skewed 70/30 rather
than near-even, which is the enterprise reality the other two do not test: most questions asked of a
document have no answer in it. Accuracy is a weak metric against a 0.70 floor, so on this corpus the
weight falls on agreement and calibration, and that is stated now rather than discovered later.

**The gate must hold on all three.** A pass on two and a failure on the third is a failure, reported
as the disagreement it is.

## A1.2 Long documents — a fixed window, and the ceiling it imposes, reported

Contracts are far longer than SQuAD paragraphs or MS MARCO passages. The protocol therefore takes a
**fixed 4,000-token window from the start of each contract**, applied identically to answerable and
unanswerable items, with no answer-aware cropping of any kind — cropping around a known span would
leak the answer's location for the positives and have no counterpart for the negatives.

**The fraction of answerable items whose gold span falls outside the window is measured and reported
beside every CUAD figure.** Those items cannot be answered correctly by any model at any quality, so
they cap achievable accuracy for a reason that has nothing to do with what is being measured. An
over-long input has silently destroyed two datasets in this tree before; the mitigation is to handle
the limit and say how often it was hit.

## A1.3 Energy — an axis, measured where possible and approximated where not

Latency alone understates the case. A decision that avoids a decode loop should cost far fewer
joules, and for an enterprise buyer reporting under CSRD that is the number that matters.

**What this host can and cannot do, checked rather than assumed.** The Hetzner VM exposes
`/sys/class/powercap` empty, no `/dev/cpu/*/msr`, no `rapl` or `amd_energy` module and no `hwmon`
entries. **Joules are not measurable on it.** What is available is exact cgroup CPU accounting.

Reported per arm, each labelled by how it was obtained:

| Quantity | Method | Label |
| --- | --- | --- |
| CPU-microseconds per decision | cgroup `cpu.stat` delta | **measured** |
| Watt-hours per decision, CPU arms | CPU-seconds x host TDP fraction, TDP read from the part's specification | **computed, approximate** |
| Watts and joules, GPU arm | NVML via `nvidia-smi` sampling across the run | **measured** |

The approximation is explicitly an approximation and may not be quoted as a measurement. Its purpose
is an order of magnitude, which is the scale the comparison actually turns on.

**Honesty condition on any energy claim.** The per-decision figure is the defensible one. Total
energy reduction is not: efficiency gains of this kind historically raise total consumption rather
than lower it, and the model this tier imitates is named after the economist who described exactly
that. No write-up may convert a per-decision saving into a sustainability claim.

## A1.4 GPU arm

`TornadoGgufBatchedMatrixKernel` implements `GgufBatchedMatrixKernel`, so it can be injected through
`PureJavaBackend.load(path, kernel)`. That arrangement keeps the backend a
`SharedPrefixInferenceBackend` — the prefix sharing the whole tier depends on survives — while the
projections execute on the device. Q4_K and Q6_K projection support is on `models` `origin/main`
(#194), which is what Granite 4.1 3B Q4_K_M needs.

- Host: Vultr `vcg-a16-3c-32g-8vram`, NVIDIA A16, 8 GiB VRAM, about USD 0.236/hour, three locations
  currently available. A 2.0 GB Q4_K_M fits with room for the cache.
- The same pre-registered gates apply unchanged. The GPU arm is a hardware comparison, not a
  different experiment.
- **`backend-cuda` is excluded.** Our own Rust PTX kernels carry a known Q6_K parity defect that is
  not bit-exact past one super-block (`models` PR #198, reproduced on an A40). Measuring a decision
  quality gate on a kernel known to be numerically wrong would produce a number about the defect.
  The arm may be added once that is fixed and its parity gate is green.

## A1.5 TPU — not reachable from this stack, and saying so

There is **no JVM-native path to a Google TPU**, and this is a structural absence rather than work
not yet done. TornadoVM targets OpenCL, PTX and SPIR-V. ONNX Runtime ships no TPU execution
provider. `libtpu` is reached through XLA from C++ or Python. A Coral Edge TPU is int8 with a few
megabytes of SRAM and cannot hold a 3B model at all. `models-backend-apple` is a bridge to Apple
Foundation Models, a different model entirely, not an accelerator for our GGUF artifact.

What is possible, if wanted, is an **external hardware control**: the same model family converted to
JAX/Flax and run on a Cloud TPU, under a small pinned Python environment of the kind this project
already permits for third-party reproduction. It would bound what the best available hardware does.
It would say nothing whatever about the JVM tier, because it would be a different implementation of
a different graph, and it must never be reported as an arm of this experiment. **Not adopted here;
recorded as available on request.**
