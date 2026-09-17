> **Frozen at this commit, before any fused output exists.** This is a faithful transcription of the
> pre-registered protocol "Logit Fusion Control Study", version 2 (HTML source
> `logit-fusion-study.html`, v1 written 2026-09-17, v2 adds quantization and reasoning-aware voting,
> before any run). Hypotheses, arms, gates, decision rules, the stop rule and thresholds are not to
> be changed after this commit. Any later deviation is recorded as a dated amendment below the
> original text, never as an edit to it.

Provenance tags used in the source are kept inline: **[measured]** checked here, in our code or data;
**[read]** read from a paper or doc; **[believed]** estimate or recall, to be tested.

# Token-level logit fusion of small same-family models: a controlled study

Study protocol · draft for pre-registration · Integrallis Models / ModelJars

Can several small models that share a tokenizer, combined one token at a time, beat their best
member? The protocol also tests fusion against the controls that usually win: voting, repeated
sampling, and one bigger model at the same memory.

Status: v2, implementation started · v1 written 2026-09-17 · v2 adds quantization and
reasoning-aware voting, before any run · Decision rules frozen before any fused output exists.

## Question

On public reasoning benchmarks, does fusing the next-token distributions of small models that share
a tokenizer beat (1) the best member alone, (2) answer-level voting among the same members, (3)
repeated sampling on the largest member at matched compute, and (4) one larger model of the same
family at matched memory?

A frontier-model comparison is secondary and optional. It is only admissible if we run the frontier
model ourselves, on the same items, prompts and scorer. Numbers taken from leaderboards or model
cards are not an arm.

## Where the idea came from, and what we are not reproducing

A Towards AI article (14 Jul 2026) claims three "tiny" local models, fused, "matched the reasoning of
Anthropic Fable 5". A verification pass found no benchmark, no scores and no frontier numbers behind
that headline **[read]**. Its code is also unsound as published **[read]**:

- One tokenizer feeds models with different vocabularies (its examples, Llama 3 8B and Gemma 2 9B),
  so the stacked logits don't line up.
- It adds raw logits rather than log-probabilities, so the 0.5 / 0.2 / 0.3 weights don't mean what
  they claim.
- It emits one token, with no generation loop and no KV cache, and runs the members sequentially
  while claiming parallelism.

The underlying technique, ensembling at the token level, has real prior work: GaC, DeePEn, UniTE,
and product-of-experts or contrastive decoding (see the survey arXiv 2502.18036) **[read]**. This
study tests a corrected version of that idea. It is not a reproduction of the article.

## The technique under test

Members M1…MK share one tokenizer, byte-identical vocabulary and special tokens. Each keeps its own
KV cache. At each decode step every member produces logits for the same prefix. The step then runs
as follows:

```
ℓᵢ = log_softmax(logitsᵢ)                        # per member, removes the per-position offset
product of experts:   s(t) = Σᵢ wᵢ · ℓᵢ(t)          # weights on the simplex, Σ wᵢ = 1
mixture of experts:   s(t) = log Σᵢ wᵢ · exp(ℓᵢ(t))
p(t) = softmax(s / T)                              # the same sampler and constraints as single-model arms
token → append to every member → forward(token, position) on every member
```

- **Implementation:** a fusion loop over `InferencePipeline.prefill` / `forward`, which return
  logits with KV retained, and the public `Sampler` **[measured]**. About 100 lines; there is no
  Python in the measured path.
- **Weights:** either uniform, or tuned by grid search (step 0.1) on a disjoint development split.
  Never tuned on test items.
- **Parallelism:** members step on one worker thread each. Wall-clock time and core contention are
  measured, not assumed.
- **Composition, not hybrid:** in our terms this is a composition, with independent weights and KV
  per member. It is not a KV-sharing hybrid. The hybrid form is listed under follow-on variants.
- **KV:** members of different sizes cannot share KV, since layer counts, KV heads and widths all
  differ. Translating KV between models was already tried and rejected in the hybrid campaign
  **[measured]**. The KV that is reusable here is each member's own prompt prefix across repeated
  samples, so SC-k, VOTE and RERANK reuse it. That keeps those controls from paying a prefill they
  wouldn't pay in production.
- **Reasoning-aware aggregation (v2):** with thinking enabled, each member's output splits into a
  reasoning trace and a final answer. For each trace three signals are extracted: the answer the
  reasoning itself reaches (a_r), the answer stated to the user (a_u), and the mean log-probability
  over the answer tokens (c). They feed three aggregation rules:
  - consistency filter: drop votes where a_r ≠ a_u;
  - confidence weighting: weight each vote by c, after per-member temperature calibration on the
    development split;
  - cross-member rerank: every member scores every distinct candidate answer by prefill-only
    log-likelihood given the question, and the fused score picks the winner.

  A mismatch between reasoning and answer is a reliability signal, not proof of unfaithfulness. The
  study can only measure whether it improves accuracy.

## Hypotheses

| ID | Hypothesis | Primary comparison |
|----|------------|--------------------|
| H1 | Fusion beats its best single member. | F-tuned vs best member. The best member is chosen on the development split. |
| H2 | Fusion beats answer-level majority voting among the same members. | F-tuned vs VOTE |
| H3 | Fusion beats repeated sampling on the largest member at matched decode compute. | F-tuned vs SC-k |
| H4 | Fusion is not worse than one larger same-family model using the same memory (non-inferiority margin 2 points). | F-tuned vs BIG |
| H5 | Fusion's gain comes from combining members, not from one dominant member. | Leave-one-out pairs; member-win shares |
| H6 | Fusion degrades less than a single model under heavier weight quantization, because quantization noise is partly independent across members. | (F-tuned-Q8 − F-tuned-Q4) vs (BIG-Q8 − BIG-Q4) |
| H7 | Reasoning-aware voting (consistency filter or confidence weighting) beats plain majority voting over the same thinking-mode samples. | VOTE-consist, VOTE-conf vs VOTE-think |
| H8 | Cross-member reranking of candidate answers matches or beats fused decoding at a fraction of its decode compute. | RERANK vs F-tuned, with compute per item |

Prior **[believed]**: H1 is plausible, by a few points. H3 and H4 are the likely failures, because a
single larger model usually wins at equal memory. The study exists to settle that, not to confirm
it.

## Models and data

### Members and controls

| Role | Model | Quant | Notes |
|------|-------|-------|-------|
| Member A | Qwen3 0.6B | Q8_0 | Tokenizer matches B **[measured]**. |
| Member B | Qwen3 1.7B | Q8_0 | Already qualified path in Models. |
| Member C | Qwen3 4B | Q8_0 | Tokenizer identity must pass gate G0. |
| Quantization factor (H6) | Members A, B, C and BIG | Q4_K_M | The same models at the lower bit width, pinned the same way |
| Memory-matched control | Qwen3 8B | Q4_K_M | About the summed member weight footprint (≈ 7 GB) **[believed]**. Confirmed from file sizes before the run. |
| Replication family (optional) | Granite 4.x sizes | Q8_0 | Only if tokenizers are identical. It tests whether the result generalises beyond one family. |

- Thinking mode is a factor, not a free choice. Fused decoding arms run non-thinking only, since
  fused thinking traces multiply an already K-fold decode cost. Member, VOTE, SC-k and RERANK arms
  run both non-thinking and thinking, so H7 compares like with like. Every arm within a mode uses one
  chat template and one answer instruction.
- Licences: Qwen3 and Granite are Apache-2.0 **[believed]**, to be confirmed against the pinned model
  cards. Llama and Gemma are excluded because of flow-down terms and vocabulary mismatch.
- GGUF revision, SHA-256 and tokenizer hash are pinned per member and recorded beside every result.

### Datasets

All public, none written by us, all scored by program. No LLM judge anywhere.

| Dataset | Items | Scoring | Role |
|---------|-------|---------|------|
| ARC-Challenge test | 1,172 | Letter exact match, and fused log-likelihood over options | Powered; cheap first signal |
| GSM8K test | 1,319 | Final-number exact match | Powered; primary generative set |
| MATH-500 | 500 | Normalised answer match (a sympy-equivalence checker, unit-tested on gold answers) | Secondary |
| Post-cutoff set (e.g. a LiveBench math release newer than the members) | as released | Program-scored | Contamination check; directional only |
| GSM8K train (500 sampled, fixed seed) | 500 | Final-number exact match | Development split: weight tuning and best-member choice only |

## Arms

Every arm uses the same prompts, maximum output tokens, answer extractor and scorer. Decoding is
greedy (T = 0) for the primary result. As a robustness check, T = 0.7 is also run with three seeds.

| Arm | What runs | Why it is here |
|-----|-----------|----------------|
| A, B, C | Each member alone | Per-component baseline |
| AB, AC, BC | Pairwise fusion, weights tuned on development split | Leave-one-out: what each member adds |
| F-uniform | A+B+C, product of experts, equal weights | Fusion with no tuning |
| F-tuned | A+B+C, product of experts, weights tuned on development split | Primary treatment |
| F-mix | A+B+C, mixture rule, tuned | Which combination rule |
| F-article | A+B+C, raw-logit sum, 0.5 / 0.2 / 0.3 | Tests the rule as published, so its defect is measured rather than asserted |
| VOTE | A, B, C each answer; majority vote, ties go to the best member | Answer-level ensemble baseline |
| VOTE-think | As VOTE, with thinking mode on | Plain-vote baseline for H7 |
| VOTE-consist | VOTE-think, dropping votes whose reasoning answer differs from the stated answer (all votes kept if every vote is dropped) | H7: the delta between reasoning and communication |
| VOTE-conf | VOTE-think, with votes weighted by calibrated answer-token confidence | H7: confidence |
| RERANK | The distinct candidate answers from A, B and C, each scored by all members through prefill-only log-likelihood and fused with the tuned weights | H8: fusion used for verification, not generation |
| F-tuned-Q4, BIG-Q4 | F-tuned and BIG with every model at Q4_K_M | H6: quantization tolerance |
| SC-k | Member C sampled k times (T = 0.7), majority vote; k set so decode compute ≈ F-tuned | Compute-matched control |
| BIG | Qwen3 8B Q4_K_M alone | Memory-matched control |
| ORACLE | An item counts as correct if any member got it right | Upper bound on fusion's headroom; not a competitor |
| FRONTIER | A frontier model on the same items, prompts and scorer | Optional. Paid API; needs explicit approval, otherwise reported as "not compared" |

## Validity gates

The gates run before any result is read. A failed gate stops the stage. It is never a reason to
adjust the result.

| Gate | Check | Guards against |
|------|-------|----------------|
| G0 | All members have byte-identical tokenizer vocabulary, merges and special tokens (hashes compared from GGUF metadata) | The article's vocabulary mismatch |
| G1 | Weight vector eᵢ (one member at 1, the rest at 0) reproduces member i alone token for token at T = 0, for every member, on 20 items | A fusion path that silently does something else |
| G2 | Fused log-probabilities match an independent re-implementation (NumPy over dumped member logits) within 1e-4 on 20 items and 50 steps | Combination-rule bugs |
| G3 | Per-token logs include: member argmax agreement rate, which member's argmax the fused token equals, fused entropy, and KL between the fused and each member distribution. An agreement rate of exactly 100% on a dataset is flagged. | A switch that never switches (fusion equivalent to one member) |
| G4 | The answer extractor scores 100% on gold answers rendered in the prompt's answer format | Scoring bugs that look like model effects |
| G5 | Truncation (maximum tokens reached) and extraction failures are reported per arm and dataset. An arm with more than 3% truncation is rerun with a higher cap. | A limit capping one arm's metric |
| G6 | The reasoning-answer and stated-answer extractors score 100% on 30 hand-constructed traces per dataset covering: agreement, disagreement, a missing thinking block and a truncated thinking block | The consistency filter measuring extractor bugs |
| G7 | Per-member confidence calibration is fitted on the development split only and frozen with the weights. Expected calibration error is reported before and after. | Confidence weighting leaking test data, or favouring one member by scale |

## Decision rules (frozen before any fused output)

- **H1 holds:** The paired bootstrap 95% interval for (F-tuned − best member) lies above 0 on at
  least two of ARC-Challenge, GSM8K and MATH-500. The Holm correction applies across H1–H4.
- **H2 holds:** Same criterion for (F-tuned − VOTE).
- **H3 holds:** Same criterion for (F-tuned − SC-k), with SC-k's decode compute within ±10% of
  F-tuned's as measured.
- **H4 holds:** The lower 95% bound of (F-tuned − BIG) is at least −2.0 points on at least two of
  the three datasets.
- **H6 holds:** The paired 95% interval of the difference-in-differences (fusion loss from Q8 to Q4,
  minus BIG's loss) lies below 0 on at least two of three datasets.
- **H7 holds:** The paired bootstrap 95% interval for (VOTE-consist − VOTE-think) or (VOTE-conf −
  VOTE-think) lies above 0 on at least two of three datasets. Holm correction across the two
  variants. The mismatch rate is reported per member either way.
- **H8 holds:** RERANK is non-inferior to F-tuned (margin 1 point) on at least two of three
  datasets, and its measured core-seconds per item are lower.
- **Practical claim:** Only if H4 holds and F-tuned's measured accuracy per joule (or per
  core-second, if energy isn't measurable) is at least BIG's on the same host. Otherwise the report
  says fusion is not a deployment option for us, whatever H1 says.
- **Reporting:** All arms are reported, including arms that lose. A flat result on a dataset that
  can't exercise fusion (ORACLE headroom under 3 points) is written as "no headroom", not "no
  effect".

**Stop rule:** After the 50-item GSM8K pilot, stop before the full runs if both hold: ORACLE headroom
over the best member is under 3 points, and F-tuned is at or below the best member. Record the pilot
as the result.

(Transcriber's note, not in the source: H5 carries no numeric decision rule in the protocol; it is reported descriptively from the
leave-one-out pairs and member-win shares.)

## Statistics and power

- **Paired tests:** every comparison is paired per item, using a paired bootstrap (10,000
  resamples) for intervals and McNemar's test as a check.
- **Power [believed]:** with 1,319 GSM8K items and about 15% discordant pairs, the standard error of
  the paired difference is about 1.1 points. The detectable effect at 80% power is about 3 points.
  ARC-Challenge is similar. MATH-500 detects only about 5 points, so it counts as the third dataset
  in the two-of-three rule, never alone.
- **Seeds:** the T = 0.7 runs report mean and spread over three seeds. They check robustness and do
  not enter the decision rules.
- **Effect sizes:** every result is reported as a point difference with its interval, never as a
  bare "significant".

## Stages, in dependency order

- **S0** Fusion loop, per-token logging, answer and reasoning extractors, voting, confidence
  calibration and reranking; plus a NumPy reference (tests first). *Exit when unit tests pass,
  including [1,0,0] equivalence on a toy model.*
- **S1** Pin members and controls; run gates G0–G2 and G4 on real weights. *Exit when every gate
  passes. G0 failing for member C swaps it for the next same-tokenizer size.*
- **S2** Pilot: 50 GSM8K items, all arms except FRONTIER. Measure tokens/s, peak RSS and
  core-seconds per arm, and replace every believed cost below. *Exit via the stop rule, or
  continue.*
- **S3** Tune weights and choose the best member on the development split. Freeze both in the
  results record. *Exit when the frozen weights are committed before any test-set fused run.*
- **S4** ARC-Challenge, all arms. Log-likelihood scoring makes this the cheap first full read.
- **S5** GSM8K, all arms.
- **S6** MATH-500, all arms.
- **S7** Post-cutoff set, members plus F-tuned plus BIG only.
- **S8** FRONTIER, only with explicit approval of the API spend.
- **S9** Apply the decision rules as written and publish the report. *The report includes every
  arm, every gate log and the full configuration.*

### Cost envelope [believed], replaced by pilot measurements

- **Workload:** GSM8K runs about 300 output tokens per item, so a fused arm is about 400k steps, each
  costing one forward pass per member. Pure-Java CPU throughput is likely in the low tens of fused
  tokens per second.
- **Hours:** that is about 5–10 hours per fused arm per generative dataset. Across all arms,
  datasets and seeds, expect several hundred core-hours.
- **Hardware:** ephemeral 16-vCPU bench hosts, one arm per host, with a watchdog on every host.
- **Hardware parity:** ARC-Challenge log-likelihood arms take minutes. Controls and treatments share
  the same host type, so compute comparisons are fair.

## Threats to validity, and what the study cannot test

- **Thinking-mode cost:** thinking arms can emit thousands of tokens per item. The cap is fixed per
  dataset from the pilot's 95th-percentile trace length, and it is the same for every thinking arm.
- **Contamination:** GSM8K, ARC and MATH are likely in pre-training data for all members. The
  post-cutoff set is the check. A gain that disappears there is reported as such.
- **Prompt favouritism:** one template can suit one member. The template is fixed from the model
  family's documented format, before any run, and is identical across arms.
- **Quantisation:** Q8_0 members against a Q4_K_M control mixes precision into the memory
  comparison. This is intended, because it is the deployable trade-off, but it is named in the
  report. A second BIG at Q8_0 is added if memory allows.
- **Tuning leakage:** weights and the best-member choice come from the development split only, and
  are frozen and committed before test runs.
- **Multiple comparisons:** Holm correction across H1–H4. Extra arms are descriptive.
- **Cannot test:** heterogeneous-vocabulary fusion (e.g. Llama + Gemma + Qwen), long-horizon agentic
  reasoning, code generation, and anything thinking mode would change. Each is out of scope, not "no
  effect".

## Follow-on variants (separately pre-registered, only after S9)

- **Entropy-gated fusion.** Run member A only. Consult B and C only on tokens where A's entropy
  exceeds a threshold tuned on the development split. This could reuse
  `TokenConfidenceSignal.fromLogits`, which already exists **[measured]**. The question is whether
  most of the gain survives at a fraction of the cost.
- **KV-sharing hybrid fusion.** One base with K activated adapters. The prompt prefix is computed
  once on the base and shared physically; the adapter distributions are fused per token after
  activation.
  - Why: this is the only form of fusion with real KV sharing in our runtime.
  - Decode cost: it becomes cheap only with batched multi-adapter decode, where K rows go through
    one base pass, each row applying its own low-rank delta. That is runtime work we don't have yet.
  - Memory: KV-cache quantization (F16/Q8) then shrinks the K diverging tails.
  - Prerequisite: reasoning adapters we don't have yet.
- **Contrastive fusion.** Use the small member as an "amateur": s = ℓ_C − α·ℓ_A. Same plumbing,
  opposite sign.
- **Cross-vocabulary fusion.** Top-k union mapping in the style of UniTE, so members from different
  families can be combined. This is a research spike, only if same-family fusion shows a gain worth
  generalising.

## Deliverables

- A fusion runner in models-bench, with configuration in data (members, weights, rule, gating)
  selectable from the command line, not recompiled per question.
- Per-item, per-arm results with configuration, including GGUF hashes, tokenizer hash, weights,
  sampler, JDK, backend and host.
- Gate logs G0–G5, pilot cost measurements, and the frozen weights commit.
- A report applying the decision rules exactly as written, with an explicit "what this cannot test"
  section.
- ModelJars composition entry only if the practical claim holds.

Sources: the verification report on the Towards AI article; the LLM ensemble survey arXiv
2502.18036; DeePEn arXiv 2404.12715; token-level ensembling across vocabularies arXiv 2502.21265;
"When to Ensemble" arXiv 2510.15346. Local references: Models `InferencePipeline`, `Sampler` and
`docs/proposals/models-composite.md`.
