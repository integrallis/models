# Entropy-gated logit fusion of same-tokenizer specialists: pre-registered follow-on

> Protocol v1, frozen 2026-11-25 before any gate, pilot or test-set run on the named weights. Hypotheses,
> arms, gates, decision rules, the stop rule and thresholds are not to be changed after a result is
> read. Changes before data go in a dated amendment at the end; nothing is edited in place.

This is the first follow-on variant the 2026-09-17 study listed under "Follow-on variants (separately
pre-registered, only after S9)": *entropy-gated fusion*. That study stopped at S2 by its stop rule, so
S9 was never reached; the owner re-opened the question on 2026-11-25 with a second source for the
same idea. Everything that study settled is reused, not re-argued: the datasets and their pins, the
answer extractors and gate G4, the tokenizer-identity gate G0, the `models-bench logit-fusion` runner,
the results-log rules and the reproduction guide.

## Question

When a lead model generates alone and consults its auxiliaries only at the tokens where its own
next-token distribution is uncertain, does the fused output on public reasoning benchmarks (1) beat
the best single member, (2) beat fusing at every token, (3) cost no more than a bounded multiple of
the lead alone, and (4) stay within two points of one larger same-family model at matched memory?

A fifth, descriptive question: does the recipe as published (weights 0.55 / 0.25 / 0.20, threshold
0.65 nats, Coder lead, Math and Instruct auxiliaries) do what its article claims on our runtime?

## Where the idea comes from, and what is already known

- **Our own measurement [measured].** `benchmark-results/2026-09-17-logit-fusion-study/`: Qwen3
  0.6B / 1.7B / 4B at Q8_0 fused at *every* token on a 50-item GSM8K pilot. Every fusion arm scored
  at or below the best member (member C 0.98; best fusion AC 0.96; three-way fusion 0.84 to 0.90).
  Two side findings: confidence-weighted voting 0.92 against plain voting 0.82, and cross-member
  reranking at about a tenth of fusion's core-seconds for 0.88 to 0.90. The stop rule fired.
- **SAFE, ICLR 2026 (arXiv 2510.15346) [read].** Ensembling at every token degrades long-form
  generation; choosing the positions recovers it. On MATH500 with Internlm3-8B (74.8) and Qwen2.5-7B
  (72.4), every-token UniTE scored 59.6 and UniTE with SAFE's position selection 77.4 while
  ensembling 3.82 % of tokens; on BBH 82.97 against 82.26 for the best member at 10.35 %. SAFE's
  selector needs every model's distribution at every candidate position (a drafter proposes, the
  verifiers run one batched pass), skips positions where the verifiers' argmax agree or the mean
  probability of the drafted token exceeds 0.5, and skips tokenisation-boundary mismatches. Its
  appendix says it "does not always guarantee superior performance compared to the best-performing
  individual model".
- **The article (Level Up Coding, 2026-10-19) [read].** Three Qwen2.5-7B specialists (Coder lead,
  Math, Instruct) in 4-bit; the lead runs alone while its Shannon entropy is under 0.65 nats, above
  that the auxiliaries catch up their KV over the unseen tokens and the weighted sum of log-softmax
  vectors is sampled. It reports no accuracy and no throughput, cites SAFE, UniTE (arXiv 2410.03777)
  and agreement-based ensembling (arXiv 2502.21265), and says a 4-bit 32B beats the trio on
  reasoning benchmarks. Its contribution for us is the configuration: same-size specialists rather
  than a size ladder, and a gate that costs nothing when it does not fire.

What is new here relative to September is therefore two factors at once: **the gate** (SAFE's
evidence says this is what every-token fusion was missing) and **specialist members of one size**
(the September members were a size ladder, and the largest member dominated). The arms separate the
two.

## The technique under test

Members share one tokenizer with byte-identical vocabulary, merges and special tokens (gate G0).
Each keeps its own weights and its own KV cache. One member is the **lead**; the rest are
**auxiliaries**. Per decode step:

```
logits_L        = lead.forward(token, position)              # the lead always steps
ℓ_L             = log_softmax(logits_L)
H               = −Σ_t exp(ℓ_L(t)) · ℓ_L(t)                  # Shannon entropy, nats
if H < τ:       s = ℓ_L                                      # fast path: auxiliaries idle
else:           for each auxiliary i:
                    ℓ_i = log_softmax(i.prefill(unseen_i, synced_i))   # catch-up over the delta, one call
                    synced_i = position + 1
                s = Σ_j w_j · ℓ_j     (PoE)   or   log Σ_j w_j · exp(ℓ_j)   (mixture)
token = argmax(s) at T = 0, otherwise the public Sampler over s (softmax(s / T))
```

- **Where it runs.** In the production path, not a bench loop: a `FusedInferenceBackend` in
  `models-runtime` implementing `InferenceBackend` and `RewindableInferenceBackend` over member
  backends, so `InferencePipeline`, generation sessions, chat templates and constraints apply to it
  unchanged. The `logit-fusion` runner gains a `gated:` arm that drives this backend, so what the
  study measures is what a ModelJars composition would ship. The September `FusionDecoder` stays as
  the every-token reference that gate G1b compares against.
- **Catch-up is a prefill, not a decode loop.** An auxiliary that has been idle for *d* tokens
  consumes them in one `prefill(delta, start)` call, the batched prompt path, whose per-token cost
  is a fraction of *d* single decode steps. How large a fraction on our CPU kernels is measured, not
  assumed (gate G8 checks it is also exact).
- **The route is printed.** Whether a batched native matrix kernel is installed on each member is
  recorded in every report (`PureJavaBackend.load(Path)` installs none; the ModelJars route does).
  The September study ran on the plain loader; this one records which loader each member used and
  runs the pilot on the kernel route.
- **Gate variants in this study:** entropy threshold only. SAFE's consensus selector needs the
  auxiliaries' distributions at every position, which on our runtime is the every-token cost unless
  the speculative verify path is cheap, and `benchmark-results/2026-10-21-speculative-ceiling/`
  measured that it is not (c(16) = 12.4 to 14.6 single steps). It is listed under follow-ons.
- **Composition, not hybrid.** Members share no KV. Under the boundary settled on 2026-09-12 this is
  a composition; the catalog kind is decided by the owner and does not change a threshold here.

## Hypotheses

| ID | Hypothesis | Primary comparison |
|----|------------|--------------------|
| H1 | Gated fusion beats its best single member. | G-tuned vs best member (chosen on the development split) |
| H2 | The gate is what matters: gated fusion beats every-token fusion of the same members. | G-tuned vs F-every |
| H3 | Gated fusion is cheap: its decode compute is at most 1.3× the lead alone, on the same host. | core-seconds per item, G-tuned vs L; trigger rate reported |
| H4 | Gated fusion is not worse than one larger same-family model at matched memory (margin 2 points). | G-tuned vs BIG |
| H5 | The recipe as published helps its lead. | G-article vs L, and vs best member; descriptive, with intervals |
| H6 | The gain, if any, comes from the specialists, not from the choice of lead. | G-best (uniform weights, best lead) vs G-tuned; leave-one-auxiliary-out pairs |

Prior **[believed]**: H2 is the most likely to hold (it is SAFE's central result, reproduced on two
model pairs). H1 is uncertain: at 1.5B the Math specialist is far stronger than the Coder on GSM8K
and MATH-500, so a Coder lead may fire the gate too rarely to be rescued, while a Math lead has
little to gain. H4 is the likely failure, as in September. The study exists to settle this, not to
confirm it.

## Members and controls

All Apache-2.0 **[read, from the model cards]**, to be confirmed against the pinned cards at S1.
Every file is pinned by revision and sha256 in `models.json` at S1 before any gate runs.

| Role | Model | Primary quant | Secondary quant | Source |
|------|-------|---------------|-----------------|--------|
| L (lead in G-article) | Qwen2.5-Coder-1.5B-Instruct | Q4_K_M (1.12 GB) | Q8_0 (1.89 GB) | official GGUF repo, revision `f86cb2c1fa` |
| M | Qwen2.5-Math-1.5B-Instruct | Q4_K_M | Q8_0 | **no official GGUF exists**; converted from the official safetensors (revision `aafeb0fc6f`) with a pinned llama.cpp release, the procedure of the September study's Amendment 1 §6; third-party quantisations are used only for the mechanics smoke and never in a measured arm |
| I | Qwen2.5-1.5B-Instruct | Q4_K_M (1.12 GB) | Q8_0 (1.89 GB) | official GGUF repo, revision `91cad51170` |
| BIG (memory-matched) | Qwen2.5-7B-Instruct | Q4_K_M (about 4.7 GB in two shards) | — | official GGUF repo, revision `bb5d59e06d` |
| BIG-3B (lower bracket) | Qwen2.5-3B-Instruct | Q8_0 (3.62 GB) | — | official GGUF repo, revision `7dabda4d13` |

- Members total about 3.4 GB at Q4_K_M and 5.7 GB at Q8_0 (the Math file's size is read after
  conversion). BIG at 4.7 GB sits between the two; BIG-3B brackets from below. H4 is read against
  BIG; BIG-3B is reported beside it.
- **Primary quant is Q4_K_M**, the article's regime and the quantisation ModelJars ships by default.
  Q8_0 is a secondary factor run only if the pilot's measured cost allows (Amendment 2 decides).
- Qwen2.5 has no thinking mode, so the September thinking factor does not apply.
- Tokenizer identity across L, M, I is gate G0; BIG is not fused and need not pass it.

## Datasets

The September pins, unchanged (`data-manifest.json` of that study, reused by path and sha256):
GSM8K test (1,319) with its 500-item development sample, MATH-500 (500), ARC-Challenge test
(1,172) with its development split. GSM8K and MATH-500 are where the Math member is strong and the
Coder lead weak, which is the article's own scenario; ARC-Challenge is where no member is a
specialist.

**What this cannot test, declared now:** a code-generation benchmark with execution (HumanEval+,
MBPP+). The runner has no sandboxed executor, so the article's "coder with math help on algorithmic
edge cases" premise is tested on math-heavy reasoning, not on code. A code set is a follow-on that
needs an executor first, and this study's report says "not tested" for code, not "no effect".

## Arms

Greedy decoding (T = 0) is the primary result; T = 0.7 with three seeds is a robustness check that
enters no decision rule. Every arm shares prompts, maximum output tokens, extractor and scorer.

| Arm | What runs | Why it is here |
|-----|-----------|----------------|
| L, M, I | Each member alone | Per-component baseline; the best member is chosen on the development split |
| BIG, BIG-3B | Qwen2.5-7B Q4_K_M alone; Qwen2.5-3B Q8_0 alone | Memory-matched controls |
| G-article | Gated, lead L, weights 0.55 / 0.25 / 0.20, τ = 0.65 nats, PoE | The recipe as published (H5) |
| G-best | Gated, lead = best development member, uniform weights, τ = 0.65 | Lead choice isolated from tuning (H6) |
| G-tuned | Gated, lead = best development member, weights and τ tuned on the development split | Primary treatment (H1–H4) |
| G-pairs | G-tuned with one auxiliary removed, each in turn | What each auxiliary adds (H6) |
| F-every | Every-token PoE, uniform weights, through the same backend with τ = 0 | The September treatment; gate-off control (H2) |
| VOTE-conf | L, M, I each answer; votes weighted by calibrated answer-token confidence | The September side finding, as the answer-level competitor |
| RERANK | Distinct candidate answers from L, M, I, each scored by all members by prefill-only log-likelihood, fused with uniform weights | The cheapest September arm, as the compute competitor |
| ORACLE | Correct if any member is correct | Headroom bound; not a competitor |

Tuning grid for G-tuned (development split only, never test items): weights on the simplex at step
0.1 with the lead's weight at least 0.4; τ ∈ {0.3, 0.5, 0.65, 1.0, 1.5, 2.0} nats. The frozen record
(weights, τ, lead, sha256) is committed before any test-set gated run; tuned arms refuse to run
without it, as the runner already enforces.

## Validity gates

A failed gate stops the stage. It is never a reason to adjust a result.

| Gate | Check | Guards against |
|------|-------|----------------|
| G0 | Byte-identical tokenizer vocabulary, merges, token types, scores and special tokens across L, M, I (`TokenizerIdentity`) | Vocabulary mismatch |
| G1 | τ = +∞ with any weights reproduces the lead alone token for token at T = 0 on 20 items | A gated path that silently does something else |
| G1b | τ = 0 with uniform weights reproduces the September `FusionDecoder` PoE run token for token on 20 items | The runtime backend and the bench reference disagreeing |
| G2 | Fused log-probabilities and gate decisions at the gated steps match an independent NumPy re-implementation over dumped member logits within 1e-4 on 20 items × 50 steps | Combination or entropy bugs |
| G3 | Per-token logs carry: fired or not, lead entropy, catch-up length per auxiliary, which member's argmax the emitted token equals, fused entropy, KL between fused and each member | Unobservable gating |
| G4 | The answer extractor scores 100 % on gold answers rendered in the prompt's format (exists) | Scoring bugs |
| G5 | Truncation and extraction failures reported per arm and dataset; an arm over 3 % truncation is rerun with a higher cap | A cap acting on one arm |
| G8 (new) | After a catch-up of *d* ≥ 1 tokens by `prefill(delta, start)`, the auxiliary's logits equal those of the same auxiliary having stepped the same tokens one `forward` at a time, within 1e-3 per logit, on 20 items; checked at d = 1, 2, 7 and 31 | The efficiency mechanism being inexact |

## Decision rules (frozen before any gated output)

Paired per item; paired bootstrap with 10,000 resamples for intervals, McNemar as a check; Holm
correction across H1–H4. Three datasets; "two of three" throughout, MATH-500 never alone.

- **H1 holds:** the 95 % interval for (G-tuned − best member) lies above 0 on at least two datasets.
- **H2 holds:** the same for (G-tuned − F-every).
- **H3 holds:** G-tuned's measured core-seconds per item are at most 1.3 × L's on the same host, on
  every dataset. The trigger rate and mean catch-up length are reported either way. If H3 fails, the
  report says the gate is not selective enough on this family whatever H1 says.
- **H4 holds:** the lower 95 % bound of (G-tuned − BIG) is at least −2.0 points on at least two
  datasets.
- **H5 and H6** are reported descriptively with intervals; they carry no pass/fail.
- **Practical claim:** only if H1, H3 and H4 all hold, and G-tuned's accuracy per core-second is at
  least BIG's on the same host. Otherwise the report says gated fusion is not a deployment option
  for us on this family, whatever H2 says about the gate.
- **Catalog consequence:** the ModelJars catalog carries an entry for the article's recipe only if
  the practical claim holds. The recipe schema and the assembly code are engineering deliverables
  and ship regardless; an entry is evidence, code is not.
- **Reporting:** every arm, including losers. A dataset with ORACLE headroom under 3 points over the
  best member is written as "no headroom", not "no effect".

**Stop rule.** After the pilot (50 GSM8K items and 50 MATH-500 items; arms L, M, I, G-article,
G-best, F-every, ORACLE), stop before the full runs and record the pilot as the result if either
holds on both datasets: ORACLE headroom over the best member is under 3 points; or G-best scores at
or below the best member *and* fires on more than 50 % of tokens (a gate that is not selective has
nothing left to test cheaply).

## Statistics and power

As in September **[believed]**: with 1,319 GSM8K items and about 15 % discordant pairs the standard
error of a paired difference is about 1.1 points and the detectable effect at 80 % power about 3
points; ARC-Challenge similar; MATH-500 about 5 points, hence never alone. Seeds: the T = 0.7 runs
report mean and spread over three seeds. Effect sizes are point differences with intervals, never a
bare "significant".

## Stages, in dependency order

- **S0** `FusedInferenceBackend` with the entropy gate, PoE and mixture rules, per-step metrics and
  the catch-up path; `FusionMath` moved from the bench into the runtime so both use one
  implementation; the bench's `gated:` arm; unit tests on toy backends for G1, G1b and G8 (no
  weights). *Exit when `./gradlew spotlessCheck build` passes and the toy equivalences hold.*
- **S1** Pin the files (download the official GGUFs; convert Math from the pinned safetensors with
  the pinned llama.cpp release); run G0, G1, G1b, G2, G8 on real weights on the pilot host, kernel
  route, route printed. *Exit when every gate passes.*
- **S2** Pilot: 50 GSM8K + 50 MATH-500, the stop-rule arms. Measure tokens/s, core-seconds, trigger
  rate, catch-up lengths, peak RSS; replace every believed cost below. *Exit via the stop rule, or
  continue.*
- **S3** Tune weights and τ, choose the best member, freeze the record (Amendment 2 fixes the grid
  from measured cost, written before S3 starts).
- **S4** ARC-Challenge, all arms. **S5** GSM8K, all arms. **S6** MATH-500, all arms.
- **S7** Apply the decision rules as written; publish the report with every gate log and
  configuration.
- **S8** ModelJars: the `fusion` composition kind, recipe loader and assembly (engineering, may start
  after S0 lands in a Models release); a catalog entry for the article's recipe only on the
  practical claim.

## Cost envelope [believed], replaced by S2 measurements

On a Hetzner ccx33 (8 dedicated cores, the account's dedicated-core limit; a larger box needs a
limit increase), a 1.5B Q4_K_M member decodes in the low tens of tokens per second on the kernel
route. The pilot is 100 items × 7 arms × about 400 output tokens, with the gated arms adding
catch-up prefill proportional to the trigger rate. Expected under 24 box-hours, under EUR 7; the
September study's measured core-seconds are not comparable (different family, plain loader).
Approved spend before S1 is requested with the S2 projection from a one-item timing of each arm.

## Threats to validity, and what the study cannot test

- One family at one size. A 1.5B specialist trio is not the article's 7B trio; the result transfers
  to neither 7B nor other families without being re-run.
- τ and weights tuned on a development split of the same datasets; the T = 0.7 seeds check
  robustness, not generality.
- The Math member is a conversion we perform, not a publisher's file; G0, the parity of its
  single-model arm against llama.cpp on 20 items (recorded at S1), and the pinned toolchain bound
  that risk.
- No code-execution benchmark; see Datasets.
- CPU cost structure. The article's economics are a GPU's; ours are a CPU's where the catch-up is a
  batched prefill. H3 is measured on our hardware only.
- Qwen2.5-Instruct's chat template is used for all members; a specialist tuned under another
  template is handicapped equally in every arm that contains it.

## Follow-on variants (separately pre-registered, only after S7)

- SAFE's consensus selector on top of the entropy gate, if a cheap verify path ever exists.
- A code-execution dataset with a sandboxed executor.
- Contrastive fusion with the Instruct member as the "amateur".
- Cross-vocabulary fusion (UniTE's top-k union), only if same-vocabulary gating shows a gain worth
  generalising.
- A KV-sharing hybrid form (one base, activated adapters per specialty), the only form that meets the
  2026-09-12 hybrid boundary.

## Deliverables

- `FusedInferenceBackend` and its gate, rules and metrics in `models-runtime`, tested on toy
  backends and on real weights through the pipeline.
- The `gated:` arm in `models-bench logit-fusion`, with the route and gate statistics in every
  report.
- Gate logs G0–G5 and G8, the pilot's cost measurements, the frozen tuning record, and a report that
  applies the decision rules as written with its "what this cannot test" section.
- In ModelJars: a `fusion` composition kind with a recipe (members, roles, weights, gate, rule,
  tokenizer identity) assembled at open time; a catalog entry for the article's recipe only on the
  practical claim.

Sources: the article (Level Up Coding, 2026-10-19, "I Fused 3 Tiny Local Models to Beat Single-Model
Limits (Without Weight Merging)"); SAFE, arXiv 2510.15346 (ICLR 2026); UniTE, arXiv 2410.03777 (ICLR
2025); agreement-based ensembling, arXiv 2502.21265; this repository's
`benchmark-results/2026-09-17-logit-fusion-study/` and `benchmark-results/2026-10-21-speculative-ceiling/`.

## Amendment 1 (2026-11-25, before any gate, pilot or test-set run; no threshold changes)

**Catalog category.** The owner decided that "hybrid" is the generic catalog category and that the
hybrids published so far are one *recipe* (an activated LoRA over a shared physical KV prefix).
Other recipes join the category without sharing architecture or KV, and recipes are named. Gated
logit fusion is therefore a **hybrid by recipe**; the sentence "Composition, not hybrid" under *The
technique under test* describes the mechanism (members share no KV), not the catalog kind. Stage S8
reads: in ModelJars, a named recipe (`gated-logit-fusion`) under the hybrid category, with members,
roles, weights, gate and rule assembled at open time; a catalog entry for the article's recipe only
on the practical claim. Nothing else in this protocol changes.
