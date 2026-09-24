# Where a typed decision's time actually goes

Measured 2026-09-24. Everything below was run here, on one box, under one protocol.
Nothing in this file is read from a paper or inferred from a vendor's claim.

## The box

Hetzner CCX33, 8 vCPU on AMD EPYC Milan (4 physical cores, SMT), 32 GiB, Ubuntu,
Temurin JDK 25.0.3. Idle apart from the run.

## What was under test

Harriet, `org.modeljars.composite:harriet`, a typed decision recipe over a frozen
Qwen3.5-4B at Q4_K_M, 2.55 GiB of weights, via `RustFfmBackend` at native kernel
ABI 6. Evidence is a 143-token contract (`document.txt`); the fifteen decision
cases are `cases.tsv`.

Harnesses are in `harness/`. Each is a single class run as

    java --enable-native-access=ALL-UNNAMED --add-modules jdk.incubator.vector \
      -Dmodels.native.gatedDeltaNet=true -cp "classes:native:lib/*" demo.<Name> <input>

## 1. A decision is compute bound, not bandwidth bound (`Threads`, `Split`)

This is the finding everything else follows from, and it is the opposite of what
the grouped-decision code was written to exploit.

Prefill time against token count, one batch, evidence prefilled separately:

| tokens | seconds | ms/token |
|--------|---------|----------|
| 17     | 0.322   | 18.9     |
| 30     | 0.556   | 18.5     |
| 107    | 2.033   | 19.0     |
| 143    | 2.684   | 18.8     |

Dead linear, no fixed cost. If reading the weights dominated there would be a
large intercept and a near-flat slope.

Prefill of the same 143 tokens against thread count:

| threads | seconds | ms/token |
|---------|---------|----------|
| 1       | 10.056  | 70.3     |
| 2       | 5.157   | 36.1     |
| 4       | 2.577   | 18.0     |
| 8       | 2.981   | 20.9     |

It halves, halves again, then saturates on the box's four physical cores. That is
arithmetic at its ceiling. **N questions cost N questions' arithmetic however they
are arranged.**

The single-token step that ends a decision behaves the other way: 0.172 s, 0.091 s,
0.064 s, 0.066 s over the same thread counts. Flat past two threads -- bandwidth.

So one decision, warm, over shared evidence:

| part                              | seconds | share |
|-----------------------------------|---------|-------|
| prefill the question's own tokens | 0.322   | 76%   |
| final single-token step           | 0.065   | 16%   |
| everything else                   | 0.035   | 8%    |
| **total**                         | 0.422   |       |

## 2. Grouping questions is worth what that final step costs (`Batch2`)

Twenty questions over one piece of evidence, both arms measured, not extrapolated.

With the native quantized decode kernel (the shipped default):

| n  | one at a time | grouped | speedup |
|----|---------------|---------|---------|
| 2  | 0.916         | 1.311   | 0.70x   |
| 5  | 2.540         | 2.899   | 0.88x   |
| 10 | 4.796         | 4.675   | 1.03x   |
| 20 | 8.951         | 8.626   | 1.04x   |
| 30 | 13.600        | 13.011  | 1.05x   |

Run-to-run band is about +-5%, calibrated from a group size past the backend's
maximum where both arms take the identical path and measured 0.98x. So grouping is
a wash at every size.

With that kernel off, same code, same box:

| n  | one at a time | grouped | speedup |
|----|---------------|---------|---------|
| 2  | 1.687         | 1.350   | 1.25x   |
| 5  | 4.366         | 2.943   | 1.48x   |
| 10 | 8.593         | 5.416   | 1.59x   |
| 20 | 16.839        | 9.935   | 1.69x   |

The only thing that changed is the price of the step grouping removes. This is why
`GroupedDecisionBackend.groupedDecisionBreakEven()` is asked of the backend rather
than fixed by the caller.

### Two defects found on the way

- The grouped path called `reset()` and re-read the whole evidence on every grouped
  call, then discarded the resumption point so the next one-at-a-time decision paid
  for it again. Worth 2.68 s per call here. Twenty questions: 10.79 s -> 8.22 s.
- The Gated DeltaNet kernel stays on the calling thread for one token of one
  sequence, which is right for decode; the grouped path called it once per branch,
  so every branch ran on one core. Handing it the whole group: 12.39 s -> 10.79 s.

## 3. Batched and single-row arithmetic disagree, by up to 0.04 (`Chunk`)

A question prefilled as one batch does not give the same answer as the same question
fed a token at a time. Six criteria over the same contract, probability of the first
option:

| criterion                                      | batched | per token | shift |
|------------------------------------------------|---------|-----------|-------|
| Is a monthly fee stated?                       | 0.9407  | 0.9522    | 0.012 |
| Is an uptime guarantee stated?                 | 0.9041  | 0.8922    | 0.012 |
| May Customer Data be used to train models?     | 0.1134  | 0.1173    | 0.004 |
| Is a liability cap stated?                     | 0.8693  | 0.8690    | 0.000 |
| Does the liability cap apply to data breaches? | 0.4600  | 0.4884    | 0.028 |
| Can the agreement be terminated for convenience?| 0.9156 | 0.9196    | 0.004 |

Max logit gap 0.5-0.8 across 248,320 logits. It is **not** the chunked associative
scan: repeating with the pure Java per-token recurrence gives the same spread
(worst 0.044). It is batch-size-dependent matrix arithmetic, and it predates all of
this work. A caller needing bit-identical answers must pick one path and stay on it.

## 4. The lever that is actually large: the options block (`Reorder`)

A decision's cost is the arithmetic of the tokens after the shared evidence. Today
those are the criterion **and** the rendered options, and the options are most of
them -- and identical across every question with the same answer space. Moving them
ahead of the criterion makes them part of the shared prefix.

Fifteen cases, per-question part only, shared prefix prefilled outside the clock in
both arrangements:

| | now | options first |
|---|---|---|
| suffix tokens, total | 463 | 240 |
| mean seconds | 0.623 | 0.356 |

**1.75x on every decision, in every configuration, with no kernel work.**

**It is not free: winners agree on only 11 of 15 cases.** Four flip. That is a
different prompt and therefore a different model, and whether the flips are better
or worse is not knowable from this harness -- it has no gold labels.

## 5. So it was scored against gold labels, and rejected (`options-first/`)

Run as a flagged arm of the existing `JevBenchRunner` (`-Ddecisions.optionsFirst=true`)
over JevBench v1.2's own public items, easy (48) and original (72), scored by importing
the benchmark's own `scoring`, `metrics` and `composite_v12` modules rather than
reimplementing them. Temperature 1.0 in both arms -- nothing fitted.

| | shipped order | options first |
|---|---|---|
| easy | 48/48 = 1.0000 | 42/48 = 0.8750 |
| judge | 59/72 = 0.8194 | 55/72 = 0.7639 |
| **overall** | **107/120 = 0.8917** | **97/120 = 0.8083** |
| Intelligence | 88.0 | 80.1 |
| Calibration | 71.9 (ECE 0.1405) | 80.6 (ECE 0.0968) |
| Speed | 64.8 (p50 2.411 s) | 64.8 (p50 2.415 s) |

Winners differ on 19 of 120. Per family, where it moved:

| family | shipped | options first | n |
|---|---|---|---|
| fact | 1.0000 | 0.5000 | 12 |
| adequacy | 0.7500 | 0.5000 | 12 |
| extraction | 1.0000 | 0.9167 | 24 |
| intent | 0.8750 | 0.8333 | 24 |
| routing | 0.8333 | 0.9167 | 12 |
| ordinal | 0.9167 | 1.0000 | 12 |

**Rejected.** Two families halve. The composite barely moves -- the Intelligence loss
is offset by a Calibration gain that is mostly the arm being less confident, which is
not an improvement anyone asked for -- and the accuracy loss is concentrated rather
than spread, which is the signature of a prompt the model reads differently and not of
noise.

**Note what the Speed column does here: nothing.** 64.8 in both arms, p50 within 4 ms.
Every JevBench item carries its own state, so there is no shared prefix for the options
to move into and the saving cannot appear. The 1.75x of section 4 is real and it is
real *only* in the regime `decideAll` is for -- many questions, one document. So this
arm was measured on a cohort that can price the accuracy cost and cannot price the
speed benefit, which is stated here rather than resolved, because the benchmark has no
shared-evidence tier to resolve it with.

**What was not run:** the hard tier (111 items, ~3,700-token states, about 70 s an item
and 2.2 hours an arm). It was cut once the fast tiers returned a loss this large and
this concentrated; running it would have refined a number that was already deciding
against the change. It also would not have helped the speed question: a 3,700-token
state makes a 50-token options block irrelevant, so the hard tier is where the reorder
matters least.

## Scoreboard for the week's work

| change | measured | verdict |
|---|---|---|
| grouped recurrence in one kernel launch | 12.39 s -> 10.79 s at n=20 | kept |
| group resumes its evidence | 10.79 s -> 8.22 s at n=20 | kept |
| grouping at all, fast decode kernel | 1.03x-1.05x, band +-5% | off by default |
| grouping at all, no fast decode kernel | 1.25x at n=2, 1.69x at n=20 | on |
| options ahead of the criterion | 1.75x per decision, -8.3 points accuracy | rejected |

## What this says about the gap to a hosted System One service

A hosted service answering in ~5-10 ms of compute is not doing a better job of
batching. It is doing far less arithmetic. On this box our arithmetic is already at
the machine's ceiling, so the levers are fewer tokens per question (finding 4), a
smaller model, or more FLOPs -- not scheduling.
