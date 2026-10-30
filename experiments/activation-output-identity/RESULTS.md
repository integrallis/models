# Activation determinism on x86: what each candidate costs and what it changes

Measured 2026-10-26/27. Every number below is from a run on an isolated Hetzner ccx33 (8 dedicated
vCPU, AMD EPYC-Milan, 32 GB) in `ash`, nothing else on the host. Raw artifacts are under
`articles/models-deep-dive/evidence/`. Re-derive with the analyzers in
`articles/models-deep-dive/tools/`.

Read `SCOPE.md` first: it establishes where the defect lives, which is not where it looks.

## The defect

`lanewise(VectorOperators.EXP)` changes its rounding once HotSpot intrinsifies the call, so a
model's first forward pass produces different floats from every later one. It is a property of the
JDK vector intrinsic, not of any model or quantization.

Determinism therefore requires building exp out of operations the JVM specifies exactly — add, mul,
fma, integer shift. No warmup trick substitutes for that: C2 compiles asynchronously and can
deoptimize mid-run, so "warm it first" is probabilistic, not a guarantee.

`PublishedArmActivations` already does this: zero `EXP` or `LOG` intrinsics, 60 fused multiply-adds
plus integer exponent shifts. **That, not the architecture, is why aarch64 at a 128-bit float
species is deterministic.** Nothing in the arithmetic is ARM-specific; only the
`USE_PUBLISHED_ARM_ACTIVATIONS` predicate and a hardcoded `SPECIES_128` tie it there.

## Throughput cost of each candidate

Protocol fixed before running: arms alternate A B A B so thermal and temporal drift hits both
equally, fresh JVM per run, warmups inside each run, median per arm, 2-percentage-point precision
target. Route verified identical across every run of every arm:
`vector-provider=PanamaVectorUtilSupport`, `active-vector-bits=256`, `compiler=hotspot-c2`,
`performance-cliffs=fused-grouped-attention-disabled` — the shipped default route for this model.

| candidate | what it does | vs baseline **mean** | vs baseline **slow mode** | pairs |
| --- | --- | --- | --- | --- |
| `b3c1245d` | `ReproducibleActivationExp`: unfused double-vector reproduction of `StrictMath.exp` | **0.9109** | 0.9965 | 9 |
| `44211d46` | the published polynomial, at its hardcoded `SPECIES_128` | **0.8802** | — | 8 |
| `b8ad9fcf` | the same polynomial at the machine's preferred width | **0.9660** | **1.0039** | 8 |

Higher is better; a ratio below 1.00 means the candidate is slower.

## The finding that reframes the trade: the baseline is bimodal

A median over the pooled baseline hides this, and reporting one was the first mistake in this
investigation. All three trials inside any single JVM agree to within half a percent, so the level
is a compilation outcome chosen during warmup and then held for that run — not thermal drift, which
would vary within a run.

In the 9-pair run against `b3c1245d`:

| arm | modes | detail |
| --- | --- | --- |
| baseline `v0.1.30` | **2** | slow: 6/9 runs, median 1.0095 tok/s · fast: 3/9 runs, median 1.3118 tok/s |
| `b3c1245d` | **1** | 9/9 runs, median 1.0059 tok/s |

Example trial triples: baseline fast `1.305, 1.312, 1.312`; baseline slow `1.006, 1.010, 1.018`;
candidate on the same pair `0.982, 0.985, 0.975`.

**Every deterministic candidate matches or beats the baseline's slow mode and loses to its fast
mode.** The widest-species polynomial is 4.5% *faster* than the slow mode. That consistency across
three unrelated implementations is the evidence that the cost is not implementation inefficiency.

The hypothesis it supports — that the fast mode is the intrinsified `lanewise(EXP)`, i.e. the
non-determinism itself — is **not yet measured**. If it holds, no polynomial can reach that path,
because what makes it fast is what makes it irreproducible, and "determinism at zero throughput
cost on x86" is not available.

## The leading candidate: the polynomial at native vector width

`b8ad9fcf` changes **no arithmetic**. It replaces the hardcoded `SPECIES_128` with a static final
species at the machine's preferred width, which is legal under
`VectorSpeciesConstantStructureTest` — that rule forbids species parameters and mutable species
fields, not wider species. Every operation is lane-wise and the integer exponent work goes through
`reinterpretAsInts` and a same-size `F2I` convert, both shape preserving.

Final, 8 pairs:

| comparison | ratio |
| --- | --- |
| median decode | **1.0026** |
| vs the baseline's slow mode, 6 of 8 runs | **1.0039** (candidate faster) |
| vs the baseline's fast mode, 2 of 8 runs | 0.7743 |
| mean-to-mean | **0.9660** |

Per-pair decode ratios: 1.0936, 0.7744, 1.0067, 0.9952, 0.7650, 1.0990, 1.1169, 0.9937. The
candidate is faster on three pairs and at parity on three. **The entire 3.4% mean deficit is the
two pairs where the baseline reached its fast mode.**

Against what a user gets in three runs out of four, this variant is marginally faster than the
released build while being deterministic. That is the closest to "no regression" any candidate has
come, and it supports the reading that the remaining deficit is not inefficiency but the price of
the irreproducible path.

**Unverified at the width that matters.** This workstation is aarch64 at 128 bits, so
`SPECIES_PREFERRED` resolves to 128 locally and the test suite never exercised the 256-bit path:
the local pass holds the axis the change varies constant. The throughput run proves it executes and
produces plausible output on real weights, not that it produces the correct bits. The queued
control includes it as a cohort arm, so a 0-of-120 result there is direct evidence of output
equivalence at 256 bits.

## Output impact, over the real 120-case cohort

One fixed template, the 120 JevBench cases, exact token comparison, greedy decode, 8 tokens per
case. The harness is discriminating rather than degenerate: **111 distinct sequences over 120
cases**, 42 distinct first tokens, no sequence repeating more than three times.

A decision model's output is its label; the trailing prose is not. Both are reported because the
difference between them is the whole result.

| arm | full 8-token sequence differs | **decision label differs** |
| --- | --- | --- |
| `31540dd8` F32 reduction reorder (shipped in 0.1.31) | **0 of 120** | **0 of 120** |
| `b3c1245d` activation substitution | 24 of 120 (20.00 pp) | **2 of 120 (1.67 pp)** |

The two label flips are both ordinal cases and they flip in **opposite directions**:

| case | expected | baseline | `b3c1245d` |
| --- | --- | --- | --- |
| `original-ordinal-03-1` | family | `0` | `3` |
| `original-ordinal-04-1` | family | `3` | `0` |

Net quality change across the cohort: **zero**.

This matters for a decision already recorded in the tree. `ActivationColdHotTest` and the #306
commit message both justify rejecting the activation substitution by citing "JEVBench 76/120
released against 75/120 candidate". That figure rests on run 38596566492, which returns 404, with a
harness whose sha256 (`0a325a6d…`) is in neither the tree nor the archive. **It does not reproduce
here**: 1.67 pp of label divergence with zero net direction, against a protocol whose stated
precision target is two percentage points. The correct reason to reject `b3c1245d` is its
throughput cost, not a quality loss.

## Known gap: there is no baseline self-difference control

Every comparison above measures one baseline JVM against one candidate JVM. The released x86 build
is the **non-deterministic** arm, and the throughput data shows it selecting one of two compilation
modes per JVM. So some part of "24 of 120 differ" may be the baseline failing to reproduce itself
rather than the arm changing anything, and nothing here separates the two.

The control is three arms with two of them the **same commit** (`baseline`, `control` both
`v0.1.31`, plus the wide polynomial). It is queued. Until it reports, every divergence count in
this document is an upper bound on the arm effect, not an attribution.

## Candidate retained but not viable: sigmoid specialisation

`35732bb2` drops the general exp's four edge-case blends and its two-step exponent scale, replacing
them with a clamp of the exponent argument into `[-87, 88]`. `swiGlu` divides by `1 + exp(-value)`,
so an exp that overflows gives `value/inf = 0` and one that underflows gives `value/1 = value` —
already the correct limits of `x*sigmoid(x)`. Roughly nine operations of the twenty-four in the
vector loop become redundant.

Validated locally: worst relative error **1e-5** against `x/(1+StrictMath.exp(-x))` in double
across realistic, saturating and tiny inputs, and **bit-identical across 200,000 iterations** of
recompilation.

Not viable as it stands. `PublishedArmActivationsTest` asserts bit-exact equality against
`ArmActivationReference`, the pinned scalar recipe, and single-step exponent scaling rounds once
where two-step rounds twice; eleven cases fail at lengths where the vector loop engages. The vector
and scalar paths agree with each other — both differ from the published recipe. That is a **parity**
problem rather than a test problem: aarch64 keeps the published recipe, so specialising only x86
would make the two platforms disagree numerically, trading one inconsistency for another.

Making it viable means applying it to **both** platforms and re-pinning `ArmActivationReference` to
the new recipe, which changes aarch64's output and so needs the cohort identity check too.

## What is not established

- **That the fast mode is the intrinsified EXP.** It is the obvious reading of three consistent
  results, and it decides whether zero-cost determinism exists. Unmeasured.
- **That any candidate has no fast mode.** Nine runs of `b3c1245d` produced none; if it reached the
  baseline's one-in-three rate, observing zero has probability (2/3)^9 ≈ 2.6%. Unlikely, not excluded.
- **Anything about the other two activation sites or other platforms.** All of the above is x86-64
  AVX2 at 256 bits, `swiGlu` only, one model. `sigmoidAndScaledSoftplus` additionally uses
  `lanewise(LOG)`, which none of these candidates addresses.
- **That 0.1.31's reduction change is output-identical anywhere but here.** It is, on this model and
  this cohort, with 0 of 120 differing. That is one model on one platform.
