# Going after latency: what is actually left

Measured 2026-09-25. A decision is **98.5% prefill** (2026-09-24: tokenise 1.2 ms, resume 4.6 ms,
score 0.2 ms, prefill 385 ms), and prefill is linear in tokens with no batching benefit, so latency
is proportional to prompt length times per-token cost. This file closes three candidate levers and
opens one.

Two hosts, both dedicated-ish and both stated:

* **Milan** -- Hetzner CCX33, 8 vCPU on 4 physical Zen 3 cores at 2400 MHz. Dedicated.
* **Zen 5** -- RunPod `cpu3c`, 32 vCPU on an AMD EPYC 9655P (96 core). Shared host, so every run
  below sampled `/proc/stat`'s steal field around it; **steal delta was 0 on all 18 runs**. Not the
  same guarantee as a dedicated box, and said so rather than implied.

## 1. Prefill batch width: no latency effect, but it changes the answer

60-token prompt on Milan, mean of 3 rounds, one JVM per width:

| width | 8 | 16 | 32 (default) | 60 | 64 | 128 |
|---|---|---|---|---|---|---|
| seconds | 1.481 | 1.395 | 1.540 | 1.579 | 1.629 | 1.528 |

Flat to slightly worse as it widens. The arithmetic-intensity argument for wider passes -- Q4_K
spends 144 bytes per 256 weights, so a wider batch feeds more multiply-accumulates per weight byte
-- **did not hold**. Recorded as a wrong prediction.

**The answer is not invariant to chunking.** Checksums of the final-position logits:

| width | checksum | max abs logit |
|---|---|---|
| 8 | -225896779.808229 | 8.445599 |
| 16 | -223534962.007473 | 8.518201 |
| 32 (shipped default) | -228452872.406674 | 8.484456 |
| 60 / 64 / 128 | -224957227.911217 | 8.506352 |

The last three agree because a 60-token prompt fits one pass at all of them, so there is a single
"unchunked" answer and the shipped default of 32 is not it. `max|logit|` moves by **0.073**, the same
order as the grouped-vs-single defect this project already fixed (up to 0.28). Whether that moves
Intelligence is **not measured**; the noise floor here is about one item in 120, so it is worth an
arm of its own. Flagged, not concluded.

## 2. Final-layer prefill pruning: already on

`models.purejava.finalLayerPrefillPruning` and `models.purejava.finalLayerKvOnlyPrefill` both default
to **true** (`booleanProperty` returns true for an unset value). Nothing to win; the lever was taken
before this session looked at it. Checked rather than assumed.

## 3. Cores: keeps paying well past four

Same 60-token prefill, one host each, mean of 3 rounds.

| threads | 1 | 2 | 4 | 6 | 8 | 16 | 32 |
|---|---|---|---|---|---|---|---|
| Milan (4 physical) | 5.082 | 2.757 | **1.471** | 1.923 | 1.554 | -- | -- |
| Zen 5 (32 vCPU) | 3.869 | 1.951 | 1.438 | -- | 1.169 | 0.670 | **0.506** |

On Milan, 86% scaling efficiency to 4 threads and then nothing: SMT buys zero and 6 threads is
actively worse than 4. That is the machine's limit, not the code's.

On Zen 5 it keeps going: 32 threads is **7.6x** one thread, and 2.8x the 4-thread figure. Best single
run **0.457 s** for 60 tokens, 7.62 ms/token, against Milan's best of 1.385 s.

Per core the two parts are nearly identical at 4 threads (1.438 vs 1.471), so **this is core count,
not instructions per cycle**. At one thread Zen 5 leads 3.869 to 5.082, about 1.31x.

**Hetzner cannot go further on this account**: CCX43 and CCX53 are both refused with
`dedicated core limit exceeded`, so 8 vCPU / 4 physical cores is the ceiling there. That is a quota,
not a technical limit, and it is why the wide arm ran on RunPod.

## 4. The open lever: VNNI, which is not blocked after all

Earlier notes record VNNI as unavailable because Hetzner's dedicated line is Zen 3 and has no
AVX-512 at all. That is true of Hetzner and **false in general**: the RunPod CPU pods are EPYC 9655P,
which reports `avx512_vnni`, `avx_vnni` and `avx512_bf16`. RunPod bills those at $0.03/vCPU/hr.

`harness/vnni.c`, one core, register-resident, best of 3, `objdump` confirming 16 `vpdpbusd` and 8
`vpmaddubsw` actually emitted:

| int8 multiply-accumulate | G-MAC/s | vs ours |
|---|---|---|
| AVX2 `maddubs` + `madd` -- what the kernel does now | 93.9 | 1.00x |
| AVX2 `vpdpbusd` | 249.4 | **2.65x** |
| AVX-512 `vpdpbusd` | 485.7 | **5.17x** |

This is an instruction-throughput ceiling and **most of it does not transfer**. Measured on the real
loop shape next.

### 4a. What actually transfers: 2.26x, and why the 2.65x did not

Reading the kernel first: per 32 weights our AVX2 path is `maddubs` + `madd` + `add` = **three** ops,
and the `madd` applies the per-group scale *and* pair-adds in one instruction. A 256-bit `vpdpbusd`
rewrite is also three (`dpbusd`, `mullo`, `add`). **At 256 bits VNNI wins nothing here**, and the
scale cannot be hoisted earlier because a 4-bit quant times a 6-bit scale leaves `u8`. The 2.65x came
from a loop with no scale multiply.

What is left is 512-bit **width**: one `dpbusd` covers 64 weights instead of 32, and the nibble unpack
halves too. `latency/q4k_shape.c` and `latency/q4k_batched.c` measure exactly that, on a 9216x2560
Q4_K tensor, 12.7 MiB of weights streaming, one core:

| arm | batch 1 | batch 18 |
|---|---|---|
| AVX2, 8 groups of 32 | 31.0 G-MAC/s | 67.3 G-MAC/s |
| AVX-512 + VNNI, 4 groups of 64 | 41.2 G-MAC/s | **152.4 G-MAC/s** |
| ratio | 1.33x | **2.26x** |

The batch dimension is the whole story. At batch 1 the nibble unpack is paid once per dot and
dominates, so halving the dot buys 1.33x. At batch 18 the unpack is hoisted above the batch loop --
which is what the real kernel does -- so the dot dominates and the width shows up as **2.26x**.

**Batch 18 is our case**, so 2.26x is the figure to plan against: it is on the real loop shape, at the
real width, with weights streaming rather than resident. It lands on the ~79% of a decision that is
matmul, and the kernel already dispatches on runtime capability, so this is an added path and not a
rewrite.

Still to be proven on `ColdShape` and `Achieved` before it counts, per the row-tiling lesson.

Two further reasons it is attractive beyond speed:

* `vpdpbusd` accumulates straight to int32. The current path goes through `maddubs`'s int16
  intermediate, where a sum of two `u8 * s8` products can reach 64,770 and **overflow int16** -- the
  hazard this codebase already carries `q4ShortPairwiseSupported` and widened kernels for.
* Integer multiply-accumulate is associative, so a VNNI path should be **bit-identical** to a
  correct widened AVX2 path rather than merely close.

### How this benchmark was wrong first

The first version reported `vpdpbusd` as **0.64x, slower than doing strictly more work**, and an
effective clock of 411,395 GHz. Both had the same cause: the compiler. The clock came from a
dependent-add chain GCC folded into one add. The ratio came from `a` and `b` being loop-invariant, so
the AVX2 arm's `maddubs`+`madd` hoisted clean out of the loop and the arm timed a bare vector add.

Fixed by pinning both operands behind an empty `asm` barrier each iteration and reporting absolute
throughput so no clock estimate is needed. The absurd clock figure is what exposed it -- a plausible
wrong number would have been published.

## What to do next

Two levers, in order of what they cost us.

**1. Cores. 2.9x, available today, zero code.** 1.471 s on Milan at 4 threads against 0.506 s on
Zen 5 at 32. Nothing to build; it is a deployment choice. Our own benchmarking is what the Hetzner
quota blocks, not a customer.

**2. An AVX-512 + VNNI `Q4_K` path behind the existing capability dispatch. 2.26x on the inner loop**
at our batch width, so something under that end-to-end on the ~79% of a decision that is matmul.
Worth building, and it has two correctness arguments on top of speed:

* `vpdpbusd` accumulates straight to int32, skipping `maddubs`'s int16 intermediate where two
  `u8 * s8` products can reach 64,770 and overflow -- the hazard `q4ShortPairwiseSupported` and the
  widened kernels exist for.
* Integer multiply-accumulate is associative, so the path should be **bit-identical** to a correct
  widened AVX2 path rather than merely close. That is a testable claim and the test is cheap.

Judge it on `ColdShape` and `Achieved`, never on an isolated matmul: per `../q4k-row-tile/NOTES.md` a
resident-tensor microbenchmark already produced +9.1% that was +0.4% on a real forward pass. Both
harnesses exist.

Arithmetic to check and not assume: 0.506 s at 32 Zen 5 threads today. If the 2.26x delivers even
half of itself end-to-end, that is roughly 0.3 s, which is the bar the live workloads
(300 ms trading blocks, 2 Hz drone control) actually want.
