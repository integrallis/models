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

This is an instruction-throughput ceiling, not a matmul prediction: the real kernel also pays nibble
unpacking, per-group scales and memory, and Amdahl will take most of it. It is still the largest
unexploited factor found, it lands on the ~79% of a decision that is matmul, and the kernel already
does runtime capability dispatch so it is an added path rather than a rewrite.

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

Implement an AVX-512 + VNNI `Q4_K` path behind the existing capability dispatch, and judge it on
`ColdShape` and `Achieved` rather than an isolated matmul -- per `../q4k-row-tile/NOTES.md`, a
resident-tensor microbenchmark already produced a +9.1% result that was +0.4% on a real forward pass.

Latency arithmetic, to be checked and not assumed: 0.506 s at 32 Zen 5 threads today, on a workload
that is ~79% matmul. Whether that reaches the 300 ms the live workloads want depends entirely on how
much of the 2.65x-5.17x survives decode and memory, which is exactly what the measurement will say.
