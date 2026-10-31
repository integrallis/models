# Laya through the Q8_0 × F32-activation kernel — 2026-10-31

#292's acceptance criterion, run. **It is met, and the headline is stronger than the criterion
asked for: the kernel is byte-identical to the shipped Java path on every probability of every case,
across five independent runs.**

## Provenance

| | |
|---|---|
| artifact | `Laya-Q8_0.gguf`, 449,397,600 bytes, on this disk |
| cohort | `../2026-09-24-decision-latency/clm-arm/cohort-jevbench-120.jsonl`, 120 items, sha256 `4bcce87d40fd8edefd8984297d76390794c9fe2613c77fb9bb334006f5969b02` — **verified before the run**, and the same digest Laya's published evidence names |
| scoring | `../2026-09-24-decision-latency/options-first/score_arm.py`, which imports JevBench's own `composite_v12`, `metrics` and `scoring`; plus `score_per_type.py` here for the per-type cells that scorer does not break out |
| harness | `../2026-10-23-laya-vs-harriet/LayaLatency.java`, unmodified |
| candidate | `RustGgufBatchedMatrixKernel.openBundledExactActivations` → `Q8_0_F32_ACTIVATION_BATCHED_MATMUL` |
| reference | `GgufBatchedMatrixKernel.none()` → `TensorOps.ggufExactBatchedMatmul`, the shipped default |
| host | i7-9750H, AVX2, **no AVX-512**; macOS, 12 processors. **Not** the i9-14900HX of the 2026-10-23 record |

## Result: the acceptance criterion

#292 asks for "accuracy 0.783333 and the published per-type cells on this cohort through the
kernel".

| | published | through the kernel |
|---|---|---|
| overall accuracy | 0.783333 | **0.7833** (94/120) |
| `choice` | 0.791667 | **0.791667** (57/72) |
| `noul` | 0.777778 | **0.777778** (28/36) |
| `score` | 0.750000 | **0.750000** (9/12) |
| ECE | 0.163770 | **0.1638** |

`winners differ on 0/120 items`, against **4 of 120** for the quantizing `Q8_0_F32_BATCHED_MATMUL`
route that #292 rejected.

### Stronger than that: byte-identical, five times over

Every probability column of every case is **identical** across all five runs — three independent
kernel runs and two Java runs:

```
all FIVE runs (3 kernel, 2 Java) byte-identical on every probability column: True
```

Not "within tolerance". So **this route cannot change an answer, because it does not change a bit**,
and it is reproducible run to run rather than identical by luck on one pass. That retires the
accuracy objection in #292 entirely — the objection was never about this route, it was about the
quantizing one.

A matching bit-parity assertion is now committed as a test rather than left as a measurement:
`NativeKernelLibraryTest.theExactActivationQ8_0RouteMatchesTheShippedJavaPathBitForBit`, fifteen
shapes against `TensorOps.ggufExactBatchedMatmul` with `containsExactly`, plus a negative control
asserting the fixtures vary on the axis fold order depends on.

## Latency: the paired ratio reproduces, the absolutes do not

Two passes, each running arm A then arm B back to back on an otherwise idle host
(`run-ab.sh`, `run-ab-replication.sh`):

| pass | kernel p50 | Java p50 | **ratio** | kernel p95 | Java p95 |
|---|---:|---:|---:|---:|---:|
| 1 | 2065.54 ms | 5989.88 ms | **2.90×** | 3113.03 | 9987.05 |
| 2 | 2837.71 ms | 7515.08 ms | **2.65×** | 5259.27 | 12099.86 |

**The absolute numbers on this host are not usable and must not be quoted.** Within-arm spread
between passes is +37% for the kernel and +25% for Java, and pass 2 was *slower on both arms under a
lower load average* (13.5 before pass 2 against 25.2 before pass 1) — arm B2 took over 16 minutes
for work B1 did in about 8. That is a laptop thermal-throttling after ~45 minutes of sustained load,
not a property of either code path. An earlier attempt was worse still: run A0 here
(`arm-a0-kernel-contended.tsv`) ran concurrently with a download and another job at a load average
over 300, and its timing is discarded for that reason while its probabilities are kept, because
those are contention-immune.

**The ratio is what survives, and it survives because it is paired.** Both arms drifted in the same
direction between passes, so the ratio moved from 2.90× to 2.65× while the absolutes moved by a
third. Two passes agreeing at ~2.6–2.9× is the reportable finding: **the kernel is roughly 2.7×
faster per decision at the published accuracy, bit for bit.**

That answers #292's "why it is worth doing" — latency was Laya's weakest axis against Harriet — but
a product figure needs a dedicated host. `../2026-09-24-decision-latency/REPRODUCE.md` requires a
CCX33 for exactly this reason.

### The earlier 6.4× objection is gone, and it was about the scalar kernel

The 2026-10-23 addendum measured this route at **27.0 s** per decision against Java's 4.2 s and
called it "6.4x too slow", with a full 120-case run abandoned at thirty minutes. That was the
**scalar** `dot_q8_0_f32_activation_row_scalar`. `Q8F32ActivationAvx2` now exists and
`selected_q8f32_kernel()` returns it on any avx2+fma host, so the arm measured here is the vector
one. The scalar kernel remains as its bit-parity oracle, which is the role Q5_1's scalar entry point
serves.

## What this does not establish

- **No product latency.** See above: the absolutes on this host are discarded and only the paired
  ratio is reported.
- **Not a re-qualification.** #292 says "then re-qualify". This is the acceptance measurement, not a
  two-arm qualification record — there is no `comparator.json` and no `verdict.json` here, and the
  catalog still publishes `0.783333` for the exact path. Switching the default is a separate change
  that needs that record.
- **One cohort, one model.** 120 items on one typed-decision model. Nothing here says anything about
  generative Q8_0 models, which keep the quantizing route untouched by design — the new capability
  is a separate entry point precisely so that flipping `activation_format` globally was not needed.
- **One host class.** AVX2, no AVX-512, x86-64 macOS. The scalar fallback is what runs where
  avx2+fma is absent, and it is not measured here.
