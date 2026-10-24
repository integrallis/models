# Laya vs Harriet, one host and one scorer, 2026-10-23

Both arms re-run here. Nothing in this file is lifted from an earlier record or a leaderboard.

## Why this exists

The two models already had numbers, and they were **not comparable**. Laya's published evidence was
measured on released 0.5.4 on macOS; Harriet's accuracy came from `../2026-09-24-decision-latency/`
on **0.3.46** on a Hetzner CCX33, and its latency from `../../models-decisions/benchmark-results/
2026-09-21-jev-sidebyside/` on RunPod Zen5 boxes. Three libraries, three hosts. Comparing across
them would have produced an ordering of press releases, not a measurement.

## Protocol

| | |
| --- | --- |
| cohort | `../2026-09-24-decision-latency/clm-arm/cohort-jevbench-120.jsonl`, 120 items, sha256 `4bcce87d40fd8edefd8984297d76390794c9fe2613c77fb9bb334006f5969b02` |
| cohort provenance | byte-identical to the file Laya's published evidence names in its own `cohortSha256`, verified by digest, not by filename |
| host | one macOS x86_64 box, 12 processors, 32 GB, both arms back to back |
| library | one local build of `main`; both arms loaded the same `backend-java` classes |
| scoring | `../2026-09-24-decision-latency/options-first/score_arm.py`, which imports JevBench's own `composite_v12`, `metrics` and `scoring` rather than reimplementing them |
| Laya arm | `LayaLatency.java` (retained here), `ModernBertDecisionModel.open` → `decide`, options built exactly as `LayaJevBenchQualificationIT` builds them |
| Harriet arm | `com.integrallis.models.decisions.JevBenchRunner`, `-Ddecisions.letterLogits=true`, context 4096, prefill batch 512, released `backend-native-0.5.4.jar` from Central on the classpath |

Both raw arm outputs and their digests are in this directory.

## Result

| axis | Laya GGUF Q8_0 | Harriet (Qwen3.5-4B Q4_K_M) |
| --- | ---: | ---: |
| overall accuracy | 94/120 — **0.7833** | 106/120 — **0.8833** |
| easy tier | 46/48 — 0.9583 | 48/48 — **1.0000** |
| judge tier | 48/72 — 0.6667 | 58/72 — **0.8056** |
| Intelligence | 76.4 | **87.0** |
| Calibration (ECE) | 67.2 (0.1638) | **75.9 (0.1203)** |
| Speed | **61.8** | 55.9 |
| p50 per decision | **3.318 s** | 6.203 s |
| p95 per decision | **4.800 s** | 10.130 s |
| one-off cost | **0.22 s** model load | 0 here — see below |
| artifact | **428 MiB** | 2.55 GB |

**Harriet is 10 accuracy points better and better calibrated. Laya is 1.9x faster per decision and
6x smaller.** They disagree on 26 of 120 items, so neither dominates item by item.

### Per family

| family | Laya | Harriet | n |
| --- | ---: | ---: | ---: |
| policy | **0.8333** | 0.6667 | 12 |
| fact | 0.9167 | **1.0000** | 12 |
| intent | 0.7917 | **0.8750** | 24 |
| ordinal | 0.7500 | **0.8333** | 12 |
| extraction | 0.8750 | **1.0000** | 24 |
| adequacy | 0.5833 | **0.7500** | 12 |
| routing | 0.4167 | **0.8333** | 12 |

`policy` is the one family Laya wins, and `routing` is where it is weakest by a wide margin.

## Cross-validation

The scorer independently reproduced Laya's **published** figures from this run's raw TSV:
accuracy `0.7833` against the catalog's `0.783333`, ECE `0.1638` against `0.163770`. Two unrelated
paths — the qualification IT and this probe plus JevBench's scorer — agree, so neither the probe
nor the published evidence is a measurement artefact.

Laya's per-decision latency also reproduced across two runs: p50 3264.85 ms and 3317.58 ms.

## What this does not establish

- **Not a dedicated timing host.** This is a shared laptop, not the CCX33 that
  `../2026-09-24-decision-latency/REPRODUCE.md` requires. The *ratio* is same-host and back to
  back, so it is meaningful; the absolute seconds are not comparable to any published figure and
  must not be quoted as a product latency.
- **Harriet ran with no prefix reuse — 0 of 120, by its own report line.** That is not a setup
  failure and not a missing capability: **every JevBench item carries its own state, so there is no
  common prefix to reuse.** `../2026-09-24-decision-latency/REPRODUCE.md` says so directly — the
  per-question latency the release is about is not measurable on this cohort. **This table is
  therefore Harriet's unshared mode, which is its worst case**, and its amortised profile
  (`prefill 5.89 s` once, then `~0.42 s` per question) is a different and legitimate measurement
  recorded in `../../models-decisions/benchmark-results/2026-09-21-jev-sidebyside/`.

  **Correction, 2026-10-23.** An earlier version of this file claimed that amortised profile was
  unreproducible because "the mechanism does not exist for Harriet's own base architecture." That
  was wrong, and it was wrong in an instructive way: it read one capability flag and inferred the
  absence of a different mechanism. There are two, and they are not interchangeable:

  | capability | qwen35 | what it does |
  | --- | --- | --- |
  | `supportsSharedPrefixes` | false | freeze one prefix and `fork()` it into parallel branches — batch candidate scoring |
  | `supportsResumption` | **true** (`Qwen35DecoderAdapter:153`) | return to a position already read without re-reading it |

  Harriet's briefing asks its questions **sequentially** against one document, which needs
  resumption, not forking — and `Qwen35ForwardPass:358` captures the Gated DeltaNet state required
  to resume a retained prefix. The reuse in those records is real, and the numbers in the same file
  prove it independently: a 5.89 s prefill paid once against ten marginals near 0.42 s is not what
  re-reading the document per question would cost.
- **Harriet's accuracy here is 0.8833 against 0.8917 and 0.9000 in the two older records** — one
  to two items, at a cohort whose stated noise floor is one item. Consistent, not identical.
- **No TTFT or TPOT for Laya.** It answers in one non-autoregressive pass; there is no first token
  and no inter-token interval. Reporting a zero or a win on those axes would be fabrication. The
  comparable quantity is time to decision, which is what the table gives.

## Addendum, 2026-10-23: the fast projection kernel is 2.2x and is rejected

Laya had no fast projection path at all — `ModernBertDecisionForwardPass` took no kernel, so every
decision ran `TensorOps.ggufExactBatchedMatmul`, the exact F32 route. For a decision model that is
the whole cost of every call: there is no prefix to amortise.

A kernel seam was added (`ModernBertDecisionModel.open(Path, GgufBatchedMatrixKernel)`) and the Rust
shim's `Q8_0_F32_BATCHED_MATMUL` route measured on this host, same cohort, same scorer as above.

| | exact, shipped | `rust-ffm-quantized-v14` |
| --- | ---: | ---: |
| accuracy | **0.7833** (94/120) | 0.7583 (91/120) |
| Intelligence | **76.4** | 74.1 |
| Calibration (ECE) | 67.2 (0.1638) | **72.2 (0.1391)** |
| Speed | 61.8 | **68.5** |
| p50 / p95 per decision | 3.318 / 4.800 s | **1.492 / 2.188 s** |

**2.2x faster and three items worse. Rejected as the default.**

The reason is not the tolerance I happened to pick. A parity check written before the run — chosen
option equal, probabilities within a fixed 1e-3 — failed at 0.0063 on a `choice` case, and that
alone would be arguable, because Laya is `RANKED_DECISION` and the catalog already says its
probability is not a fitted one. What settles it is the cohort: **the two paths pick different
winners on 4 of 120 items**, so the kernel changes answers rather than last bits. Per family:
`tool_selection` 1.0000 to 0.9167, `extraction` 0.8750 to 0.8333, `intent` 0.7917 to 0.7500.

Shipping it would leave the catalog's published 0.783333 describing a path that no longer runs,
which is the one thing evidence discipline here exists to prevent. The tolerance was not widened
to make the check pass, and the seam's default stays `GgufBatchedMatrixKernel.none()`.

The seam is kept because it is what made this measurable at all, and because the win is real enough
to be worth earning: 2.2x on every decision, on a model whose latency is its weakest axis against
Harriet. ### Why the numbers differ, which is not what this file first said

Not accumulation order. `GgufTensorType::activation_format` in the kernel
(`model-kernels/src/lib.rs:306-312`) maps Q8_0 weights to `ActivationFormat::Q8_0`, so the shim
**quantizes the activations to eight bits** before the dot product; the exact Java path keeps them
in F32. That is a precision reduction on the input to every projection, which is the right trade
for a generative model — it is llama.cpp's own convention — and the wrong one for a model whose
output is an argmax, where a flipped winner is simply a wrong answer.

The capability is called `Q8_0_F32_BATCHED_MATMUL`, and the `F32` there describes the **output**.
An earlier version of this addendum, and a comment in the forward pass, read it as a promise about
the activations. Both are corrected; it was inferred from a name rather than read from the kernel.

### The fix has a precedent in the same file

`Q5_1` already maps to `ActivationFormat::F32`, and its row kernel is written to fold "exactly as
Java's `PinnedReduction.dot` folds" (`lib.rs:4961`) — native speed with the Java path's
accumulation. A Q8_0 weights x F32 activations route of the same shape is what this model needs,
added as its own capability and entry point rather than by changing the global mapping, because
flipping Q8_0 to F32 activations everywhere would slow every generative Q8_0 model for a precision
none of them need.

Acceptance criterion: reproduce accuracy 0.783333 and the published per-type cells on this cohort
through the kernel, then re-qualify.

Raw native arm: `laya-native-jev120.tsv`.

## The Q8_0 x F32-activation kernel: bit-exact, and 6.4x too slow

Built the route the section above specified: `WeightFormat::Q8_0F32Activation` (format code 8) with
`dot_q8_0_f32_activation_row_scalar`, transcribed from the Java path rather than approximated —
`quant as f32 * scale` as one multiply, because `Q8_0Dequantizer` writes `quant * scale` with no
addend; FMA into thirty-two per-lane accumulators, one Q8_0 block per iteration; and the
`((l4+l0)+(l6+l2)) + ((l5+l1)+(l7+l3))` fold from `reduceLanesPinned`. A new variant rather than a
change to `Q8_0`'s mapping, so every generative Q8_0 model keeps the quantizing route untouched.

**Parity is exact**, 10 cases, 50 options, against the Java exact path on this host:

| | |
| --- | --- |
| max absolute probability delta | **0.000e+00** |
| argmax agreement | 10/10 |

Not "within tolerance" — identical. The fold transcription is right, which is the half of this that
was uncertain.

**Speed is the problem.** Same 10 cases, same host:

| path | mean per decision |
| --- | ---: |
| Java exact (Panama, 256-bit) | **4.2 s** |
| this kernel (Rust, scalar) | 27.0 s |
| Rust quantizing Q8_0 route | 1.5 s (and three items wrong) |

6.4x slower than the Java path it reproduces, because that path is vectorized and this kernel is
not. A full 120-case run was abandoned at thirty minutes for the same reason.

**So it is not the default, and the reason has changed.** The accuracy objection is gone — this
route cannot move an answer, because it does not move a bit. What remains is purely that the scalar
kernel is slower than Java's vectorized one, which is a SIMD gap and nothing to do with precision.

The kernel is kept rather than reverted because it is now the parity oracle for that SIMD work: an
AVX2 implementation has something to be bit-compared against, which is exactly the role Q5_1's
scalar entry point serves ("the scalar entry point remains usable by direct ABI correctness
tests"). The remaining task is a `Q8F32ActivationAvx2` variant mirroring `Q5_1Avx2`.

Raw arms: `laya-exact-activation-kernel-10.tsv`, `laya-java-exact-10.tsv`.
