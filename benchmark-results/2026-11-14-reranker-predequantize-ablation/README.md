# Pre-dequantizing K-quant BERT weights (#360): 1.33x faster warm scoring, 1.77x slower cold load, and it changes the scores

**Measured here.** Nothing in this directory is projected. Every number below is read from a file in
`runs/` or `runs-regression/` by `derive.py`; the tables are its output pasted verbatim.

    python3 derive.py runs
    python3 derive.py runs-regression

## Why this was measured

`perf(#360)` (2d9e13d0) widened every Q4_K/Q5_K/Q6_K tensor in a BERT model to F32 at load. Its
commit message has a heading reading "Measured improvement:" above text saying the before/after
experiment still needed running. It was not run, and the change was on `main` as part of the 0.5.21
release candidate (#452). This is the measurement, on the registered qualification host class.

## Host, artifact, protocol

| | |
| --- | --- |
| instance | `m6a.4xlarge`, AMD EPYC 7R13 (Zen 3, AVX2, **no AVX-512**), 8 cores / 16 threads |
| JDK | Temurin 25.0.5 |
| artifact | `ms-marco-MiniLM-L-6-v2-q4_k-imatrix-g7c-f7.gguf`, sha256 `0752a92bc33289f3fe230d1a33cce471f1a7fe1c8d5a491ca7fff6b068b0fc83`, 19,986,112 bytes — the registered fixture `ms_marco_minilm_l6_v2_q4_k_imatrix_g7c_f7` |
| experiment | `MsMarcoMiniLmRerankerPerformanceExperiment`, `warmups=5 pairIterations=60 batchIterations=20`, six documents |
| route | `GgufRerankingModel`, which installs no batched matrix kernel — the pure-Java path #360 targets |

Three arms, every one on `origin/main` (`cdbc9edc`) plus the #456 fix — without it `scoreAll`
throws on this probe for any batch of unequal lengths, on every arm — built as separate worktrees
and run **interleaved A, B, C for five rounds**, so drift hits each arm equally:

| arm | what | commit |
| --- | --- | --- |
| `A-off` | `preDequantizes` forced `false`: the pre-#360 arithmetic | `679ecabd` |
| `B-narrow` | the #454 fix: projections only, and only when no kernel takes the type | `a32e13f6` |
| `C-main` | `main` as it stood: the original #360 condition | `32d70190` |

Each run is one forked JVM wrapped in `/usr/bin/time -v`; `*.out` holds the experiment's JSON,
`*.err` the wrapper's block. `cpu.txt`, `java-version.txt` and `sha-*.txt` are recorded alongside.

## Result

```
| metric | V-0520 | A-off | B-narrow | C-main | B/A | C/B | A/V |
| --- | --- | --- | --- | --- | --- | --- | --- |
| coldLoadMillis | 168.57 [168.57..168.57] | 221.79 [217.80..241.56] | 392.99 [375.48..404.13] | 392.43 [388.46..412.34] | 1.772 | 0.999 | 1.316 |
| pairP50Millis | 67.06 [67.06..67.06] | 79.88 [76.65..80.84] | 58.96 [58.75..58.98] | 58.80 [58.03..59.02] | 0.738 | 0.997 | 1.191 |
| pairP95Millis | 101.26 [101.26..101.26] | 99.89 [99.74..101.55] | 74.17 [73.76..76.44] | 73.90 [73.45..74.59] | 0.742 | 0.996 | 0.986 |
| batchP50Millis | 451.15 [451.15..451.15] | 596.82 [593.38..603.84] | 450.19 [447.88..451.60] | 447.76 [446.77..451.41] | 0.754 | 0.995 | 1.323 |
| batchP95Millis | 646.40 [646.40..646.40] | 598.35 [594.12..605.84] | 450.82 [448.83..453.40] | 448.50 [447.66..452.58] | 0.753 | 0.995 | 0.926 |
| batchP50DocumentsPerSecond | 13.30 [13.30..13.30] | 10.05 [9.94..10.11] | 13.33 [13.29..13.40] | 13.40 [13.29..13.43] | 1.326 | 1.005 | 0.756 |
| maxRssMiB | n/a (no /usr/bin/time wrapper) | 675.75 [596.92..688.85] | 609.29 [596.96..628.41] | 613.36 [598.49..634.84] | 0.902 | 1.007 | n/a |
```

`V-0520` here is **one** run of released `v0.5.20` with no timing wrapper, kept as a numerics
reference (next section); its timing column is a single sample and is not a measurement. The
five-round timing comparison against v0.5.20 is in `runs-regression/`, below.

- **Warm scoring is 1.33x faster with #360 on** — pair p50 79.9 -> 59.0 ms, batch p50 597 -> 450
  ms. Spreads are tight and do not overlap. The #360 commit was right about the direction; this is
  the measurement it did not have.
- **Cold load is 1.77x slower** — 222 -> 393 ms, +171 ms every time the model is loaded. The
  original commit did not state this cost.
- **B and C are the same measurement** at 0.995–0.999. Predicted from the artifact's header: its
  tables are **Q8_0**, which #360's K-quant condition never matched, so the "projections only"
  restriction is inert on this model and the two arms run identical code. Their difference is the
  noise floor of the harness.
- **`maxRssMiB` is not a memory result.** The spreads overlap across all arms; RSS of a
  gradle-forked JVM does not resolve a +12 MB change. The memory figure is the exact one, from the
  GGUF header:

| role | type | tensors | stored | as F32 | effect of #360 |
| --- | --- | --- | --- | --- | --- |
| table (`token_embd`, `position_embd`) | Q8_0 | 2 | 12.66 MB | 47.67 MB | none — not K-quant |
| projection | Q4_0 | 30 | 3.98 MB | 28.31 MB | none — not K-quant |
| projection | Q4_K | 6 | 1.99 MB | 14.16 MB | **+12.17 MB, 7.11x** |
| projection | F32 | 6 | 0.01 MB | 0.01 MB | none |
| other (norms, biases) | F32 | 61 | 0.71 MB | 0.71 MB | none |

  So on this model #360 is exactly six tensors and +12.2 MB. Q4_K -> F32 is 7.11x, not the "4x"
  the commit stated. The 30 Q4_0 projections — two thirds of projection weight bytes — still
  dequantize on the fly, which bounds what the change can recover.

## It changes the scores

```
scoreChecksum per arm (distinct values, 9dp): {'V-0520': [-161.44592762], 'A-off': [-161.44592762], 'B-narrow': [-160.842909813], 'C-main': [-160.842909813]}
A-off reproduces released v0.5.20 scores to 9dp: True
B-narrow equals A-off to 9dp: False
C-main equals B-narrow to 9dp: True
```

Both arms are deterministic — one distinct score vector across all five runs each — and they differ
from one another. Per document, on the six-document `scoreAll` probe:

| document | off | on | shift |
| --- | --- | --- | --- |
| 0 | +8.81581 | +8.80164 | −0.01417 |
| 1 | −10.98703 | −10.97222 | +0.01482 |
| 2 | +7.30477 | +7.28886 | −0.01591 |
| 3 | −11.26974 | −11.27076 | −0.00102 |
| 4 | −5.15279 | −5.11455 | +0.03824 |
| 5 | −4.85561 | −4.81727 | **+0.03834** |

Full ordering `[0, 2, 5, 4, 1, 3]` is preserved; the pair-path mean shift is +0.0101 per score.

The shift is expected, not a bug: F32 dot products on widened weights are **different arithmetic**
from the quantized-activation K-quant matmul (`TensorOps.ggufBatchedMatmul` with Q8 activation
quantization), not a reordering of the same operations. With #360 off the scores equal released
v0.5.20 to nine decimal places; with it on they do not.

## Decision

This model is a qualified catalog model. Its record is a claim about the exact scores it produces.
An arithmetic change that moves those scores cannot be the default — the same rule #408 applies to
F16 KV — and a 1.77x cold-load regression on every load is not something to impose for a warm-path
win a caller did not ask for. So #454 makes #360 **opt-in, default off**:
`-Dmodels.purejava.bert.preDequantizeKQuant=true`. Off ships the qualified arithmetic bit for bit;
on buys the measured 1.33x for anyone who accepts the trade.

## "On" fails the model's qualification gate — `qualification/`

`msMarcoMiniLmRerankerIntegrationTest` — the model's own pinned equivalence tests, integration-tagged
and never run by CI — executed on arms A and B on the same box, same artifact. Per-test results are
in `qualification/<arm>-results/*.xml`, gradle output in `qualification/<arm>.log`.

| test | A: off | B: on |
| --- | --- | --- |
| `sentencePairTokensAndTypesMatchThePinnedReference` | pass | pass |
| `repeatedScoresAreDeterministicAndPairOrderMatters` | pass | pass |
| `correctedQuantizedScoresMatchTheOnnxScaleAndOrdering` | pass | **FAIL** |
| `batchScoresMatchSequentialScores` | pass | pass |

The failure: document 5 scores `-4.817265` against the pinned ONNX reference `-4.892`, tolerance
`0.05`, difference `0.0747`. That is the same +0.038 shift in the per-document table above, landing
outside the envelope the model was qualified in. So the score change is not cosmetic: **with #360
on, this model no longer passes its own qualification.** Off, all four pass — and
`batchScoresMatchSequentialScores` passing on A is the real-model confirmation of the #456 fix.

## The pure-Java reranker path regressed between v0.5.20 and `main` — `runs-regression/`

A separate question the single `V-0520` run above raised but could not answer. Released `v0.5.20`
and `A-off` (`main` + #456, #360 off — the arithmetic that ships), five interleaved rounds, same
settings, same wrapper:

```
| metric | V-0520 | A-off | A/V |
| --- | --- | --- | --- |
| coldLoadMillis | 156.73 [156.14..203.71] | 226.50 [222.16..259.64] | 1.445 |
| pairP50Millis | 62.56 [62.31..63.08] | 79.64 [76.71..80.92] | 1.273 |
| pairP95Millis | 100.30 [99.32..100.92] | 100.39 [99.80..101.83] | 1.001 |
| batchP50Millis | 449.42 [445.78..452.62] | 604.77 [596.25..609.28] | 1.346 |
| batchP95Millis | 645.60 [643.29..650.80] | 606.45 [596.70..610.53] | 0.939 |
| batchP50DocumentsPerSecond | 13.35 [13.26..13.46] | 9.92 [9.85..10.06] | 0.743 |
| maxRssMiB | 585.31 [581.04..599.05] | 667.93 [596.64..701.49] | 1.141 |
```

- **`score()` (sequential pair path) is 1.27x slower** on `main`: 62.6 -> 79.6 ms, spreads disjoint.
- **`scoreAll` is 1.35x slower**: 449 -> 605 ms. On v0.5.20 `scoreAll` is the sequential loop; on
  `main` it is #440's batched `encodeBatchCls`. The batched path is slower than the loop it replaced
  on this host, by a third.
- **Cold load is 1.45x slower**: 157 -> 227 ms.
- **Numerics identical to 9 dp** across all ten runs, so this is speed, not arithmetic.

This is why the ablation's `C-main` batch p50 (448 ms) looked level with v0.5.20 (449 ms): #360-on
had bought back what the rest of `v0.5.20..main` lost. With #360 correctly off, 0.5.21 as it stands
ships a reranker path 27–35% slower than 0.5.20. The culprit is being bisected across the commits in
that range in a separate run; nothing here names one.

## What this does not establish

- **Any other model.** The gain is bounded by the K-quant share of projection work, which is a
  third of projection weight bytes here and differs per quantization. A model with K-quant tables
  would also be affected by the "projections only" restriction, which this one cannot exercise.
- **Any other host.** Memory-traffic and dequantization costs do not transfer across cache and
  bandwidth; this is EPYC 7R13 with AVX2.
- **The native `bert`/`nomic-bert` path.** `PureJavaBackend` installs a real kernel there; the
  fix's "no kernel takes the type" restriction leaves that path quantized whatever the property
  says, and nothing here measures it.
