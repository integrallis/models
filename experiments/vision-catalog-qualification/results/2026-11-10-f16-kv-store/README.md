# F16 KV store: decode effect on the registered host class

## What was measured and why

`NativeCausalAttention` already rounded every retained K/V element through F16, then kept the
result in `float[]`. Storing the F16 bits instead halves the bytes a decode step's read of the
retained prefix touches, and because widening binary16 to binary32 is exact it changes no operand.
This run measures what that is worth to **our** decode.

It had to be measured rather than carried over. models#408 recorded 1.24-1.27x for the external
reference's own F16 KV option, and a bytes-read traffic model predicting 1.077x for the same
halving — about three times apart, with no verified explanation. Neither number transfers to our
code: the mechanism is shared, the magnitude is not.

## Host and inputs

| | |
| --- | --- |
| instance | `m6a.4xlarge` — the registered qualification host class |
| CPU | AMD EPYC 7R13 (Zen 3, AVX2, **no AVX-512**), 8 cores / 16 threads |
| threads | `VISION_THREADS=8`, the physical core count |
| candidate | `05595e73b684307b0d4fb589a2e7fb475846ae2e` |
| baseline | `168b7df4bd8eb5bc09e80833569c24ffae4f194d`, the published 0.5.18 runtime |
| external reference | llama.cpp `2cbca389f515df455412dafa07e2b1359cc72a6d` |

Both runs executed sequentially on the same box as the baseline, against the same pinned artifacts
and the same external decoder reference build. `cpu.txt`, `java-version.txt`,
`payload-sha256.txt`, `native-library-sha256.txt` and `candidate-commit.txt` are recorded per run.

## Result

Reproduce the table from the committed artifacts — it is not hand-typed:

    python3 derive.py --baseline baseline-0.5.18-admission-attempt.json

| schedule | reference KV | baseline decodeRatio | decodeRatio | change |
| --- | --- | --- | --- | --- |
| initial, 2 warmups / 5 repetitions | f32 | 0.88712 | 0.80947 | 1.096x faster |
| initial, 2 warmups / 5 repetitions | f16 | 1.09546 | 0.98606 | 1.111x faster |
| confirmation, 3 warmups / 7 repetitions | f32 | 0.88802 | 0.81583 | 1.088x faster |
| confirmation, 3 warmups / 7 repetitions | f16 | 1.10080 | 0.98727 | 1.115x faster |

`greedyTokensMatch` and `numericalPass` are true on all four arms, and both runs wrote
`STATUS = INTEGRATION_PERFORMANCE_PASS` where the baseline's confirmation run did not.

**Our decode is 1.09-1.12x faster**, consistently across four arms and two warmup schedules. That
matches neither of the two prior figures, which is the point of having measured it.

## The gate this moves

`integrationConfirmation` is the 3w/7r row. On the published 0.5.18 runtime it read **1.10080
against a 1.10 ceiling — 0.0727% over** — and it was the sole remaining Q4_K_S blocker, the other
seven admission gates having passed. It now reads **0.98727**, 10.2% inside the ceiling.

## What this run does not establish

- **The task stage.** Q4_K_M also failed `strongestReferencePerformance`, which measures the same
  quantity over the four registered performance cases at the task stage (its failing case read
  `decodeMs = 1.10279`, the other three 1.0917 / 1.0696 / 1.0658). This run covers the integration
  stage only. That gate is re-run in full, not extrapolated from the figures above.
- **Admission.** These are two integration arms, not the eight-gate admission. No qualification
  claim follows from this directory.
- **Any other host class.** Measured on EPYC 7R13 with AVX2 and no AVX-512. The effect is a
  memory-traffic one, so it is expected to vary with cache and bandwidth; this run says nothing
  about how much, on any other machine.
