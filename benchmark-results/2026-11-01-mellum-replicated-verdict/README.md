# mellum, replicated: `FAILED_RELATIVE_GATE` at 0.763. The single passing run was an outlier.

**This directory exists because one run said `QUALIFIED` and three replications say it does not.**

Measured 2026-11-01, pod `9lpalxaf8vzgpg`, 16 vCPU of an AMD EPYC 7713 (no AVX-512), workload
`coding`, template `chatml-no-think`, `--warmups 1 --iterations 3`, ollama 0.40.13 on the identical
artifact, both arms sequential in one session per replication, `exclusions: {}` throughout.

| run | our decode | ollama decode | decodeRatio | verdict |
|---|---:|---:|---:|---|
| first run (`outlier-run-verdict.NOT-A-QUALIFICATION.json`) | 29.23 | 36.03 | **0.8112** | QUALIFIED |
| rep1 | 27.80 | 37.64 | 0.7386 | FAILED_RELATIVE_GATE |
| rep2 | 28.20 | 35.87 | 0.7862 | FAILED_RELATIVE_GATE |
| rep3 | 27.76 | 36.32 | 0.7643 | FAILED_RELATIVE_GATE |
| **replicated mean** | **27.92 ± 0.24** | **36.61 ± 0.9** | **0.7630** | **FAILED_RELATIVE_GATE** |

**Passes: 0 of 3.** Floor is 0.80. Contribution passed in every run at `modelAnswerRate` 0.7778 and
`modelAnswerCorrectRate` 1.0000.

## Why the first run read as a pass

Both arms drew favourably at once. Our decode came in at 29.23 against a replicated 27.92 -- itself
about 1.3 sigma high -- while the comparator drew 36.03 against a replicated 36.61. Neither
deviation is remarkable alone; together they moved the ratio from 0.763 to 0.811 and across the
floor.

That is the whole hazard: the comparator's own spread on this shared box is about ±8%, so running a
pair until the ratio clears 0.80 will eventually succeed, and the result would be a published
qualification that the next run contradicts. A qualification record is a claim that cannot be
retracted, so a pass obtained from the favourable end of a comparator's noise is not a pass. The
`QUALIFIED` verdict in `outlier-run-verdict.json` is retained here deliberately, as the thing that
was not believed.

**Read `endToEndLatencyRatio` with care in all four files** (0.2604 to 0.3395). It does not mean we
are three times faster: ollama generated 1488 output tokens to our 825 in every run, so the two
arms did different amounts of work. Only `decodeThroughputRatio`, a rate, is comparable here.

## The optimisation ladder behind these numbers, replicated

| stage | decode | note |
|---|---:|---|
| baseline | 21.70 | |
| + vectorised swiGLU | 21.08 | prefill +76%, decode unmoved |
| + Q5_0 independent dispatch | 23.87 | +13.2% |
| + fused grouped attention | **27.92** | +17.0%, replicated across three runs |

Cumulative decode **+28.7%** from three native-shim fixes, all replicated. The fused-attention gain
survives the obvious objection: our generated tokens fall from 1134 to 825 when it is enabled, which
shortens the average KV context and inflates tokens/second mechanically, so it was checked per case
at **identical** token counts -- +16.7%, +30.3%, +22.8%, +19.5%, +18.2%, about +21% -- and the token
drop traces almost entirely to one case, `gradle-java-25`, going 138 to 45 tokens. That is the
documented token flip, not a throughput artifact. Our 825 is stable across all three replications,
so it is deterministic rather than noise.

## Where this leaves the port

Decode needs 0.80 of a 36.61 comparator, so **29.29 tok/s against our 27.92 -- a further 4.8%**.
That is down from the 25% it needed before this work and from 57% for lfm2moe, and the remaining
cost is the quantised GEMV's bandwidth efficiency: mellum reads roughly 1.6 GB of weights per token,
so 27.92 tok/s is about 45 GB/s against the comparator's 59 on the same 16 cores.

Not qualified, and not claimed to be. One untested lever remains (`q6BatchedKernel`, a variant
choice for the Q6_K output head, which at 98304 x 2304 is a real share of what each token reads);
beyond it the gap is kernel work rather than wiring.

## Null: the Q6 two-query-block kernel (added 2026-11-01)

The last untested lever. `q6-batched-kernel` was `DISABLED` by property rather than unsupported --
`models.purejava.q6BatchedKernel` selects one-query blocks by default -- and mellum's output head is
Q6_K at 98304 x 2304, a real share of what each token reads, so the specialised variant was worth
measuring.

| configuration | decode runs | mean | tokens |
|---|---|---:|---:|
| one-query-block (default) | 27.80, 28.20, 27.76 | **27.92** | 825 |
| two-query-block | 28.46, 27.90 | **28.18** | 825 |

**+0.9%, which is inside the noise.** The default's runs span 27.76-28.20 and the variant's span
27.90-28.46; the means differ by 0.26 tok/s against spreads of 0.44 and 0.56. Output tokens are
identical at 825 in every run, so unlike the fused-attention measurement there is no generation-
length confound to unpick -- this is a clean comparison and it shows nothing.

The toggle was observable, which is why the null can be trusted: the run reports
`q6-batched-kernel: ENABLED` with `kernel: two-query-block, requested-kernel: two-query-block`, so
the variant executed rather than silently declining. A flat result from a setting that never took
effect would be "no data", not "no effect".

Ratio against the comparator mean: **0.7698, FAIL**. Artifacts `q6-two-query-run1.json` and
`q6-two-query-run2.json`.

## Final standing for mellum

`FAILED_RELATIVE_GATE`, replicated, at **0.763-0.770** against a 0.80 floor. Every dispatch-level
lever is now taken or measured null:

| lever | decode effect |
|---|---|
| kernel seam, grouped expert dispatch | part of the 21.70 baseline |
| vectorised swiGLU | prefill +76%, decode unmoved |
| Q5_0 independent dispatch | **+13.2%** |
| fused grouped attention | **+17.0%** (+21% at matched token length) |
| native kernel threads 16 -> 8 | **null**, 33% worse |
| Q6 two-query-block | **null**, +0.9% within noise |

Decode went 21.70 -> 27.92, **+28.7%**, and needs 29.29. The remaining 4.8% is the quantised GEMV's
bandwidth efficiency -- about 45 GB/s against the comparator's 59 on the same 16 cores -- which is
kernel work, not wiring, and is where this stops.
