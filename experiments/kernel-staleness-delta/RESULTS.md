# Result: the stale kernel does not require requalifying anything

Derived by `analyse.py` from the raw artifacts in `raw/`. Protocol and decision rule were
registered in `README.md` before the run.

```
python3 experiments/kernel-staleness-delta/analyse.py
```

## Kill criterion satisfied

The two arms loaded different kernels, proven by the `.so` digest each JAR declares:

| arm | jar | `.so` sha256 |
| --- | --- | --- |
| A | `models-kernels-linux-x86_64-ee8dab27-sept.jar` | `ce25a986…` |
| B | `models-kernels-linux-x86_64-0.3.56.jar` | `064cfae7…` |

## H1 — numerics: identical

`totalOutputTokens`, `correctAnswerRate` and `modelAnswerRate` are identical across all four runs
of each model, and `performanceTier` is `PRODUCTION_READY` in all eight. The two kernels compute
the same thing.

**The campaign's correctness findings therefore stand**, including the published nulls in
`benchmark-results/2026-10-09-max-tokens-sweep`. Those turn on `modelAnswerRate` and
`truncatedAnswerRate`, not on speed.

## Effect on the two inputs the gate actually reads

The relative gate reads `decodeThroughputRatio` against a 0.80 floor and `endToEndLatencyRatio`
against a 1.50 ceiling. Prefill throughput is not a gate input.

| model | p50 decode tok/s | endToEnd p95 ms |
| --- | --- | --- |
| `qwen2_5_coder_0_5b_instruct_q4_k_m` | 69.910 → 69.871 (**−0.05%**, own-arm spread 2.18%) | 692.1 → 682.0 (−1.46%, spread 6.44%) |
| `qwen2_5_coder_1_5b_instruct_q5_k_m` | 28.924 → 29.023 (**+0.34%**, spread 0.14%) | 1720.2 → 1750.5 (+1.76%, spread 0.91%) |

Worst case against us: decode **−0.05%**, end-to-end p95 **+1.76%**.

## Why that settles it

Every QUALIFIED verdict's headroom is larger than the effect. The tightest, on both axes, is
`eurollm_1_7b_instruct_q4_k_s`: **1.4%** above the decode floor against a −0.05% effect, and
**3.2%** below the end-to-end ceiling against a +1.76% effect. No verdict changes sign.

**So: no requalification.** The twelve vetted models can be re-run against the released build for
provenance — their reports must name a released version to land — but not to correct a number.

`eurollm_1_7b_instruct_q4_k_s` is the one to watch: 3.2% against 1.76% is under a 2× margin, and it
is the model whose qualification depended on the workload-match correction. It is not at risk by
this measurement, and it has the least room to spare if anything else moves.

## What this experiment did NOT test, and why it does not matter here

The kernel delta has two parts: the worker-pool activation-gate fix (`fd0de371`, a shared path) and
new Q5_1 support. **Only the first was exercised.** Both test models turned out to carry no Q5_1
tensors at all — `qwen2_5_coder_1_5b_instruct_q5_k_m` is Q5_K 72% / Q6_K 28%, because `Q5_K_M` is a
scheme name and does not imply Q5_1 blocks. Choosing it to isolate Q5_1 was an error.

It does not change the conclusion, because none of the thirteen pending models carries Q5_1 either
— verified by range-fetching all thirteen headers. The Q5_1 path cannot affect them. It remains
untested, and any future model carrying Q5_1 is outside what this result covers.

## Why the staleness was invisible

Both kernels declare `abi=6`, so the stale one loaded cleanly and every report recorded
`native-kernel-abi = 6` consistently. ABI compatibility is not identity, and nothing recorded the
`.so` digest. Both workers now require the kernels JAR as a named input, log that digest, and are
covered by a check that fails if any fleet script names a measurement input literally.
