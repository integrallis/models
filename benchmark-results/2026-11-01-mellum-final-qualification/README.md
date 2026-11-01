# mellum, first admitted qualification: fails on decode by 25%, and is NOT terminal

**Verdict: `FAILED_RELATIVE_GATE`** on decode alone. Unlike lfm2moe, this one has identified,
untried levers, so the port is kept.

Measured 2026-11-01, pod `9lpalxaf8vzgpg`, 16 vCPU of an AMD EPYC 7713 (no AVX-512), both arms
sequential in one session, `--warmups 1 --iterations 3`, workload `coding`, template
`chatml-no-think`, ollama 0.40.13 on the identical artifact. `exclusions: {}`.

| | candidate | ollama | ratio | gate |
|---|---:|---:|---:|---|
| p50 decode tok/s | 21.70 | 33.93 | **0.6396** | >= 0.80 **FAIL** |
| p50 prefill tok/s | 18.69 | 90.46 | 0.2066 | not gated |
| end-to-end latency | | | **1.1002** | <= 1.50 **PASS** |

**Contribution is strong**: `modelAnswerRate` **0.7778** -- 7 of 9 cases answered by the model, all
7 correct -- against a 1/3 minimum and a 0.90 correctness minimum. That is substantially better
than lfm2moe's 0.4444, so this is a more useful model on its matched workload, and `coding` is the
right workload for it (the runbook's `code-completion`/`fim` -> `coding` mapping).

Reaching the floor needs decode at 27.14 tok/s, **+25.1%**.

## Why this is not the same situation as lfm2moe

lfm2moe needed +57% with nothing left to try: its profile put ~75% of runtime in a quantised GEMV
that already implements ggml's algorithm, and both cheap levers came back null (thread fan-out
-33%, repack-tile parity). mellum needs +25% with two levers that have **not** been tried, each of
them the same class of defect that was worth +17.6% on lfm2moe:

1. **It was running the scalar SwiGLU.** `LlamaForwardPass.usesVectorizedSwiGlu()` admitted only
   QWEN2 and QWEN3, so mellum fell through to `TensorOps.swiGlu` -- a loop calling `Math.exp` once
   per element. mellum routes 8 of 64 experts across 28 blocks: 224 SwiGLU calls per token over
   `expert_feed_forward_length` 896 each, about 200k scalar exponentials per token. It declares no
   `swiglu_clamp_exp`, so `activateExpert` does not intercept it with the clamped scalar route
   first. **Admitted to the vectorised path in this change set; measurement pending.**

   That whitelist is about output parity, not speed -- the vectorised form evaluates `exp` by
   polynomial, so an architecture moved onto it stops reproducing the greedy outputs its published
   records carry. mellum has no published records: the catalog's 80 qualified entries contain no
   mellum, checked rather than assumed.

2. **Its expert down-projections are Q5_0, which the shim groups neither way.** 14 of 28 layers
   therefore dispatch their down projections one expert at a time while their gate/up projections
   group. `RustGgufBatchedMatrixKernel.groupable` excludes Q5_0 on a measured basis -- 37.33 against
   38.94 tok/s -- but that method's own javadoc records the figures as having **no committed raw
   artifact** and being "due a re-run under protocol", and they were taken on Qwen2.5-0.5B *dense*
   projections, which is nothing like 8 expert down-projections of 2304 x 896. Re-measuring Q5_0
   grouping for this shape is the sanctioned next step, not an override of a settled result.

## Disposition

Kept. +25% from two untried levers of the same kind that already delivered +17.6% elsewhere is not
a terminal gap, and the contribution gate passing at 0.7778 with every model answer correct means
qualification is worth having. If both levers land and decode still falls short of 0.80, the
remaining cost is the same quantised-GEMV ceiling lfm2moe hit and the same conclusion follows.
