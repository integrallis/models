# mellum: two native-shim defects fixed, decode +10%, still short of the gate

Measured 2026-11-01 on pod `9lpalxaf8vzgpg`, 16 vCPU of an AMD EPYC 7713 (no AVX-512), workload
`coding`, template `chatml-no-think`, `--warmups 1 --iterations 3`, ollama 0.40.13 on the identical
artifact, both arms sequential in one session. `exclusions: {}` on every verdict.

## The ladder

| stage | prefill | decode | decodeRatio | e2eRatio |
|---|---:|---:|---:|---:|
| baseline | 18.69 | 21.70 | 0.6396 | 1.1002 PASS |
| + vectorised swiGLU | **32.98** | 21.08 | 0.5352 | 1.2525 PASS |
| + Q5_0 independent dispatch | 29.10 | **23.87** | **0.6422** | **0.9654 PASS** |

`modelAnswerRate` 0.7778 and `modelAnswerCorrectRate` 1.0000 at every stage -- 7 of 9 cases
answered by the model, all 7 correct. End-to-end we are now **faster than the comparator** (0.9654).
Decode is the only failing gate.

## Read the ratios against the comparator's own spread, not individually

ollama's mellum decode on this box, across three sessions: **33.93, 39.42, 37.17** tok/s. Mean
36.84, spread about +/-8%. That is why the middle row's `decodeRatio` looks like a regression when
its candidate decode barely moved -- the comparator got faster, not us.

Normalised against the 36.84 mean, our arm went **0.589 -> 0.648**. The within-arm decode numbers
(21.70, 21.08, 23.87) are the reliable signal; any single ratio carries the comparator's spread.

## The two defects

**1. The vectorised SwiGLU was whitelisted away from mellum.** `usesVectorizedSwiGlu()` admitted
only QWEN2 and QWEN3, so mellum took `TensorOps.swiGlu` -- a scalar `Math.exp` per element -- across
8 of 64 experts in each of 28 blocks: 224 calls a token over `expert_feed_forward_length` 896, about
200k scalar exponentials. It declares no `swiglu_clamp_exp`, so `activateExpert` does not intercept
it with the clamped route first.

That whitelist is **output parity**, not performance: the vectorised form evaluates `exp` by
polynomial, so an architecture moved onto it stops reproducing the greedy outputs its published
records carry. mellum has no published records -- checked against all 80 qualified catalog entries,
not assumed -- so nothing depends on its arithmetic.

Worth **+76% prefill and nothing on decode**, which is itself the useful finding: at batch 1 the
activation is a small share of a memory-bound token, while in a batch-32 prefill it is a serial
scalar loop across the whole batch.

**2. Q5_0 was refused on the independent dispatch for a grouped-dispatch reason.** The grouped entry
point refuses Q5_0 because grouping shares one quantized activation across the group and that
sharing measured slower (37.33 against 38.94 tok/s), which is why the grouped Q5_0 capability was
retired -- its bit is in `registry.json`'s `retiredCapabilityBits`. An **independent** dispatch gives
every matrix its own activation, so that measurement never applied to it, and
`compute_batched_row_range` already carried the Q5/Q5Avx2/Q5Neon arms. The guard was a copy of the
grouped one: one measurement had retired two capabilities.

This is not hypothetical for mellum. Read from the artifact's tensor table rather than the feasibility
notes:

```
Mellum2-12B-A2.5B-Thinking-Q4_K_M.gguf  —  339 tensors
  Q4_K 155   attn_q/k/output, ffn_{gate,up}_exps, token_embd
  F32  141   norms, ffn_gate_inp router
  Q6_K  15   attn_v, output head
  Q8_0  14   ffn_down_exps, 14 of 28 blocks
  Q5_0  14   ffn_down_exps, the other 14 blocks
```

The Q8_0 half took one dispatch per layer. The Q5_0 half fell back to one per expert: 8 a layer,
112 a token. Fixing it is the **+13.2% decode** above, and the first change that moved decode at all.

Three guards all consulted `supportsGrouped` and all had to be threaded: `isIndependentEligible`,
`isRaggedIndependentEligible`, and the shared `validateGroupedProjection` (now
`validateProjectionForDispatch`, with the dispatch named explicitly). Grouping still refuses Q5_0 and
that is pinned by test, so widening one path cannot quietly widen the other.

## What is already on, checked rather than assumed

For mellum the `ExecutionPlanner` diagnostics are meaningful -- it is a Llama-path model -- and they
report `grouped-projections: ENABLED` with `qkv: grouped, gate-up: grouped`, and `batched-prefill:
ENABLED` at batch 32. Those were working before this change set. (The same block is **false** for
lfm2 models, which is filed separately: it claims "no Q4_0 projection tensors" for a file that is
132/256 Q4_0.)

## What is left

Decode needs 0.80 of a ~36.8 tok/s comparator, so about **29.5 tok/s against our 23.87 -- a further
24%**. mellum reads roughly 1.6 GB of weights per token, so 23.87 tok/s is about 38 GB/s against the
comparator's 59 on the same 16 cores. The dispatch-level wins are now taken; what remains is the
quantised GEMV's bandwidth efficiency, the same ceiling lfm2moe reached, and that is a kernel
project rather than a wiring one.

Not terminal, and not claimed to be: contribution passes comfortably, end-to-end passes, and the gap
is a known quantity with a named cause.
