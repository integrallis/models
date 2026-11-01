# lfm2moe decode and prefill optimisation: what moved, and what the profile actually said

Measured 2026-11-01 on RunPod pod `9lpalxaf8vzgpg`, 16 vCPU of an AMD EPYC 7713 (no AVX-512), each
arm `--warmups 1 --iterations 3`, 27 of 27 cases successful, same host and protocol as
`../2026-11-01-lfm2moe-ollama-qualification/`.

| stage | prefill p50 | decode p50 | decode vs ollama | artifact |
|---|---:|---:|---:|---|
| baseline (seam only) | 25.54 | 23.88 | 0.3974 | `../2026-11-01-lfm2moe-ollama-qualification/candidate.json` |
| + vectorised attention and swiGLU | 28.92 | 27.67 | 0.4605 | `lfm2moe-perf1.json` |
| + grouped Q/K/V | **29.32** | **28.08** | **0.4674** | `lfm2moe-perf2.json` |

Cumulative: decode **+17.6%**, prefill **+14.8%**. Comparator unchanged at 60.08 decode and 139.31
prefill (ollama 0.40.13, same artifact SHA-256). Contribution held at `modelAnswerRate` 0.4444 and
`modelAnswerCorrectRate` 1.0000 at every stage. **Still `FAILED_RELATIVE_GATE`**: the floor is 0.80.

## Provenance note on commit e7d15331

That commit contains all three changes including `projectQkv`, but its message quotes 23.88 ->
27.67 and 25.54 -> 28.92 -- the figures measured **before** the Q/K/V grouping was added. The code
it ships measures 28.08 and 29.32, the third row above. Recorded here rather than by rewriting the
pushed commit.

## The profile that chose this work was misread

The two optimisations were picked from a JFR profile that appeared to show `attend` and
`TensorOps.swiGlu` at 811 of 1065 samples, 76% of cost. That was wrong, and the correction is the
more useful result.

`jdk.ExecutionSample` samples threads in **Java** state only. Every FFM downcall into the Rust
kernels runs in native state and is recorded in `jdk.NativeMethodSample`, which was not read:

```
jdk.NativeMethodSample   5140 events
jdk.ExecutionSample      1538 events
```

Java was ~23% of sampled time. The two frames were ~12% of **total**, so 0.76 x 23% predicts about
17% -- against 15.9% measured for them, and 17.6% once Q/K/V grouping is added. The gains are real;
the reasoning that selected them was not. `baseline-decode.jfr` holds both event types, so this is
checkable.

### Where the time actually is

Native samples, leaf frame:

```
2469  NativeKernelLibrary.invokeBatched                        (non-grouped projections)
1556  NativeKernelLibrary.invokeGrouped                        (grouped experts)
1001  NativeKernelLibrary.quantizedF32IndependentBatchedMatmul
----
5026  of 5140, about 75% of total runtime
```

against `attend` at 6.9% and `swiGlu` at 5.3% of total.

**This retires the dispatch-count thesis in its MoE form.** The non-grouped entry costs *more* than
the grouped expert entry, so expert dispatch was never the dominant cost for this model.

**And the kernel is not naive.** `WeightFormat::Q4_0` maps to `ActivationFormat::Q8_0` with
`selected_q4_kernel()` (`backend-native/src/main/rust/model-kernels/src/lib.rs:345,355`), i.e. the
integer Q4_0 x Q8_0 AVX2 dot that ggml's `vec_dot_q4_0_q8_0` uses. The remaining 1.71x is tuning of
a correct kernel, not a missing algorithm.

## Why Q/K/V grouping is worth only 1.5%

`lfm2moe.attention.head_count_kv` is `[0,0,8,0,0,0,8,0,...]` over `block_count = 24`: a zero marks a
convolution block, so only about six blocks have attention at all. Grouping three calls into one in
a quarter of the layers cannot be worth more than a few percent, and measured 1.5%. It is kept
because it is strictly fewer kernel calls, not because it is a lever.

## What each change was

Both are existing implementations this decoder was never connected to -- the same shape of omission
as the missing kernel seam:

- `Lfm2ForwardPass.attend` hand-rolled scalar attention (a dot loop and a weighted-value loop over
  `headDim`, per head per past position) while `GroupedQueryAttentionKernel` in the same package,
  already used by the Llama path, provides `scoreGroup`, `softmax` and `accumulateGroup` vectorised
  with pinned host-independent reductions.
- All three activation sites called `TensorOps.swiGlu`, a scalar loop invoking `Math.exp` per
  element, while the Rust kernel's `trySwiGlu` and `VectorUtil.swiGlu` were both reachable.

Neither is bit-identical to what it replaced: the score reduction folds in a different order and the
vectorised swiGLU evaluates `exp` by polynomial. The Llama path gates the same swiGLU choice behind
`usesVectorizedSwiGlu()`, which admits only QWEN2 and QWEN3 -- an **output-parity** whitelist rather
than a performance one, since each architecture there keeps the activation its published records
were measured with. lfm2 is under no such constraint: the catalog carries no lfm2 architecture,
checked against all 80 qualified entries rather than assumed.

## Standing assessment

Decode is the binding constraint at 0.4674 against a 0.80 floor, and decode cannot be helped by
batching since it is one token at a time. Reaching the floor needs the quantised GEMV path roughly
1.7x faster while it already implements ggml's algorithm. One prior attempt at that layer -- the
repack-tile layout, worth 1.49-1.87x inside llama.cpp -- measured at parity in our port and was
published as a null. Treat further work here as a kernel project with uncertain payoff, not as
wiring.
