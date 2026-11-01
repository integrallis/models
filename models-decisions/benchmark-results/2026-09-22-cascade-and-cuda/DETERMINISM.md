# The measured path was not reproducible, and it was JIT warmup. Measured 2026-09-22.

Found while checking whether a CUDA kernel change altered numerics. It did not; the check was
wrong. The path itself was nondeterministic, so the comparison was measuring noise.

Host: RunPod A40, Qwen3.5-4B Q4_K_M, letter-logit mode, prefillBatchSize=512, maxContextLength=4096.

## The observation

Two runs of the *same build* on the *same input* disagree:

```
CUDA,  run A vs run B:  track_order 0.9385  vs 0.9176
CPU rust-ffm, C vs D:   track_order 0.9284  vs 0.9236
```

Both paths. So it is not the CUDA kernel.

## Isolation

| hypothesis | test | result |
| --- | --- | --- |
| parallel GGUF row reduction | `-Dvectors.gguf.parallel=false`, 2 runs | still nondeterministic |
| JIT-dependent FMA fusion | `-Dvectors.useVectorFMA=false -Dvectors.useScalarFMA=false`, 2 runs | still nondeterministic |
| JIT warmup | same item scored 5x in one JVM, twice | **confirmed** |

An early attempt used `-Dvectors.ggufParallel=false`, which is not a real property name; the log
still read `ggufParallel=true` and that run proved nothing. The real name is
`vectors.gguf.parallel`. A toggle that silently does nothing produces exactly the "this stage
does not matter" reading the working agreement warns about, so the toggle is now asserted from
the log before the result is read.

## The evidence

The same item, scored five times in one process, in two independent JVMs:

```
process K:  0.9370657952416532  0.9387301883135959  0.9316622781200229  0.9316622781200229  0.9316622781200229
process L:  0.9253396379161376  0.9398595450585500  0.9316622781200229  0.9316622781200229  0.9316622781200229
```

It converges by the third repetition to a value that is bit-identical across processes. A race
does not converge to the same bits in a second JVM; partly-interpreted code that later compiles
does. With six warmup items prepended, all five real items are bit-identical across two
independent JVMs.

## Consequences

* **Every 5-item diagnostic in this project had roughly 40% of its items scored cold.** The
  old-vs-grid-Y kernel comparison is the one that was invalidated: re-checked on the two items
  that were warm in both runs, the two kernels agree bit-for-bit on all ten probabilities.
* A 231-item run has about two contaminated items, under 1%, so Intelligence numbers are
  essentially unaffected. **Latency is affected more**: cold items are the slowest, and JevBench's
  Speed axis is computed from p50 and p95.
* `qwen4b-letters.tsv` (231 items, Hetzner) was launched before this fix and has no warmup. Its
  first two rows are cold.

## Fix

`JevBenchRunner` now scores the first item `-Ddecisions.warmupItems` times (default 3) and
discards the results, restarting both the tallies and the wall clock afterwards.

## Open, not investigated

The CUDA and CPU paths disagree with each other even when both are warm, stably. The GPU path
quantises activations to Q8_K before each projection (`Q8KActivations`); whether the CPU path
does the same thing bit-for-bit has not been checked. This is pre-existing and is not caused by
the grid-Y change. Recorded as not done, not as cannot be done.
