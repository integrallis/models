# Flaky `injectedAttentionKernelPreservesPrefillAndFallsBackForDecode`: root cause

Date: 2026-09-16. Branch `wave2/flaky-test`, based on `origin/main` at `167a8abd` (Models 0.3.41).

Everything below was **measured here** unless it is marked *read in source* or *believed*.

## The failure

`LlamaForwardPassTest > NanoModel > injectedAttentionKernelPreservesPrefillAndFallsBackForDecode()`
failed once in CI run 35164869486, attempt 1 (job 105023594106, Linux x86_64 GitHub runner) and passed
on the rerun. The failing line was `assertThat(actual).containsExactly(expected)` on the prefill
logits (`LlamaForwardPassTest.java:874`). The two arrays differed only in the last bit of some
elements, for example `-0.10874769f` actual against `-0.10874768f` expected.

## Configuration

| | |
|---|---|
| Host | MacBook Pro, Intel Core i7-9750H (AVX2), macOS Darwin 25.6.0, x86_64 |
| JVM | Temurin 25.0.3+9, `--add-modules jdk.incubator.vector` |
| vectors-core | 0.1.22 (256-bit species, persistent executor, 12 threads by default) |
| Harness | `harness/Repeat.java` runs one JUnit method N times in one JVM through the JUnit Platform Launcher. `harness/Drift.java` builds the test's Q4 nano model and runs a fresh `prefill` N times, comparing each result with the first (cold) one. `harness/Reduce.java` calls `FloatVector.reduceLanes(ADD)` on fixed inputs until the result changes. |
| Classpath | the `backend-java` test runtime classpath (classes directories plus the Gradle cache jars) |

## Reproduction

| Configuration | Runs | Failing |
|---|---|---|
| fresh JVM per run, 1 test execution, default tiered JIT | 40 | 0 |
| fresh JVM, `-XX:TieredStopAtLevel=1` | 40 | 0 |
| fresh JVM, `-Xint` | 40 | 0 |
| fresh JVM, `-Dvectors.gguf.threads=1` | 40 | 0 |
| fresh JVM, `-Dvectors.maxBits=128` | 40 | 0 |
| one JVM, 200 consecutive executions, default | 1 JVM x 200 | 0 |
| one JVM, 1,500 consecutive executions, default (background compilation) | 3 JVMs x 1,500 | 0 |
| **one JVM, 1,500 consecutive executions, `-XX:-BackgroundCompilation`** | **3 JVMs x 1,500** | **exactly 1 in each of the 3 JVMs** |

With synchronous compilation, each of the 3 JVMs failed exactly once. The **actual values printed
were identical to the CI log** (`-0.07185684f, -0.10874769f, -0.027812036f, ...`), so the local
failure and the CI failure are the same numbers (`repro-in-jvm-before.txt`).

Each JVM fails once and only once. That is the signature of a one-time state transition. Racy
parallelism, uninitialised scratch memory, or shared static state would fail repeatedly or at
random.

## Root cause

1. **The same prefill changes its low bits once C2 compiles one kernel (measured).** `Drift`, 3,000
   fresh prefills of the Q4 nano model in one JVM:
   - Default tiered JIT: 2,264 of 2,999 results differ from the cold result. The first difference
     is at iteration 736, and the maximum absolute difference is 2.98e-8.
   - `-XX:TieredStopAtLevel=1`: 0 differ.
   - Default JIT with `-XX:CompileCommand=exclude,com.integrallis.vectors.core.PanamaVectorUtilSupport::ggufF32MatVecDot`: 0 of 1,499 differ.

   With `-XX:+PrintCompilation` (`drift-printcomp-excerpt.txt`), C2 started the tier-4 compile of
   `PanamaVectorUtilSupport::ggufF32MatVecDot` at 22,029 ms. Its tier-3 version was made not entrant
   at 22,224 ms, and the first differing prefill was observed at 22,411 ms.
2. **The kernel's float reduction has no fixed order across tiers (read in source, then measured).**
   `ggufF32MatVecDot` in vectors-core 0.1.22 sums its FMA accumulators with
   `acc.reduceLanes(VectorOperators.ADD)`. `Reduce` shows that on this JVM the call returns
   `317.06598` (the sequential lane sum) before C2 compiles it and `317.06595` after call 18,468. Under
   `-XX:TieredStopAtLevel=1` it never changes. The Vector API documentation says a floating-point
   `ADD` reduction may be computed in any order (*believed*, recalled from the javadoc and not re-read
   here). vectors-core already has an explicit-order helper, `reduceAdd`, which its heap
   `dotProduct` uses, but this kernel does not use it.
3. **How that becomes a flaky test (measured mechanism; the CI timing is inferred).** In the nano
   model, `output.weight` is F32. `TensorOps.ggufMatmul` sent F32 projections to
   `VectorUtil.ggufF32BatchDotProduct`, which calls that kernel. The test computes `expected` with
   the baseline pass and then `actual` with the accelerated pass. If the tier-4 compile of the
   kernel is installed between those two logits heads, the two computations use different reduction
   orders. In a long CI test JVM the compile lands at an arbitrary point, and occasionally it lands
   inside this window. `-XX:-BackgroundCompilation` pins the switch to the invocation that triggers
   it, which is why it reproduces deterministically once per JVM.

**Ruled out, as far as these runs can rule anything out:**
- Attention itself. The test's reference kernel and the production path use the same order-defined
  `VectorUtil.dotProduct`, `TensorOps.softmax` and `addScaledInPlace`, and excluding only the F32
  GEMV removed all drift.
- Executor thread count (`threads=1`: 0/40).
- Species width (`maxBits=128`: 0/40).
- Uninitialised scratch and test ordering (one failure per JVM, never more).

The Q4_0 projections are integer dots scaled by FMA. They were compiled by C2 before the first
drift and caused none.

**The comparison itself is correct.** Two passes over the same weights and inputs should agree bit
for bit. The defect was in the kernel's arithmetic.

## Fix

Test first:
- `TensorOpsTest.Matmul.ggufF32MatrixIsBitIdenticalToTheOrderDefinedRowDotProduct`: a 7 x 67 F32
  projection (a 4-row group plus tail rows, and a vector body plus a scalar tail) must be
  bit-identical to `VectorUtil.dotProduct` row by row.
- Red: 3 of 7 rows differ (`evidence-f32-contract-red.txt`).
- Green after the fix (`evidence-f32-contract-green.txt`).

`TensorOps.ggufMatmul`'s F32 case now copies each mapped weight row into one reused heap row and
scores it with `VectorUtil.dotProduct`. That is the pattern `ggufExactBatchedMatmul` already uses.
The arithmetic now has one order in every tier.

After the fix (`repro-in-jvm-after.txt`):

| Configuration | Result |
|---|---|
| `-XX:-BackgroundCompilation`, 3 JVMs x 1,500 executions | 0 failures (before the fix: 1 per JVM) |
| `Drift` default JIT, 3,000 prefills | 0 differ from cold (before the fix: 2,264) |
| `:backend-java:test` | 671 tests, 0 failures, 18 skipped |
| `msMarcoMiniLmRerankerIntegrationTest`, `msMarcoMiniLmL12RerankerIntegrationTest` on the local fixtures | pass (these models have an F32 `classifier.dense.weight`) |

## What this does and does not show, and what is left

- **Reach:** a scan of the 36 local GGUFs found 2-D F32 tensors only in the MS MARCO classifiers
  (384x384), Qwen3.5 `ssm_conv1d`, and BERT token-type or position tables. None of the pinned LLM
  fixtures has an F32 projection, so no greedy oracle could move.
- **Speed:** not measured. The copy adds a `cols`-float memcpy per row. On the local models this
  path carries at most a 384x384 classifier.
- **Upstream, still open (read in source, drift not measured):** the vectors-core 0.1.22 float
  kernels that still reduce with `reduceLanes(ADD)` are:
  - `matVecDot` (heap GEMV behind `VectorUtil.batchDotProduct`, which `TensorOps.matmul` calls)
  - `matVecSquaredL2`
  - `dotProduct(MemorySegment, MemorySegment, int)` and `dotArraySeg`
  - `sum`
  - the `cosine`/`batchCosine` family
  - `squareDistanceBody`

  Each can give a JIT-state-dependent result in the same way. The real fix belongs in vectors-core:
  replace float `reduceLanes(ADD)` with the explicit-order `reduceAdd`. Once a release does that,
  the Models F32 path could return to the fused kernel.
- CI's exact compile timing cannot be observed after the fact. That the CI failure had this cause
  rests on the identical failing values and the once-per-JVM reproduction, not on a CI compile log.
