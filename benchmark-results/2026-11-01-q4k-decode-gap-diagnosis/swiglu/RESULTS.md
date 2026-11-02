# SwiGLU: the second lever, and which half of the catalogue each fix reaches

Box: i7-9750H (AVX2), 6 cores / 12 threads. **The box was 11-15 GB into swap for part of this
session**, so absolute milliseconds drift by ~6% between repeats; every comparison below is
same-process or same-session A/B, and the exactness results are digests, which load cannot affect.

Shape throughout: one mathstral-7B token's FFN activations, 32 layers x 14336 elements.

## The routing, and who lands where

`LlamaForwardPass` takes four branches, only one vectorised, gated to two architectures:

    usesGeluFfn()            -> geluGlu                              Gemma family
    usesGraniteScaling()     -> GroupedQueryAttentionKernel.swiGlu   Granite
    usesVectorizedSwiGlu()   -> trySwiGlu / VectorUtil.swiGlu        QWEN2/QWEN3 only
    else                     -> TensorOps.swiGlu                     everyone else, incl. LLAMA

So **improving the vectorised branch does not help llama-arch models at all** -- they never enter
it. mathstral and mellum run the scalar branch, which is what JFR caught at 42.8% of Java time.

Counting `architecture` fields in `catalog/*.json` (entries, not models -- a model may appear under
several capabilities, and `qwen35`/`lfm2`/`gpt-oss`/`gemma4` belong to other forward passes):
~152 entries on the vectorised branch (qwen2 113, qwen3 39) against ~156 on the scalar branch
(llama 123, phi3 20, hunyuan-dense 5, mistral3 3, qwen3moe 3, smollm3 2). `qwen3moe` is on the
scalar side despite the name.

## The size of the prize

| route | ms/token |
|---|---|
| `TensorOps.swiGlu` (scalar `Math.exp`) | 12.5 - 13.7 |
| `VectorUtil.swiGlu` (Panama, single thread) | **1.613** |

Against the 111 ms/token measured for mathstral-7B decode on the EPYC pod, the scalar route is
**~11.3% of decode** -- which independently reproduces the 10.3% that JFR attributed to it. Two
different instruments agreeing on this share is the reason it is worth fixing.

## What was fixed now: a bit-exact parallel split

SwiGLU is elementwise with no reduction, so splitting the range **cannot** change any lane's
arithmetic. That is a construction argument, not a measurement -- but it was verified anyway, by
digesting the entire output array rather than sampling lanes:

| workers (threshold) | ms/token | SHA-256 of output[0:8] |
|---|---|---|
| 1 (serial) | 13.709 | `7eaabfc6174081c0` |
| 3 (4096) | 10.320 | `7eaabfc6174081c0` |
| 7 (2048) | 8.845 | `7eaabfc6174081c0` |
| 12 (1024) | **7.352** | `7eaabfc6174081c0` |
| 12 (512) | 7.352 | `7eaabfc6174081c0` |

**1.86x, byte-identical output at every worker count.** Default threshold set to 1024, the
saturation point, overridable via `models.purejava.swiGluParallelThreshold` so the crossover can be
re-measured without a rebuild. It must be re-measured on a quiet box: this one was in swap.

Nested fan-out is prevented by a thread-local guard, so a chunk running on a pool thread never
splits again. `:backend-java:test` passes for `*TensorOps*`, `*SwiGlu*`, `*LlamaForwardPass*` and
`*PureJavaBackendTest*`, including the `InjectedSwiGlu` suite.

## Why the arithmetic was left alone

Two prior experiments attacked the per-element arithmetic to get an exact faster exp. Both were
**rejected**, and neither touched parallelism -- which is precisely why parallelism was the lever
still available:

- `audit-artifacts/swiglu-table-exp`: candidate/current warm ratios **2.068 / 2.616 / 2.680** at
  512 / 14336 / 32768 elements, i.e. 2-2.7x **slower**.
- `audit-artifacts/swiglu-native-fdlibm`: warm latency **4.44-4.55x worse** than published on x86.

## The decision NOT taken

Widening `usesVectorizedSwiGlu()` to llama-arch would buy the full 7.75x rather than 1.86x, but it
is **not numerically free**: Panama and scalar `Math.exp` disagree on **586 of 14336 lanes**, max
absolute 9.537e-07, max relative 2.206e-07 (~1-2 ULP of float). That changes every llama-arch
model's output and is a re-qualification question, not a perf change, so it is left to the owner.
Filed as integrallis/models#331 together with the vectorised-branch work.

No evidence llama was ever tested and excluded: `git log -S` shows the gate was *added* by
a3147589 ("perf(qwen): retain vectorized SwiGLU execution", 2026-08-30); before it, everything
including Qwen used the scalar path. It was built for Qwen and never extended.

## Reproduce

    # prices the two routes and their divergence
    java --add-modules jdk.incubator.vector -cp <dist>/lib/'*':. SwiGluProbe
    # serial vs parallel, with an output digest; vary the threshold
    java -Dmodels.purejava.swiGluParallelThreshold=<n> ... SwiGluParallelProbe
