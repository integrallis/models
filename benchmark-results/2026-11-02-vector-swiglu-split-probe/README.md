# Splitting the vectorised SwiGLU across workers: a null (issue #331)

**Do not parallelise `VectorUtil.swiGlu`.** On a quiet dedicated machine the pool hand-off costs more
than the whole activation at every FFN width a catalogued model has.

## Measurement

Hetzner ccx33 (8 dedicated AMD EPYC vCPU, AVX2, 256-bit Panama), load average 0.00, JDK 25.0.3,
2026-11-02. `SwiGluSplitProbe.java` runs the **published** `com.integrallis:vectors-core:0.1.32`
(the version Models pins) through JBang, nothing from this repository: serial `VectorUtil.swiGlu`
against the same call split the way `TensorOps.swiGlu` splits the scalar path (a fixed pool of
cpus - 1 threads, the caller taking chunk 0, chunks aligned to 64 floats). Best of 9 timed blocks
after warm-up. Raw output: `output-ccx33.txt`.

| FFN width | serial | split, 2 workers | split, 8 workers |
| --- | ---: | ---: | ---: |
| 1,024 | 0.86 us | 39.82 us | 41.39 us |
| 14,336 (mathstral 7B) | 11.87 us | 45.27 us | 63.49 us |
| 18,944 (Qwen2.5 7B) | 15.22 us | 49.69 us | 49.36 us |
| 65,536 | 54.27 us | 97.27 us | 94.34 us |

The vector kernel costs about 0.8 ns per element. A pool round trip costs 30 to 40 us, so the split
loses by 1.7x to 46x across the sweep. The scalar `TensorOps.swiGlu` split (1.86x, #333) works
because the scalar exponential costs about ten times more per element, which puts the work above the
hand-off cost; the vectorised path never gets there.

## What this does not say

- It measures the hand-off of an idle pool. Inside a decode step the projection workers have just
  run, so a hand-off may be cheaper there; even at a tenth of the cost (3 to 4 us) the gain at
  14,336 would be under 8 us per call. Not worth a second code path without a model-level measure.
- An earlier probe on the i7-9750H dev machine (load average 90+) showed the same direction (0.6x
  to 1.1x) and is not used: the machine was saturated.

## Exactness, for the record

The split was implemented and checked before it was timed: the SHA-256 of the whole output equals
one serial call for one to twelve workers and unaligned offsets. In Vectors 0.1.32 the kernel's
scalar tail reproduces its vector lanes exactly, so the result held even with the 64-float
alignment removed. The code was discarded after this measurement; `SwiGluSplitTest` keeps the
whole-array digest check for the scalar split, which ships and had no such test.
