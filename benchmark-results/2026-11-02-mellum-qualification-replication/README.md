# mellum replication: decode holds, end-to-end latency does not. 1 of 3 runs fails the gate.

**The qualification entry in ModelJars/modeljars#258 was landed on a single paired run. This is the
replication it should have had first, and it does not come back clean.**

Measured 2026-11-02, pod `ayx914h735s2dm`, 16 vCPU of an **AMD EPYC 9754** (avx2 and avx512f on the
host), workload `coding`, template `chatml-no-think`, top-k 1, `--warmups 1 --iterations 3`,
`--threads 16`, ollama 0.40.13 on the identical artifact, three complete paired runs sequentially in
one session each. Protocol frozen in `PROTOCOL.md` before any data. No run discarded.

| run | ours | ollama | `decodeThroughputRatio` | `endToEndLatencyRatio` | verdict |
|---|---:|---:|---:|---:|---|
| 1 | 34.61 | 35.85 | 0.9654 | 1.0638 | QUALIFIED |
| 2 | 32.64 | 38.65 | 0.8445 | **2.1365** | **FAILED_RELATIVE_GATE** |
| 3 | 33.94 | 38.65 | 0.8782 | 1.0086 | QUALIFIED |
| **mean** | **33.73 ± 1.01** | **37.72 ± 1.61** | **0.8960 ± 0.0624** | | **2 of 3** |

Floors: decode >= 0.80, end-to-end latency <= 1.50. Contribution passed in every run at
`modelAnswerRate` 0.7778 and `modelAnswerCorrectRate` 1.0000.

## Decode replicates. The frozen decision rule was the wrong rule.

`PROTOCOL.md` fixed one criterion in advance: mean `decodeThroughputRatio` >= 0.80 earns the entry.
By that criterion it is earned -- 0.8960, with **all three** runs above the floor and the lowest
still 5.6% clear.

**That rule was under-specified and is not being used to claim a pass.** It guarded only the gate
the kernel work had been chasing. Run 2 failed on `endToEndLatencyRatio` at 2.1365 against a 1.50
ceiling, a gate the protocol never mentioned. A qualification requires both, so the honest reading
is **2 of 3**, not 3 of 3.

## Why run 2 failed, measured rather than guessed

| run | our ttft p95 | ollama ttft p95 | our e2e p95 | ollama e2e p95 |
|---|---:|---:|---:|---:|
| 1 | 4348.8 | 1776.5 | 5732.6 | 5388.6 |
| 2 | **9718.9** | 3158.9 | **11392.3** | 5332.2 |
| 3 | 4325.7 | 1733.3 | 5470.4 | 5423.9 |

Output tokens were **identical in all three runs** -- 825 ours against 1131 ollama's -- so the token
asymmetry that makes this ratio awkward to read is constant and is not the cause. Run 2 took a
time-to-first-token spike: our p95 doubled.

The reason is structural. **Our prefill is 57.54 tok/s against ollama's 107.80, about 0.53x**, so our
TTFT already sits around 2.4x theirs (4348 against 1776 in the clean runs). `endToEndLatencyRatio`
is a p95 measure and normally lands near 1.0 because decode dominates the total; with prefill at half
the comparator's rate there is no headroom, and one contended prefill-heavy case carries the p95
past the ceiling.

## What this says about the port

- **Decode is fixed and robust.** The Q4_K per-block unpack (models `dcd8b172`) is replicated here
  at a 0.8960 mean across three runs and two machines. Against the pre-fix replicated baseline of
  0.7630 in `2026-11-01-mellum-replicated-verdict`, that is the lever working.
- **Prefill is the unfixed gate**, and it is what makes the qualification fragile rather than
  comfortable. Across all four paired runs ever measured with the fix, `endToEndLatencyRatio` reads
  0.7895, 1.0638, 2.1365, 1.0086 -- one breach in four.
- The host swing is large and symmetric: our decode moved 44.99 -> 32.64 across boxes while the
  comparator moved 51.89 -> 38.65, which is why only the ratio is comparable. Our arm proved **less**
  sensitive to neighbour load than the comparator's in run 1.

## Correction to earlier records

Records in `2026-11-02-mellum-q4k-kernel-verification` quote mellum's pre-fix ratio as 0.6422, taken
from `2026-11-01-mellum-shim-optimisation`. The replicated pre-fix figure is **0.7630**
(`2026-11-01-mellum-replicated-verdict`, three runs), because fused grouped attention landed between
the two. The Q4_K fix therefore closed roughly 4.8%, not 24.6%.

## Also a null, from the same session

The last ggml technique not yet ported -- broadcasting the six-bit scales with a register
`vpshufb` from a `get_scale_shuffle_k4`-style table instead of `set1_epi16` from a general register
-- was implemented and **measures as nothing**: six interleaved pairs against ggml in-process,
3 wins each, with identical binaries swinging 4.44 to 7.24 GB/s on the measuring host. The general
register was not the bottleneck. Correct (48 crate tests pass) but not shipped, on branch
`perf/q4k-scale-shuffle`.

## Files

`rep1/`, `rep2/`, `rep3/` hold each paired run's candidate, comparator, verdict and verdict log.
`HOST.txt` carries the CPU, core count, ISA, the kernel digest `0c6127f1...bdd474` -- byte-identical
to the library the qualification and the correctness smoke were measured with -- the artifact digest
`489cf0d7...3ccb` and the ollama version. `reprun.sh` ran the three pairs; `summ.py` and `e2e.py`
derive the tables above.
