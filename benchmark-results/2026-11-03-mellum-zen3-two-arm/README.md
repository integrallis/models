# mellum on Zen 3: qualifies at 8 vCPU, fails at 16. Both measured, both reported.

**Measured here**, 2026-11-03, on two Zen 3 AVX2-only boxes. Nothing in this directory is
projected or extrapolated. The protocol was frozen in `PROTOCOL.md` before any data was collected.

## Why this exists

mellum's qualification was published (ModelJars/modeljars#258, marker live on Maven Central) on the
strength of runs taken on an **AMD EPYC 9754, which carries `avx512f`**. 78 of this catalog's 82
qualification entries were measured on **Zen 3 EPYC without AVX-512** at 8 or 16 vCPU. On that
hardware class mellum had exactly one measurement: 0.763, FAIL, from before the Q4_K unpack fix.

`0.763 x 1.181 = 0.901` was cited four times as the reason to believe it now passed. That is
arithmetic across two runs on different boxes, not a measurement. Both numbers below replace it.

## Result

| host | vCPU | threads | ours | ollama | decodeRatio | verdict |
|---|---:|---:|---:|---:|---:|---|
| EPYC-Milan (dedicated, idle) | 8 | 8 | 29.76 | 34.27 | **0.8683** (sd 0.0030) | **QUALIFIED, 3/3** |
| EPYC 7713 (shared, host load ~179) | 16 | 16 | 29.88 | 37.95 | **0.7872** (sd 0.0041) | **FAILED_RELATIVE_GATE, 0/3** |

Both are three complete paired runs, candidate then comparator in one session per pair, same
workload (`coding`), template (`chatml-no-think`), context 2048, max-tokens 256, `COMPARATOR=ollama`,
`--artifact` passed. Contribution passed in all six runs: `answerRate` 0.7778, `correctRate` 1.0000.

**The published entry is supported at 8 vCPU on a reference CPU string, and is not supported at 16
threads on a shared 7713.** Neither number is the whole story and quoting either alone would
misrepresent it.

## The mechanism: we do not scale past 8 threads

| | ours | ollama |
|---|---:|---:|
| 8 vCPU | 29.76 | 34.27 |
| 16 vCPU | 29.88 | 37.95 |
| 8 -> 16 | **+0.4%** | **+10.7%** |

Our decode throughput is flat from 8 to 16 threads while the comparator gains 10.7%. That, not the
kernel, is what the 16-thread failure is made of. **This is not a controlled comparison** -- two
different CPU models and one shared host -- so it is a lead, not a finding. The controlled version
is an 8-vs-16 sweep on one box, which this account's dedicated-core quota (8) cannot run.

## The kernel: isolated gain is real, end-to-end gain is not resolved

Our Q4_K row dot against ggml's own `ggml_vec_dot_q4_K_q8_K`, in one process, single-threaded,
outputs asserted equal before any timing was believed (llama.cpp `2cbca389`, `kernel-ab/`):

| variant | 36 MiB | 9 MiB | vs ggml output |
|---|---:|---:|---|
| shipped baseline | 0.9405 | 0.9190 | 7.8e-09 |
| `mins` (vector min accumulator) | +1.29% | +1.63% | **0.000e+00** |
| `combo` (mins + `vpshufb` scale + `#[inline]`) | **+1.54%** | **+2.33%** | **0.000e+00** |
| `chain` (fold pair before accumulating) | +1.37% | +2.11% | null vs combo |

n=8 per arm per size, arms alternated, ggml in-process as the drift control. Both winning variants
make our Q4_K output **bit-identical to ggml**, which is a correctness gain independent of speed.

End-to-end, the same `combo` library measured **0.8744 (sd 0.0105)** against the baseline's 0.8683
(sd 0.0030) -- +0.7% on the mean with **overlapping ranges** (0.8623-0.8816 against 0.8664-0.8717).
At three passes that is **not resolved**, and it is not claimed. A single first pass read 0.8816 and
would have supported a "+1.7% end-to-end" claim; the other two passes do not, which is exactly the
single-run inference this model of error keeps producing.

`combo` therefore does **not** ship from this record. It also cannot ship as written: it breaks
`q4_k_avx2_preserves_single_row_rounding_across_batch_sizes`, because the batched kernel computes
the minimum correction in eight lanes (`mullo_epi32` over `setr_epi32`) while the row kernel uses
four (`madd_epi16` over hadd'd sums). Vectorising both yields different float groupings, so making
them bit-identical requires unifying the lane layout across two kernels. That is a deliberate piece
of work, not a rushed one.

## Nulls, published

- **`vpshufb` scale broadcast alone**: null on a laptop, worth ~0.7% on a quiet box. The earlier
  `perf/q4k-scale-shuffle` null was measured where the noise floor exceeded the effect. Laptop
  ggml spread was 10% against 0.7% on the dedicated box; the laptop could not have resolved it.
- **Folding the pair before accumulating** (`chain`): null, slightly negative against `combo`.
  LLVM was already reassociating those adds. The reasoning that our Q5_K does it this way and is at
  parity was wrong about the cause.
- **`record_k_quant_weight_block_decode()` in the hot loop**: not a cost. It is `#[cfg(test)]`-gated
  and compiles to nothing in release. Checked rather than assumed.
- **Hardware counters on Hetzner**: `perf` installs and `perf_event_paranoid` can be lowered, but
  the KVM guest exposes no PMU, so `perf stat` returns no cycle or instruction counts. Sampling
  would only re-confirm the time is in the row dot.

## What remains unmeasured

- 8-vs-16 threads on one box, which is the controlled form of the scaling lead above.
- `EPYC 7R13` specifically, and any ARM host.
- Whether the comparator's own AVX-512 tier explains part of the EPYC 9754 result: on that box
  ollama had AVX-512 kernels and we have none, and we still passed decode there.

## Reproducing

`kernel-ab/kab-linux.sh` builds ggml at the pinned revision and runs the head-to-head.
`milan-8vcpu/` holds every raw candidate, comparator and verdict JSON for the six paired runs;
`HOST.txt` records both library sha256s, the model sha256 (matching the catalog entry) and the
source revision.
