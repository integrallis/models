# mellum QUALIFIES: the Q4_K per-block unpack is worth +18.1% decode and carries the gate

**Verdict: `QUALIFIED`** under `production-rag-model-contribution-v6`, `exclusions: {}`.
Protocol frozen in `PROTOCOL.md` before any number was read.

Measured 2026-11-02, pod `3wq3hkfosq31gv`, 16 vCPU of an **AMD EPYC 9654** (AVX-512 present on the
host), ollama 0.40.13, workload `coding`, template `chatml-no-think`, top-k 1,
`--warmups 1 --iterations 3 --threads 16`, context 2048, max-tokens 256. Both arm sets ran
**sequentially in one session on one box**, differing only in which `libjmodels_kernels.so` was
loaded.

| arm set | candidate decode | candidate prefill | ollama decode | decodeRatio | e2eRatio | ttftRatio | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| `main-33700300` | 38.10 | 52.55 | 51.81 | 0.7355 | 0.5430 | 2.4048 | `FAILED_RELATIVE_GATE` |
| **`q4kfix-760f7776`** | **44.99** | **68.35** | 51.89 | **0.8669** | **0.7895** | **1.4873** | **`QUALIFIED`** |
| delta | **+18.1%** | **+30.1%** | +0.15% | +0.1314 | | | |

Gates for the passing arm: `decodeThroughputRatio` 0.8669 >= 0.80; `endToEndLatencyRatio` 0.7895
<= 1.50; `modelAnswerRate` 0.7778 >= 1/3; `modelAnswerCorrectRate` 1.0000 >= 0.90.
`absoluteTier` `PERFORMANCE_REDUCED`. 21 of 27 cases answered by the model, all 21 correct.

## Why the delta is attributable to the change

- **The comparator did not move.** ollama's decode was 51.81 and 51.89 tok/s across the two arm
  sets -- **0.15% apart** -- and prefill 135.87 against 135.72. The 2026-11-01 record warned that
  ollama's mellum decode on its box spanned 33.93 / 39.42 / 37.17 (about +/-8%), which is why that
  caution was carried into this protocol. It did not materialise here, so the entire ratio change
  came from our arm rather than from the comparator drifting.
- **One variable.** Same box, same session, same artifact
  (`489cf0d7ca86ef4683e34e2efe8a46a9b52573bfe994a6ad9c86bf57c7173ccb`, byte-identical to the
  2026-11-01 runs), same workload, template, thread count and protocol. The only difference is the
  shared library: `d9fa85fe...` (main) against `0c6127f1...` (fixed), confirmed distinct by sha256
  before launch because `kernel-implementation` reads `rust-ffm-quantized-v14` for both.
- **The route was recorded, not inferred.** Both candidate reports carry
  `native-quantized-decode = true`, `native-kernel-decode-threads = 16`, `kernel-runtime = rust-ffm`.
- **Admissible.** `artifactSha256` is non-null and matches the comparator's own layer, so the
  comparison is not excluded; `exclusions: {}` on both verdicts.

## What the change was

`qk_unpack_scales_and_mins` unpacks all eight six-bit scales and all eight mins of a K-quant block
once, from three u32 words, instead of calling the branchy per-group extractors sixteen times per
256-element block (eight for scales in the pair loop, eight more in a scalar eight-iteration mins
loop). The mins term becomes `hadd_epi16` + `madd_epi16` over the activation group sums. Bit layout
and masks are ggml's (MIT), attributed in the source.

Isolated kernel measurement on an i7-9750H against ggml in-process as the control: Q4_K
**5.64 -> 9.49 GB/s (1.68x)**, reaching 0.905 of ggml's own row dot. Q5_K was a null (1.03x) because
main had already improved it.

**Prefill gained more than decode** (+30.1% against +18.1%), which was not predicted: the per-block
overhead is paid per weight block per row regardless of batch, so a batched prefill pays it across
every batch row. `timeToFirstTokenRatio` improved 2.4048 -> 1.4873 as a result.

## Scope, honestly

- The two earlier levers are **already on main** and are in both arm sets here: the vectorised
  SwiGLU admission for `MELLUM` and the Q5_0 independent dispatch
  (`benchmark-results/2026-11-01-mellum-shim-optimisation`). This result sits **on top of** them,
  it does not replace them.
- **This box is not the box the 2026-11-01 ladder used.** That was an EPYC 7713 without AVX-512 and
  ended at decodeRatio 0.6422; this is an EPYC 9654 whose host reports AVX-512, and main's own
  baseline here measures 0.7355. The two baselines are not comparable, which is exactly why arm
  set 1 re-measured it. Our kernels carry no AVX-512 tier while llama.cpp/ollama do, so if anything
  this box favours the comparator. US-KS-2 had no capacity for matching hardware.
- `timeToFirstTokenRatio` is 1.4873 and is **not** a gate in this policy; it is recorded, not
  claimed as a pass.
- A single pair of arm sets. The comparator's stability here is strong evidence, but this is one
  session; a confirming run on a second box would make the +18.1% a replication rather than a
  measurement.

## Files

`arm-main/` and `arm-fixed/` hold each arm set's `candidate.json`, `comparator.json`,
`verdict.json`, per-case logs and the ollama `Modelfile`. `KERNEL-SHA256.txt` has both library
digests, the artifact digest, the ollama version and the CPU. `podsetup.sh`, `podbuild.sh` and
`podrun.sh` are the exact scripts that produced it; `podrun.log` is the session transcript.
