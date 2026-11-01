# mellum: two-arm qualification — 2026-10-31

**Verdict: `FAILED_RELATIVE_GATE`.** The contribution gate passes and matches the reference exactly.
Decode is **2.65× slower** than the reference, which is what keeps it out of the catalog.

## The rope defect this run fixed

mellum was registered in `DecoderArchitecture` without an entry in `usesNeoxRope()`, so it served
**NORM** rope where `llama_model_rope_type` returns **NEOX**. NORM-for-NEOX rotates the right angles
onto the wrong element pairs: exact at position 0 where there is no rotation, degrading as positions
grow, and nothing throws.

| | before the fix | after |
|---|---|---|
| `modelAnswerRate` | **0.0000** (0/9) | **0.7778** (7/9) |
| `modelAnswerCorrectRate` | 0.0000 | **1.0000** (7/7) |
| raw generation | 256 newline tokens, EOS never reached | correct cited answers |

`candidate-before-rope-fix.json` is the pre-fix arm, kept because the contrast is the evidence.

`usesNeoxRope()` is now an exhaustive switch with no default, so a newly registered architecture is
a compile error until its rotary layout is read from the reference table.

## Gates

Two arms, sequential, one host. `sameWorkload()` matched on all eight compared fields including
threads. Workload `coding` — matched to the model's `code-completion` capability per the campaign
runbook — template `chatml-no-think`, topK 1, 256 output tokens, context 2048, 16 threads.

| | candidate | comparator | gate | |
|---|---:|---:|---|---|
| `modelAnswerRate` | **0.7778** | 0.7778 | ≥ 0.3333 | **PASS** |
| `modelAnswerCorrectRate` | **1.0000** | 1.0000 | ≥ 0.90 | **PASS** |
| p50 decode tok/s | **11.34** | **30.06** | | |
| `decodeThroughputRatio` | | | ≥ 0.80 | **0.3771 — FAIL** |
| e2e p95 ms | **81 106** | **10 630** | | |
| `endToEndLatencyRatio` | | | ≤ 1.50 | **7.63 — FAIL** |

9 attempts per arm, 9 successful, zero runtime failures either side. Both arms answer the same 7 of 9
and are correct on all 7, so the model and our decoder now agree on content.

Host: RunPod `cpu3c`, 16 vCPU AMD EPYC 7713, AVX2, no AVX-512. Candidate on `rust-ffm` with the Rust
kernel built from this tree on the pod; comparator `llama-server` at
`ggml-org/llama.cpp@3f03b17724ad8b69040825f45c43c59b092a0244`.

## The template choice, and why it is not benchmark tuning

The first attempt used `chatml`, and both arms spent all 256 tokens inside a reasoning preamble —
the comparator answered 4 of 9 and we answered 0. `chatml-no-think` is the envelope the artifact's
own chat template defines for `enable_thinking = false` (`<think>\n\n</think>\n\n`), it is declared
per model, and it was applied to **both** arms, so no gate moved in our favour. Raising
`--max-tokens` would have been tuning the benchmark; changing the template is a declared per-model
protocol choice, which is what the runbook prescribes.

## What this does not establish

- **No profile.** 2.65× short on decode, 7.63× end to end. Where the time goes is unmeasured.
- **One host class.** EPYC 7713, AVX2, no AVX-512, 16 vCPU, and the host's own load average was ~90
  across both arms. The arms ran back to back, so the ratio holds; the absolute rates do not
  transfer.
- **Not a parity claim.** Per-layer parity was separately measured at 0.0005% worst over 28 layers
  at position 0 with F16 KV (`../2026-10-31-mellum-oracle-parity/`), and the multi-position ladder
  that exposed the rope defect is not re-run here.
- **8.07 GB is the smallest published quantisation**, so the "edge" caveat in #281 still stands for
  the intended host class.
