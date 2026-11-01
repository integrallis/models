# maple: two-arm qualification — 2026-10-31

**Verdict: `FAILED_RELATIVE_GATE`.** The contribution gate passes comfortably. We decode maple
**4.8× slower than the reference**, so it cannot be published under the two-arm policy.

This answers #318, which replaced #299 once maple's parity was settled (worst per-layer delta
0.00064% over 24 layers, `../2026-10-31-maple-oracle-ladder-postfix/`). #318's stated blocker —
"no comparator backend can read the artifact" — turned out to be false: Ollama cannot parse TQ1_0,
but `llama-server` built at the pinned oracle revision reads it fine, and that is the arm below.

## Provenance

| | |
|---|---|
| artifact | `maple-preview-TQ1_0-head-Q4_K.gguf`, deepgrove/maple-preview-GGUF rev `f5466f918e0c…` |
| sha256 | `54016e4d543bd688829e67103fc85b8396db94b7f8eb3f81fa95884e44393872` — **verified on the pod after download**, and equal to the digest the 2026-10-24 record pins |
| host | RunPod `cpu3c`, **16 vCPU, AMD EPYC 7713**, 32 GB, AVX2, **no AVX-512**, US-CA-2 |
| candidate | `rust-ffm`, `models@0.5.9+maple-ccb8e31d`, Rust kernel built from this tree on the pod |
| comparator | `llama.cpp`, `ggml-org/llama.cpp@3f03b17724ad8b69040825f45c43c59b092a0244` — the same revision that produced the parity ladder |
| arms | **sequential, same host, same session**; server readiness gated on HTTP 200 |

`sameWorkload()` is satisfied on every field it compares — workload `general`, corpus
`4b27eba8f166…`, template `chatml-no-think`, retrievalTopK 1, maxOutputTokens 256, contextLength
2048, **threads 16 on both arms**, grounding policy
`bounded-context-injection-screened-citation-safe-statement-grounding-v24`. A comparator that
defaults its thread count is excluded with "benchmark workload differs", so both were passed
explicitly.

## Gates

| | candidate | comparator | gate | |
|---|---:|---:|---|---|
| `modelAnswerRate` | **0.7778** | 0.7778 | ≥ 0.3333 | **PASS** |
| `modelAnswerCorrectRate` | **1.0000** | 1.0000 | ≥ 0.90 | **PASS** |
| p50 decode tok/s | **13.02** | **62.43** | — | — |
| `decodeThroughputRatio` | | | ≥ 0.80 | **0.2085 — FAIL** |
| end-to-end p95 ms | **95 003** | **2 688** | — | — |
| `endToEndLatencyRatio` | | | ≤ 1.50 | **35.35 — FAIL** |

9 attempts per arm, 9 successful, **zero runtime failures on either side**.

**The contribution numbers reproduce the earlier fleet result exactly.** Shard 900 on an
`m6a.4xlarge` (EPYC 7R13) recorded `modelAnswerRate` 0.7778 and `modelAnswerCorrectRate` 1.0 on
`models@0.5.4+maple1-416ccc3d`. Same figures here on a different host and a later library, which is
independent corroboration rather than a repeat of one measurement.

**`correctAnswerRate` is 1.0 on both arms and means nothing here.** It scores the pipeline: the
grounding policy substitutes text extracted from the retrieved document when a generation is screened
out. Reporting it as model quality is how 14 models were once announced as qualified when 5 passed.
`modelAnswerRate` and `modelAnswerCorrectRate` are the model's numbers and are reported beside it.

## What the failure is, and what it is not

**Not accuracy.** maple answers 7 of 9 and is correct every time it answers, matching the reference
arm case for case.

**Not parity.** The arithmetic is settled: 0.00064% worst per-layer delta against this same oracle
revision.

**Throughput.** 13.02 against 62.43 tok/s. #299 carried an *expectation* of "~2.5-3x short" and
explicitly flagged it as an expectation rather than a measurement. Measured, the gap is **4.8×**, so
that estimate was optimistic by a factor of about 1.8.

The 35× end-to-end ratio is larger than the decode ratio because the prefill is included and TQ1_0
prefill is where the exact path is weakest: 168 of maple's tensors are TQ1_0, and a 2048-token
context is prefilled per case.

## What this does not establish

- **Not a statement about the whole runtime.** This is one architecture whose weights are 1.6-bit
  ternary, served through a kernel route that exists for TQ1_0 alone. It says nothing about Q4_K or
  Q8_0 models, several of which are qualified in the catalog.
- **No profile.** This says we are 4.8× slower, not where the time goes. #299 recorded a JFR profile
  putting the matmul at ~43% with `attendGroupHeadByHead` and `KvCache.writeAttentionScores` above
  it, which is why the AVX2 kernel returned 1.76× rather than the 8-16× first predicted. That profile
  predates the computation fix and would need re-taking.
- **One host class.** EPYC 7713, AVX2, no AVX-512, 16 vCPU. The host's own load average was ~100
  across both arms (a shared 64-core machine), which affects the absolutes; the arms ran back to back
  under the same conditions and the ratio is 4.8×, far outside any plausible noise band for that.
- **F16 KV was not enabled.** maple's *parity* requires `-Dmodels.purejava.halfPrecisionKvCache`,
  which is off by default. This run measured the shipped default, which is the right thing to
  qualify, but it means the qualified configuration is not the bit-matching one.
