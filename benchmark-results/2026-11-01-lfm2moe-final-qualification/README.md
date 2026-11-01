# lfm2moe, final admitted qualification: e2e gate now passes, decode gate is terminal

**Verdict: `FAILED_RELATIVE_GATE`**, on decode alone, after every wiring-level optimisation
available. This is the record the decision to drop the port rests on.

Measured 2026-11-01, pod `9lpalxaf8vzgpg`, 16 vCPU of an AMD EPYC 7713 (no AVX-512), both arms
sequential in one session, `--warmups 1 --iterations 3`, ollama 0.40.13 comparator on the identical
artifact. `exclusions: {}` -- the policy compared the arms rather than discarding either.

| | candidate | ollama | ratio | gate |
|---|---:|---:|---:|---|
| p50 decode tok/s | 28.74 | 56.51 | **0.5087** | >= 0.80 **FAIL** |
| p50 prefill tok/s | 32.72 | 143.01 | 0.2288 | not gated |
| end-to-end latency | | | **1.4393** | <= 1.50 **PASS** |

Contribution passes: `modelAnswerRate` 0.4444 against a 1/3 minimum, `modelAnswerCorrectRate`
1.0000 against 0.90. `absoluteTier` `PERFORMANCE_REDUCED`, not a failure tier.

## The whole ladder

| stage | prefill | decode | decodeRatio | e2eRatio |
|---|---:|---:|---:|---:|
| baseline (kernel seam only) | 25.54 | 23.88 | 0.3974 | 2.8670 FAIL |
| + vectorised attention, swiGLU | 28.92 | 27.67 | — | — |
| + grouped Q/K/V | 29.32 | 28.08 | — | — |
| + prefill skips the output head | **32.72** | **28.74** | **0.5087** | **1.4393 PASS** |

Prefill +28.1%, decode +20.4%, and the end-to-end gate crossed from failing to passing. Contribution
was identical at every stage, and `Lfm2RewindTest.prefillReturnsWhatAdvancingTokenByTokenReturns`
holds prefill's logits bit-identical to the per-token path, so none of this traded accuracy.

The comparator's own decode moved between runs (60.08 earlier, 56.51 here, about -6%) which is host
noise on a shared 256-core box. Every ratio quoted is from the paired same-session arm, which is why
pairing is the protocol.

## Why decode is terminal and prefill was not

Prefill was doing wasted work: the output projection is `vocab_size x embedding_length`, 128000 x
2048 = 262M parameters, the widest matmul in the pass and about 21% of the weight bytes a token
reads, and it was computed for all ~190 tokens of a prompt with 189 results discarded. Removing
waste is cheap and it is what moved e2e.

Decode has no such waste left. The profile (`../2026-11-01-lfm2moe-decode-optimisation/`) puts ~75%
of runtime in the quantised matmul path, and that path already uses ggml's own algorithm: `Q4_0`
maps to `ActivationFormat::Q8_0` with `selected_q4_kernel()`, the integer AVX2 dot equivalent to
`vec_dot_q4_0_q8_0`. The model reads roughly 1.0 GB of weights per token, so 28.74 tok/s is about
29 GB/s against the comparator's ~57 GB/s, and for 16 cores of this part the comparator is
plausibly near the practical memory-bandwidth ceiling.

Reaching 0.80 therefore needs the quantised GEMV about **1.6x faster** than it is -- a kernel
project against a mature, hand-tuned reference, not a wiring fix. Two results bound the odds:

- **Thread fan-out is a null.** 16 -> 8 native workers measured **33% worse** (28.08 -> 18.77). The
  kernel scales with cores, so there is no overhead win hiding in the dispatch shape.
- **The repack-tile layout is a prior null.** Worth 1.49-1.87x inside llama.cpp, it measured at
  parity in our port.

## Disposition

Decode at 0.5087 against a 0.80 floor, with the binding cost in a kernel that already implements the
reference algorithm and a comparator near hardware limits, is a terminal failure for this port under
this gate. The lfm2moe decoder is therefore removed; see the commit that does it for the list of
general fixes the port surfaced and which are kept, each of which stands on its own measurement.
