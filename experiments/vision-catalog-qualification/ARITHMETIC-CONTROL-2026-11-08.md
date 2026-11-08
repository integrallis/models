# LLaVA arithmetic correction and qualification repeat

Targets remain LLaVA 1.5 Q4_K_S and Q4_K_M. The existing accuracy, token-agreement, image
contribution and performance gates are unchanged. No catalog admission is claimed here.

## Measured localization

The retained baseline reached 63/64 nonblank exact-token matches for each quantization.
The independent decoder component replay identified native RoPE recurrence, Flash Attention
operand/accumulator rounding, and a missing final-layer native KV prefix as distinct causes.
The cache-continuity correction is tracked separately in PR #399.

For the diagnostic TextVQA image 35501, the corrected CLIP encoder matches all 23 reference
transformer layers bit for bit. Matching requires the patch convolution's HWC accumulation order,
fused pairwise Welford normalization, 192-element affine reductions initialized from the bias,
and the reference float32 softmax/QuickGELU arithmetic. The projector's two affine projections
also match independently. Its remaining GELU difference was localized to oneDNN's AVX2 formula;
the translated formula produces bit-identical complete projector output for that image.

The real unreleased Java/Rust runtime then reproduced the expected tokens on all four targeted
cases: Q4_K_M 38337 and Q4_K_S 35501, 36979 and 39587. Final-layer pruning stays enabled. These
four cases are a regression screen, not qualification. A frozen public-API corpus repeat is running.

## Production boundary

Java retains the graph, tokenization, image preprocessing, layer normalization, decoder state,
cache ownership and generation. Models-owned synchronous Rust kernels implement the projection,
attention and activation arithmetic. No external inference runtime or model graph is linked.
The generic text/deberta projection and causal-attention APIs retain their existing semantics.
The rotary recurrence is selected only by the LLaVA image attention provider.

Algorithm sources and retained licenses:

- ggml 2cbca389f515df455412dafa07e2b1359cc72a6d: attention arithmetic (MIT).
- PyTorch 2.9.0: Java pairwise Welford normalization (BSD-3-Clause).
- SLEEF 5a1d179df9cf652951b59010a2d2075372d67f68: float32 EXP for CLIP (Boost).
- oneDNN 8d263e693366ef8db40acc569cc7d8edf644556d: projector GELU/EXP (Apache-2.0).

## Performance experiment

The faithful one-row matrix kernel adds prefill work. The next candidate shares loaded B vectors
across six output rows and sixteen columns while preserving each output's increasing-K FMA order.
A vector GELU kernel preserves the scalar formula. Both must pass the independent fixtures and
real-model correctness before same-host timings. Losing production experiment paths are removed;
portable and ragged-shape fallbacks remain required implementations.

The current worker's full public-API run is a correctness screen: native compilation and small
synthetic fixtures may overlap. Its timings will not be submitted for admission. The final
qualification repeat must use a frozen source commit, isolated candidate/reference runs on the
same host, both reference KV formats, initial 1/3 and confirmation 3/7 repetition counts, and a
released-coordinate ModelJars consumer repeat after the required libraries are published.

Remote evidence is collected under `audit-artifacts/pr380-attention-dispatch-2026-11-08` in the
workspace. The owned instance i-08c5a8b85f1833c6d stops workloads at 20:05 UTC, archives by 20:08,
and has an unchanged hard termination deadline of 20:20 UTC and USD 2 ceiling.
