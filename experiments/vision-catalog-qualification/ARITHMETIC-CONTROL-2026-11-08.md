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
four cases are a regression screen, not qualification. The bounded public-API screen completed
71/80 Q4_K_M cases (55 nonblank) before its deadline. One new mismatch remains: TextVQA 38508
produces `Unknown` versus reference `0`. Its full image embeddings are bit-identical; disabling
pruning and substituting independent ggml attention both retain `Unknown`. This isolates the
remaining discrepancy to another decoder operation, not the encoder or attention kernel alone.
Q4_K_S did not start before that screen deadline; no full-corpus qualification is claimed.

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

## Isolated repeat and CI follow-up

The diagnostic worker was archived, its archive hash verified locally, and its instance, volume,
security group and key deleted at 20:13 UTC. The next bounded campaign has two m6a.4xlarge workers,
one variant each, with a USD 6.50 combined ceiling and deletion by 2026-11-09 00:13:34 UTC.
The deployed production source is `282bbd8d3c29a7a2c0176380dc73e4dd4dd16862`; subsequent
documentation and test-only commits do not change that payload. Each worker independently runs
the unchanged full task and integration gates, including both reference KV formats. Q4_K_M
first captures a separate last-token decoder layer trace for TextVQA 38508. Diagnostics finish
before timing begins. Failure remains a failure; these runs do not waive the known discrepancy.

The first PR CI run found native Java instruction coverage at 79.88%. Added tests exercise ordered
projection cancellation and ownership, a pruned last query versus the complete 64-row attention
batch, rejected calls preserving state, and prepared-kernel lifetime. The complete native suite
now passes locally with instruction coverage 80.61%, above the unchanged 80% gate.

ModelJars 0.1.77 and its site update are published independently. All 48 Maven JAR/POM/signature
files match the signed bundle; the published SDK uses Models 0.5.17 and its backend dependencies
use Vectors 0.1.33. That release does not include this unreleased arithmetic branch.
