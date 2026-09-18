# GPU acceleration for large models: pre-registration

Frozen 2026-09-18, before any kernel runs on a GPU host and before any device is provisioned.
Nothing below is retuned after a result is known.

## Why this campaign exists

The published Java GPU path (`backend-tornado`, TornadoVM PTX) accelerates **Q4_0 projections only**.
Attention always runs on the CPU Vector API, and the qualification gates were measured on Qwen3
0.6B Q4_0 (A16-2Q prefill 1.93x / decode 1.21x; A40-4Q 4.40x / 1.32x). Every large model in the
ModelJars catalog is Q4_K_M, so the accelerator does nothing for them. That is a capability gap,
not a caveat, and this campaign closes it or reports honestly why it cannot be closed.

## The three work items (dependency order)

1. **K-quant device kernels** (`feat/tornado-kquant-projections`): Q4_K (and Q6_K where natural)
   projections as Java TornadoVM kernels.
2. **Attention on device** (`feat/tornado-attention`): decode-path attention, with an explicit,
   justified decision about where the KV cache lives and what that does to our published physical
   KV prefix-sharing claim.
3. **Large-model plan and eligibility** (`feat/tornado-large-model-plan`): memory arithmetic, plan
   shape, and an accurate capacity gate for 27B-class Q4_K_M models.

## Gates, fixed now

**G1 — parity (correctness, blocking).** On each accelerated path, greedy decoding must reproduce
the Vector API token sequence exactly for 20 prompts x 64 tokens on the same GGUF. Any divergence
fails the gate; a numeric tolerance is acceptable only inside a kernel unit test with the contract
stated, never at the token level.

**G2 — routing observability (blocking).** Every run records which work ran on device, per format
and per stage. A run that reports 0 accelerated operations is an inert accelerator and fails,
whatever its speed.

**G3 — no-regression (blocking).** With the accelerator present but ineligible (unsupported device
or format), tokens/s must be within 5% of the same build with the module absent. The fallback must
cost nothing.

**G4 — decode performance (the one that matters).** On a qualified NVIDIA device, decode tokens/s
for a 27B-class Q4_K_M model must beat the same host's CPU-only Vector API decode by **>= 3.0x**.
Rationale fixed in advance: a decode win below that does not justify a GPU dependency for a model
of this size, given the published CPU path already serves it. Prefill is reported but does not gate.

**G5 — startup honesty.** Eager readiness (plan compile) time is reported with every result. If
readiness exceeds 120 s for the 27B profile, the result is reported as "not deployable as measured"
even when G4 passes, and the fix is named.

**G6 — KV sharing.** If GPU attention cannot preserve physical KV prefix sharing, the activated-LoRA
hybrid path must refuse the accelerator rather than silently lose sharing. Tested, not asserted.

## Measurement protocol

- **Device:** one qualified NVIDIA GPU, chosen by the memory arithmetic (Vultr A40 24 GB or
  L40S 48 GB). Recorded with driver, TornadoVM version, JDK, and CPU model.
- **CPU control on the same host**, same prompts, same seed, same build with the accelerator absent.
- **Model:** a 27B-class Q4_K_M GGUF pinned by revision and sha256 from the ModelJars catalog, plus
  Qwen3 0.6B Q4_0 as the continuity control against the published numbers.
- **Prompts:** the frozen prompt set recorded with the run; 3 warmups, then medians over >= 10 runs.
- Every number lands in a report JSON carrying the full configuration and environment, per our
  reproducibility standard, with a REPRODUCE.md beside it.

## Stated in advance

- **Decode is memory-bound.** The published 1.21x-1.32x decode figures are what projections-only
  acceleration buys. If K-quant kernels plus device attention do not move decode past G4, the honest
  conclusion is that this design does not pay for large models, and we report that rather than
  quietly shipping a 1.1x accelerator.
- **Vulkan is out of scope here.** It is the cross-vendor route (AMD/Intel/Apple) and is unstarted;
  it is not a substitute for closing these three gaps on the path we already ship.
- **Re-release rule:** when a gate passes, the docs' "Qualified Scope" section and the catalog's
  backend flags are updated in the same release that ships the capability. The published guide must
  never claim more than the gates measured, and must not under-claim what they did.

## Amendment 1 (2026-09-18, before any kernel runs on a device): our own Rust GPU kernels

**Decision.** The GPU compute path is our own minimal Rust kernels compiled to PTX (NVIDIA's CUDA
Rust, released 2026-09-08: `cuda-oxide` compiles ordinary Rust to PTX; `cutile-rs` is the tile-level
track on stable Rust). TornadoVM is demoted from destination to comparison arm.

**Why, stated plainly:**
- The TornadoVM jar is not self-contained: applications must install a TornadoVM distribution and
  launch through it, so "add a dependency and get acceleration" is false today. A Rust shim ships
  the way `backend-native` already does, as hash-verified platform binaries inside a jar.
- We control the kernels the formats actually need (fused K-quant dequantise-and-multiply, grouped
  attention) instead of trusting an external compiler with them.
- The published decode figures (1.21x-1.32x) are weak enough that the compiler is not visibly
  earning its dependency.

**Policy boundary, fixed here.**
- **Minimal and format-driven.** Only the kernels our hot paths need: K-quant projections and
  decode attention. No general engine, no vendor math library, no third-party inference runtime.
- **Java owns everything else.** Model parsing, tokenizer, graph, KV cache ownership, sampling and
  the generation loop stay in Java, exactly as `backend-native` states for the CPU shim.
- **Sunset condition, named rather than implied.** This shim is retired when OpenJDK's own
  accelerator work (Project Babylon / HAT) can compile our Java kernels to the same devices at
  parity. Until then it does not fade, and we say so instead of pretending otherwise.
- **No Vulkan, no CUDA C++.** Cross-vendor support, if ever, is a separate decision with its own
  pre-registration.

**Gates are unchanged.** G1 parity, G2 routing observability, G3 no-regression, G4 decode >= 3.0x
for a 27B-class Q4_K_M model, G5 startup honesty, G6 KV sharing. The Rust arm and the TornadoVM arm
are measured on the same host, same prompts, same CPU control, and the winner is chosen on G4. If
neither clears it, we ship neither and report that.

**Comparison arms:** (a) CPU Vector API control; (b) TornadoVM Java kernels; (c) our Rust PTX
kernels. Same GGUF, same seeds, same prompt set.
