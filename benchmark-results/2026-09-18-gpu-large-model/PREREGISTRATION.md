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

## G1 result, 2026-09-18T00:26Z: FAIL (measured on device)

Host: RunPod pod `e35e369wco2coq`, NVIDIA L40S 46 GB, compute capability 8.9, driver 595.91.07,
CUDA 13.2, 128 vCPU, US-TX-4. Model: Gemma 4 26B-A4B IT Q4_K_M, 16,796,015,136 bytes, sha256
88f4a13b… verified on the host. Kernels built `-C target-cpu=sm_80` from the pinned nightly.

**Passed before G1, and standing:**
- `ptxas -arch=sm_80 -O3` assembles the emitted module. No CI check covered this.
- Capability gate: module loads, all three kernels resolve, kernel id 8f6fcdb39837, **readiness 525 ms**
  against the G5 ceiling of 120 s. For contrast, the TornadoVM arm's plan compilation was estimated
  at ~948 s for this model class.
- `:backend-cuda:check` green on the host as well as on macOS.

**G1 FAILED.** Greedy decode diverges from the Vector API control at **prompt 0, token 0**:
accelerated token id 236743, control 108.

| evidence | value |
|---|---|
| attention routed | false |
| projections routed | true |
| Q4_K decode projections | 10,956 |
| Q6_K decode projections | 446 |
| refusals | 0 |
| kernel launches | 11,402 |
| host transfers | 21,154 |

The accelerator was not inert, and attention did not run, so the pre-registered `expf` tolerance
explanation does not apply. The K-quant path is bit-exact by construction and is verified so by 21
off-device tests, therefore a device-side difference is a **kernel defect**. Under G1 no tolerance
is admissible and the gate is not relaxed.

**G4 is not run.** A decode speed number from a kernel that computes the wrong answer would be
meaningless; the gate order exists to prevent exactly that. G4, G3 and the CPU control wait on a
fix that passes G1 on this device.

**Prime suspect**, recorded before diagnosis so it cannot be retrofitted: `UPSTREAM.md` CU-003 —
`link_section = ".shared"` silently emits into the global address space, which compiles, runs, and
races across blocks with no warning. That is the shape of a defect that passes sequential
off-device tests and fails under real parallelism. To be confirmed or eliminated with evidence.

## G1 root cause, 2026-09-19T16:13Z: Q6_K fold is not bit-exact past one super-block

Measured on NVIDIA A40 (46 GB, cc 8.6, driver 570.211.01, CUDA 12.8), after the L40S pod could not
be restarted (its host had no free GPUs; stopping a pod does not reserve hardware — recorded as an
operational lesson).

**Hypotheses eliminated with evidence, in order:**

| hypothesis | how it died |
|---|---|
| shared scratch emitted into global space (ledger CU-003) | the PTX declares `.shared` correctly |
| missing `cvta.to.shared` | adding it caused `illegal memory access (700)`, proving `mov.u64` already yields a shared address; reverted |
| batch index confused with batch count | host passes an index |
| missing barrier | four `bar.sync` emitted |
| transposed output index | `batch * rows + row` matches the host allocation |
| kernel argument packing | Q4_K passes 8 args, Q6_K 7 (needs no sums); both match |
| host/kernel constant drift | 256 / 16 / 32 / 128 agree on both sides |
| MoE expert-stack offset | each `(layer, expert, tensor)` is its own zero-copy pre-offset slice uploaded to its own buffer; **and dense Granite 4.1 3B fails identically** (prompt 0 token 0, accelerated 83017 vs control 40665) |
| Q4_K kernel | bisected 1→48 super-blocks on device, spanning the 32-lane warp boundary: bit-exact at every width |

**Root cause.** The **Q6_K** projection fold is bit-exact for a single super-block and diverges as
soon as a second joins the one-lane ascending-order float accumulation: cpu bits `-1098673107` vs
gpu bits `-1098673106`, one ULP. Per operation that is negligible; compounded over 30 layers it
flips an argmax, which is why G1 failed at the very first token.

**Why nothing caught it.** The 21 Rust parity tests compile `kquant.rs` for the *host* target. They
say nothing about what that same source does compiled for `nvptx64-nvidia-cuda` and executed on real
hardware. Only a device-executing test can guard this class of defect, and one now exists
(`CudaQ6KDeviceParityTest`, PR #198): two real Q6_K super-blocks from Granite's `token_embd.weight`
and a fixed-seed activation, with no model, forward pass, attention or MoE involved. It fails today
for the documented reason and skips cleanly off-device.

**G4 remains unrun**, correctly: a decode speed number from a kernel that computes the wrong answer
would be worse than no number.

**Latent fragility recorded, not chased:** `resident()` keys its device-buffer cache on
`MemorySegment.address()` with no length check. A synthetic sequence of many growing Q4_K calls
before a Q6_K call on a shared scratch produced 10^5–10^7 magnitude corruption and occasional NaN.
It did not reproduce under the fixed-size, one-call-per-tensor pattern of real use, so it is not
G1's cause, but the missing length check is real.

## G1 resolved in code, 2026-09-26T16:30Z — and the same defect existed upstream

**The fix.** The device fold was a transcription of the *wrong CPU reduction*. `models` carried two
Q6_K row reductions that were not bit-identical to each other:

| path | shape |
|---|---|
| `PanamaVectorUtilSupport.ggufQ6_KQ8_KMatVecDot` | one exact `i32` per super-block, one `fma` into one `f32` accumulator |
| `VectorUtilSupport.ggufQ6_KQ8_KScalarRowDot` (the `VECTOR_BITSIZE < 256` fallback) | eight `f32` lane accumulators, summed at the end |

`Q6KRowAccumulator` was transcribed from the scalar one. The control it is measured against is the
Panama one, because `TensorOps.ggufMatmul` takes that path on any host with 256-bit vectors — which
is every host that can also run CUDA. **The two folds coincide at exactly one super-block and differ
by one ULP from two on**, which is why a one-super-block test passed for weeks while G1 failed at
prompt 0, token 0. Reproduced off-device at the same bits the A40 reported: single accumulator
`-0.256989866` (bits `-1098673107`), eight lanes `-0.256989896` (bits `-1098673106`).

It was never an `fma` contraction problem. The emitted PTX contains `fma.rn.f32` and no unfused
`mul`/`add` pair either side of the change; what changed is how many fused operations there are and
in what shape. Per super-block: 8 `fma` + 8 `add` + 8 shared stores + 8 shared loads became 1 `fma`,
no `add`, 1 store, 1 load. Shared scratch dropped from 4096 to 1024 bytes. **Q4_K needed no change
and never had the defect** — its two CPU reductions are already the same shape — and that contrast is
what localised this.

**The same defect existed in `vectors`, and was fixed there independently.** Vectors PR #76 (merged
2026-09-22) made the scalar path bit-identical to the Panama path, unifying on the single-accumulator
integer `blockSum` + one `fma` form — the *same direction* this kernel fix took. Verified by reading
the published `vectors-core-0.1.23-sources.jar` from Central, and Models now pins
`vectorsVersion = 0.1.23`. So the follow-up this campaign recorded — "vectors disagrees with vectors
on Q6_K, so a Q6_K result is not reproducible across machines" — was real and is now closed upstream.
Worth stating plainly: a GPU parity gate surfaced a CPU reproducibility bug that affected every host
with narrow vectors, independent of any GPU.

**Status of the gates.** G1 is fixed in code and covered by tests that run without a device
(14 Rust host tests, including one that measures the two CPU reductions disagreeing rather than
asserting it from the source, and one that fails unless the crate is the Panama control). The
device-executing bisection at 1, 2, 3, 32, 33 and 48 super-blocks skips off-device. **No device run
has confirmed the fix, and G3, the CPU control and G4 remain unrun.** They are not being run now: the
prior campaign overspent, and a decode number from a kernel whose parity is unconfirmed would be
worth less than the GPU hour it cost.

**Kernel work already on main.** PR #196 merged the Rust PTX kernels and #204 changed the batch row
to come from `blockIdx.y` rather than a scalar parameter. **The defective eight-lane fold is
therefore live on main**, and `backend-cuda` is deliberately not published to Central, so no released
artifact carries it.

## Device confirmation, 2026-09-26 — measured on an NVIDIA A40

Host: NVIDIA A40, compute capability 8.6, driver 580.159.04, CUDA 13.0, Intel Xeon Gold 6342 at
2.80 GHz, 8 processors, 50 GB RAM, RunPod EU-SE-1. Kernel sha256
`070bb81cc7a10f077c6808a2825cdf69379423e1efbfea74aaa63f38d9cd9a2c`, PTX target `sm_80`, toolchain
`nightly-2026-09-17`. Model: dense Granite 4.1 3B Q4_K_M, sha256 `662b0626…fb32eb29`,
2,099,501,664 bytes — verified on the host. Models revision `203a591b`. Total billed time ~42 min.

### The Q6_K fix is confirmed bit-exact on device

`CudaQ6KDeviceParityTest`, raw-bit equality with no tolerance, **7 of 7 arms pass** — including the
two-super-block row that originally failed G1, and the bisection at 1, 2, 3, 32, 33 and 48
super-blocks, spanning the 32-lane warp boundary in both directions. The one-ULP fold defect is
closed.

### G1 parity: the projections pass, the attention kernel is the remaining divergence

Two runs on the same host and model, separated by the per-stage ablation switch:

| run | result |
|---|---|
| both stages routed, 20 prompts × 32 tokens | **FAIL** — first divergence prompt 0, **token 7**; accelerated 86897 vs control 22559 |
| **attention ablated, projections only**, 5 prompts × 32 tokens | **PASS** — 160 tokens token-for-token identical |

The passing run routed 35,376 projection operations to the device — Q4_K decode 27,200, Q6_K decode
6,976, Q4_K prefill 960, Q6_K prefill 240 — and recorded exactly one refusal,
`DECODE_ATTENTION/ablated-by-models.cuda.attention.disabled`. The ablation is **observable**, not
silent, which is what makes this a measurement rather than an absence.

**Conclusion.** The K-quant projection kernels are bit-exact at model scale. The divergence is
entirely in `models_gqa_decode_attention`, which carries a stated 2.0e-5 relative-L2 contract because
of `expf` (UPSTREAM.md CU-005). At token 7 that tolerance flipped an argmax. Note what the earlier
failure hid: while the Q6_K defect was live, parity died at token 0 and the attention tolerance was
never reachable as a distinct cause. Fixing one defect exposed the next.

This is a **G1 failure**, recorded as such. A tolerance that is acceptable for a tensor norm is not
acceptable for greedy argmax, so the pre-registered gate is correct to reject it. Either the attention
kernel becomes bit-exact, or G1 is amended with a pre-registered token-level rule — and that amendment
must be argued on its merits before any run, not chosen after seeing this result.

### G4 decode: 1.905×, below the 3.0× gate, and the overhead term explains why

| | |
|---|---|
| accelerated | **6.76 tok/s** |
| same-host CPU control | 3.55 tok/s |
| **speedup** | **1.905×** |
| gate threshold | 3.00× |
| verdict | **FAIL** |

Measured over 1,260 decode steps, 256,296 decode projections, 201 weight uploads:

| per decode step | measured |
|---|---|
| kernel launches | **241** |
| host transfers | **402** |
| activation bytes | **12,660,712** (~12.7 MB) |

`backend-cuda/README.md` predicted ~3.4× as dispatched and ~9.8× with grouped dispatch and one sync
per layer. **The measurement is below even the pessimistic figure**, so per the pre-registration the
assumed per-operation cost was wrong — the prediction is not being quietly dropped. 402 host transfers
and 12.7 MB moved per token, for 241 launches, is the dominant term: the device is being fed one
projection at a time with a round trip each. That is a dispatch-architecture problem, not a kernel
arithmetic problem, and it is what makes a 27B-class model impractical on this path today.

### What was not run, and why

The 26B MoE model was not fetched. There was no point: G1 fails on a 2.1 GB dense model for a reason
that has nothing to do with model size or MoE routing, and a 16.8 GB download plus its decode run
would have cost billed time to re-measure a known failure. The dense arm is also the better instrument
— it removes expert routing from the picture entirely.

G3 no-regression was not run. It gates an accelerator that does not yet pass G1.
