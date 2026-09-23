# backend-cuda

Models-owned GPU kernels for K-quant projections and decode attention. Written in Rust, compiled
to PTX, driven from Java through Panama FFM.

**Nothing in this document is a GPU measurement.** No kernel in this module has run on a device.
Every number below is either read out of this repository, read out of a vendor specification and
labelled as such, or arithmetic over those. The GPU-host checklist near the end lists what still
has to be proved on a device, and `benchmark-results/2026-09-18-gpu-large-model/REPRODUCE.md`
gives the exact commands.

## Scope

Two kernels go on the device, and nothing else:

| kernel | what it does |
| --- | --- |
| `models_q4k_decode_projection` | fused Q4_K dequantise-and-multiply projection |
| `models_q6k_decode_projection` | the same for Q6_K |
| `models_gqa_decode_attention` | grouped-query attention for the single-token decode step |

Q6_K is not optional. Q4_K_M promotes the value projection to Q6_K while query and key stay Q4_K,
so the fused QKV group is literally `Q4_K/Q4_K/Q6_K`; a Q4_K-only kernel would leave the attention
projections on the CPU for every large model in the catalog.

Model parsing, the tokenizer, the graph, norms, RoPE, residuals, sampling, the generation loop,
activation quantisation and KV cache ownership all stay in Java, exactly as `backend-native`
states for the CPU shim. No engine, no vendor math library, no inference runtime.

## Which track, and why

The campaign amendment names two candidate Rust-to-PTX tracks. Checking the registry rather than
assuming (details and the raw responses are in `UPSTREAM.md`):

- **`cuda-oxide`** on crates.io is `0.4.0`, last updated **2021-06-16**, a "high-level, rusty
  wrapper over CUDA". It is a driver-API binding, not a compiler, and it is five years stale. The
  amendment's description does not match this crate — most likely a name collision.
- **`cutile-rs`** does not exist. **`cutile`** does: `0.3.1`, published 2026-09-04, the cuTile
  tile-kernel track. Real and current.

We chose **neither**, and took rustc's own in-tree `nvptx64-nvidia-cuda` target instead.

**Why.** The amendment's policy boundary is "minimal and format-driven, Java owns everything else,
no vendor math library, no third-party inference runtime". The in-tree target satisfies that more
completely than either candidate, because there is no third-party compiler in the build at all —
the Rust we already write becomes PTX with a target flag. Concretely:

1. **Nothing new enters the dependency tree.** The kernel crate has zero dependencies. `cutile`
   pulls `cuda-core`, `cuda-async`, `cutile-compiler`, `cutile-ir`, `cutile-macro`, `tokio`,
   `dashmap`, `linkme` and `serde`.
2. **`cutile` also owns the host side.** It "authors *and executes*" kernels. Execution is the
   part the amendment assigns to Java. Adopting it would mean either not using half of it or
   giving up the boundary.
3. **CI can build it with no GPU and no CUDA toolkit.** Demonstrated: the PTX in this module was
   compiled on macOS x86_64 with neither installed. A track that needs the CUDA toolkit or MLIR at
   build time makes the GPU host a build dependency, not just a test dependency.
4. **The same source is the CPU reference.** `kquant.rs` and `attention.rs` compile unchanged for
   the host, so `cargo test` exercises the arithmetic that the PTX contains rather than a parallel
   re-implementation of it. That is what makes off-device verification worth anything.

The cost is that PTX kernels need a nightly toolchain and several unstable features, and that
`core` is missing things a device kernel wants. Those are enumerated, with reproductions, in
`UPSTREAM.md` — the alpha status is not a blocker, and that file is the basis for what we send
upstream.

**Version pins.** `nightly-2026-09-17` (`rustc 1.100.0-nightly (330d31712 2026-09-17)`), target
`nvptx64-nvidia-cuda`, `-C target-cpu=sm_80`, `-Zbuild-std=core,compiler_builtins`. Pinned in
`src/main/rust/models-cuda-kernels/rust-toolchain.toml`, cross-checked by `verifyRustToolchain`,
written into the artifact descriptor, and reported in every result JSON.

## Bit-exactness, and where it stops

The projections are **bit-exact** with the CPU K-quant path. That is a design property, not luck.

Every per-super-block reduction in the K-quant formats is `i32`, and integer addition is exact and
associative — so a device may compute the per-block partials in any order, on any thread. The only
floating-point operations are the two scale applications per Q4_K super-block (eight for Q6_K),
and those *are* order-sensitive, so the kernels replay them on a single lane in ascending
super-block order with the same fused multiply-add. PTX lowers that to `fma.rn.f32`, the same
IEEE-754 operation with the same rounding.

So the expensive part parallelises freely and the cheap tail is replayed in order. One warp per
output row, lane `l` taking super-blocks `l, l+32, …` into shared memory, then lane 0 folding.
`tests/kquant_parity.rs` asserts this for lane counts 1, 2, 3, 5, 8, 32 and 64 against an oracle
transcribed verbatim from the shipped CPU kernel.

Q6_K additionally keeps **eight** float lane accumulators. That is a determinism contract, not an
optimisation — the CPU scalar path maintains them so its float reduction order matches the 8-lane
SIMD order of the Vector API path. `q6k_single_accumulator_would_diverge_so_the_eight_lanes_are_load_bearing`
scans 64 fixtures and fails if collapsing them ever stops being observable, so the test cannot
quietly stop testing anything.

**Attention is not bit-exact**, and the reason is exactly one function. `core` has no `exp` and
rustc emits no libdevice linkage for `nvptx64`, so the crate ships its own. Everything else in the
head — the score dots, the scaling, the maximum, the ordering of the softmax sum, the value
accumulation — is bit-identical, and `the_device_decomposition_is_bit_exact_with_the_crate_reference`
proves the warp decomposition changes nothing. The stated contract against the CPU kernel is
relative L2 ≤ `2.0e-5`, the same contract the sibling TornadoVM attention branch adopted so the two
arms are held to one standard. `the_declared_contract_is_tight_enough_to_catch_a_wrong_kernel`
checks the bound is not an unreachable ceiling.

## Weights are uploaded once, from the mapped file

`cuMemcpyHtoD` takes a host pointer, and a mapped GGUF tensor's `MemorySegment` address is that
pointer. So a weight upload reads the mapped file directly and **no host-side duplicate is created
at any point**. Uploads are keyed on the segment address, so a tensor is uploaded once and shared
by every batch shape thereafter.

This is a real difference from the TornadoVM arm rather than a stylistic one. There,
`ByteArray.fromSegment` allocates a fresh off-heap segment and copies into it, and prefill and
decode are separate plans — about **33 GB of host memory for a 26B model before a byte reaches the
device**, per the analysis on `feat/tornado-large-model-plan`. Here that term is zero.
`CudaRoutingCounters.weightUploadBytes()` reports it rather than leaving it claimed.

## KV cache ownership, and gate G6

The KV cache stays in Java, and the accelerator **refuses** any attention step whose attended
window is not a single contiguous span.

A shared KV prefix is exactly what produces a second span: a fork and its source point at the same
host rows. A device-resident mirror would serve both spans correctly and quietly hold two copies
of the prefix — a silent loss of a property we publish, rather than a visible failure. Refusing
keeps the sharing intact by construction, and `CudaRoutingCounters` counts the refusal so it is
observable rather than assumed. `CudaKvSharingRefusalTest` pins it in both directions: it must fire
for a shared-prefix shape and must *not* fire for an ordinary one, or the accelerator would be
inert everywhere and G2 would catch it.

## What is verified off-device

All of this runs on any host — no GPU, no CUDA driver, no CUDA toolkit. It is what `./gradlew
:backend-cuda:check` runs.

| check | where |
| --- | --- |
| Q4_K and Q6_K row dots bit-exact vs the CPU kernel, on real block fixtures | `tests/kquant_parity.rs` |
| The warp decomposition is bit-exact for 1–64 lanes | `tests/kquant_parity.rs` |
| The Q6_K eight-lane contract is observable and observed | `tests/kquant_parity.rs` |
| `f16` conversion across all 65,536 bit patterns | `tests/kquant_parity.rs` |
| Six-bit scale/min unpacking over 4,096 random packings | `tests/kquant_parity.rs` |
| Q8_K rounding, sums, zero blocks | `tests/kquant_parity.rs` |
| `expf` within 1e-6 relative of the platform `expf` over [-120, 40] | `tests/attention_parity.rs` |
| Attention device decomposition bit-exact vs the crate reference | `tests/attention_parity.rs` |
| Attention within the 2.0e-5 L2 contract vs the CPU kernel, both Gemma 4 shapes | `tests/attention_parity.rs` |
| The contract is tight enough to catch a wrong kernel | `tests/attention_parity.rs` |
| **Java Q8_K + block-order fold bit-identical to `VectorUtil`** | `Q8KActivationsTest` |
| PTX exports the three kernels, targets `sm_80`, uses `fma.rn.f32` | `PtxModuleContractTest`, build |
| PTX scratch is in `.shared`, not global | `PtxModuleContractTest`, build |
| Packaged PTX digest matches its descriptor | `PtxModuleContractTest`, build |
| Absent driver, disabled switch, unsupported format, ineligible shape all fall back | `CudaAcceleratorFallbackTest` |
| Routing counters are per format and per stage; inert is visible | `CudaRoutingObservabilityTest` |
| Shared-prefix attention is refused; ordinary attention is not | `CudaKvSharingRefusalTest` |

Between them these close the chain: the Rust tests prove the arithmetic in the PTX matches the CPU
Rust kernel, and `Q8KActivationsTest` proves the Java side quantises and folds exactly as
`vectors-core` does. What neither can prove is that the *compiled* PTX executes as its source says.

## What needs a GPU host

1. **`ptxas -arch=sm_80` accepts the module.** `ptxas` ships with the CUDA toolkit (Linux), so the
   inline-PTX fragments in `device.rs` have never been assembled. This is the first step and it is
   cheap; it needs no timing run.
2. **The kernels produce the values the host tests predict.** Everything above is the algorithm on
   a CPU; a device run is the algorithm on a GPU.
3. **G1 token-level parity** on a real GGUF: greedy decoding reproducing the Vector API token
   sequence for 20 prompts × 64 tokens.
4. **G4**, and with it the per-token launch and transfer overhead, which is the term the
   large-model analysis names as decisive and has no measurement for.
5. **G5 readiness.** Expected to be trivial here — there are no plans to compile, only a
   `cuModuleLoadData` and the weight uploads — but "expected" is not "measured".

## Expected decode ceiling, before any GPU run

Arithmetic, not measurement. Inputs, each labelled:

- Per-token weight traffic **2.42 GiB** (resident 1.54 + 8/128 × 15.13) — *computed* on
  `feat/tornado-large-model-plan` from the model's measured tensor inventory.
- CPU control **12.91 tok/s** median decode — *measured*, `GEMMA4_QUALIFICATION.md`.
- L40S bandwidth 864 GB/s — *vendor specification, not verified here*. At 70% achieved: 605 GB/s.
- Weight-read floor: 2.599 GB ÷ 605 GB/s = **4.30 ms/token**, i.e. 233 tok/s, 18.0× CPU.
- Per-operation host cost (launch, small copy, or synchronise): **~5 µs assumed**. This is the
  term with no measurement; the counters in this module exist to replace it with one.

The design's own overhead depends entirely on dispatch shape, and that turns out to be what
decides G4 — not the kernels.

| dispatch shape | ops/token | overhead | total | decode | vs CPU |
| --- | ---: | ---: | ---: | ---: | ---: |
| **A. As implemented.** Per-projection round trip and synchronise; each of 8 experts dispatched separately | ~3,700 | 18.5 ms | 22.8 ms | ~44 tok/s | **~3.4×** |
| **B. Grouped experts, one synchronise per layer** | ~720 | 3.6 ms | 7.9 ms | ~127 tok/s | **~9.8×** |

Shape A is what the current SPI forces: `multiply` returns a `float[]`, so it must synchronise
before returning, and a 30-layer MoE with 8 active experts does that roughly 900 times per token.
It lands just above the 3.0× bar with no margin — which on a term estimated at 5 µs is not a
result anyone should plan around.

**So the honest answer on G4: yes, these two kernels are sufficient in principle, and no, the
current dispatch shape does not give them room.** The two kernels cover 2.42 GiB of the 2.42 GiB
touched per token; there is no third kernel that would help, and adding one would not move the
binding term. What has to change is the *shape* of the dispatch, in this order:

1. **One synchronise per layer rather than per projection.** The single biggest term. Requires an
   SPI that can hand back a device-resident activation handle instead of a `float[]` — a change in
   `backend-java`'s `GgufBatchedMatrixKernel`, outside this module's two-kernel boundary, and the
   thing to propose next.
2. **Grouped expert dispatch**: the 8 active experts' gate and up projections in one launch over a
   flattened row space. This is a kernel change, and it is the only one worth making before a
   device measurement.
3. Only then, if the kernels prove instruction-bound rather than latency-bound, `dp4a` for the
   Q4_K inner loop (CU-004).

If a device measurement puts per-operation cost well under 5 µs, shape A may be enough on its own.
That is precisely why the counters report launches, transfers, bytes and synchronises per decode
step: the first run on a GPU should *measure* this term rather than produce another estimate of it.

## Build

```bash
./gradlew :backend-cuda:check          # everything above, no GPU required
./gradlew :backend-cuda:compilePtx     # just the PTX
```

The PTX build needs the pinned nightly with the `rust-src`, `llvm-bitcode-linker` and `llvm-tools`
components and the `nvptx64-nvidia-cuda` target; `rust-toolchain.toml` declares all of them, so
`rustup` installs them on first use.

## Layout

```
src/main/rust/models-cuda-kernels/
  src/kquant.rs      K-quant arithmetic; host and device
  src/attention.rs   GQA decode arithmetic and expf; host and device
  src/float.rs       the fma shim core does not provide (UPSTREAM.md CU-002)
  src/device.rs      ptx-kernel entry points; scheduling, no arithmetic
  tests/             off-device parity against the CPU kernel
src/main/java/.../cuda/
  BundledPtxModule            hash-verified PTX from the jar
  CudaDriver                  Panama FFM to the CUDA driver API
  CudaGgufBatchedMatrixKernel the SPI implementation; routing and fallback
  CudaRoutingCounters         per-format, per-stage observability (G2)
  Q8KActivations              host-side activation quantisation
```
