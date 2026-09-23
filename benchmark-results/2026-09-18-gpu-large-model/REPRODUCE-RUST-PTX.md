# Reproducing the Rust PTX arm

Campaign: `benchmark-results/2026-09-18-gpu-large-model/PREREGISTRATION.md` (Amendment 1).
Arm (c): our own minimal Rust kernels compiled to PTX. Branch `feat/native-cuda-kernels`.

**Nothing in this arm has run on a GPU.** Everything below marked "off-device" has been run and is
green; everything marked "GPU host" has not been run at all. The distinction is the point of this
file.

## Toolchain pins

| | |
| --- | --- |
| Rust | `nightly-2026-09-17` (`rustc 1.100.0-nightly (330d31712 2026-09-17)`) |
| Target | `nvptx64-nvidia-cuda` |
| Flags | `-C target-cpu=sm_80`, `-Zbuild-std=core,compiler_builtins`, `-Zbuild-std-features=compiler-builtins-mem` |
| Components | `rust-src`, `llvm-bitcode-linker`, `llvm-tools` |
| JDK | 25 |
| CUDA driver | any shipping `libcuda.so.1` with a compute-capability ≥ 8.0 device |
| CUDA toolkit | **not required to build**; required only for the `ptxas` check below |

Pinned in `backend-cuda/src/main/rust/models-cuda-kernels/rust-toolchain.toml`, cross-checked by
the `verifyRustToolchain` Gradle task, written into the packaged artifact descriptor, and emitted
in the `kernel.toolchain` field of every report JSON.

## Off-device — run these anywhere, no GPU

```bash
./gradlew :backend-cuda:check
```

Runs, on any host with no GPU and no CUDA toolkit:

- 21 Rust parity tests (`cargo test --release`) — Q4_K and Q6_K bit-exact against an oracle
  transcribed from the shipped CPU kernel, the warp decomposition bit-exact for 1–64 lanes, the
  Q6_K eight-lane determinism contract, `f16` over all 65,536 bit patterns, `expf` within 1e-6 of
  the platform `expf`, attention within its stated 2.0e-5 L2 contract;
- `cargo clippy -D warnings`;
- the PTX compile, plus structural assertions on the emitted module (three entry points, `sm_80`,
  `fma.rn.f32` present, scratch in `.shared` and not global);
- 18 Java tests — packaging and digest, absent-driver fallback, format and shape refusals, routing
  counters, the KV-sharing refusal, and Q8_K + block-order fold bit-identical to `VectorUtil`.

Verified green on macOS 25.6.0 x86_64, 2026-09-18.

## GPU host — none of this has been run

Provision per the pre-registration's memory arithmetic: **L40S 48 GB**, not A40 24 GB. Record
driver version, JDK, CPU model and the exact host.

### 0. Assemble the module — do this first, it is cheap and it is the one thing no CI check covers

```bash
ptxas -arch=sm_80 -O3 \
  backend-cuda/build/generated/cuda-resources/META-INF/models/cuda/models-cuda-kernels.ptx \
  -o /dev/null
```

`ptxas` ships with the CUDA toolkit and is Linux-only, so the inline-PTX fragments in `device.rs`
have never been assembled. If this fails, nothing below is worth running.

### 1. Capability gate — device, kernel identity, readiness (G2 scaffolding, G5)

```bash
./gradlew :models-bench:installDist
models-bench/build/install/models-bench/bin/models-bench cuda-kernel-gate \
  --report benchmark-results/2026-09-18-gpu-large-model/rust-ptx-capability.json \
  --models-revision "$(git rev-parse HEAD)" \
  --mode capability \
  --require-device true
```

Exit `0` passes. Proves the module loads on this device, all three kernels resolve, and reports
readiness. Expected to be fast — there are no plans to compile, only `cuModuleLoadData` and the
weight uploads — but "expected" is not "measured".

### 2. Parity — G1

```bash
models-bench/build/install/models-bench/bin/models-bench compare \
  --model "$GEMMA4_Q4KM_GGUF" \
  --prompts benchmark-results/2026-09-18-gpu-large-model/prompts.txt \
  --max-tokens 64 --temperature 0 --top-k 1 --seed 42 \
  --report benchmark-results/2026-09-18-gpu-large-model/rust-ptx-parity.json
```

Greedy decoding must reproduce the Vector API token sequence **exactly** for 20 prompts × 64
tokens on the same GGUF. Any divergence fails; no token-level tolerance is permitted.

The projections should be bit-exact by construction. Attention carries a stated 2.0e-5 relative-L2
contract because of `expf` (`UPSTREAM.md` CU-005), so if G1 fails, the first thing to check is
whether the divergence is confined to layers where attention routed — the routing counters say
which. If it is, the fix is to make the device `expf` agree with the host libm exactly, not to
loosen G1.

### 3. CPU control — same host, same prompts, same seed, accelerator absent

```bash
MODELS_BENCH_OPTS="-Dmodels.cuda.disabled=true" \
models-bench/build/install/models-bench/bin/models-bench profile-decode \
  --model "$GEMMA4_Q4KM_GGUF" \
  --prompts benchmark-results/2026-09-18-gpu-large-model/prompts.txt \
  --warmups 3 --iterations 10 --max-tokens 64 --temperature 0 --top-k 1 --seed 42 \
  --report benchmark-results/2026-09-18-gpu-large-model/cpu-control.json
```

`-Dmodels.cuda.disabled=true` is also the G3 arm: the same build with the module present but
refusing must land within 5% of a build without the module on the classpath. Run both.

### 4. The 27B-class decode measurement — G4

```bash
models-bench/build/install/models-bench/bin/models-bench profile-decode \
  --model "$GEMMA4_Q4KM_GGUF" \
  --prompts benchmark-results/2026-09-18-gpu-large-model/prompts.txt \
  --warmups 3 --iterations 10 --max-tokens 64 --temperature 0 --top-k 1 --seed 42 \
  --report benchmark-results/2026-09-18-gpu-large-model/rust-ptx-decode.json
```

Model: Gemma 4 26B-A4B IT Q4_K_M, 16,796,015,136 bytes, pinned by revision and sha256 from the
ModelJars catalog. Also run Qwen3 0.6B Q4_0 as the continuity control against the published
figures.

G4 needs decode ≥ **3.0×** the step-3 CPU control on the same host. The measured CPU control for
this artifact on 8 AMD EPYC-Milan vCPUs is 12.91 tok/s (`GEMMA4_QUALIFICATION.md`), so on a
comparable host the bar is about 38.7 tok/s — but the control must be re-measured on whatever host
is provisioned, not carried over.

### 5. Read the overhead term out of the report

This is the point of the exercise beyond pass/fail. The `routing` block carries
`launchesPerDecodeStep`, `transfersPerDecodeStep` and `activationBytesPerDecodeStep`. The
large-model analysis identifies per-token launch and transfer latency as the term that decides
G4 and records that it has no measurement; these fields plus the wall clock give it one. Report it
whether or not G4 passes — a failing run that pins the overhead term is more useful than a passing
one that does not.

`backend-cuda/README.md` predicts, from arithmetic alone and at an assumed 5 µs per operation:
~3.4× with the current per-projection dispatch, ~9.8× with grouped expert dispatch and one
synchronise per layer. If the measurement disagrees with both, the assumed per-operation cost is
what was wrong, and the report should say so rather than the prediction being quietly dropped.

## What a complete result set looks like

| file | gate |
| --- | --- |
| `rust-ptx-capability.json` | G2 scaffolding, G5 |
| `rust-ptx-parity.json` | G1 |
| `cpu-control.json` | the G4 denominator, and the G3 baseline |
| `rust-ptx-decode.json` | G4, and the overhead term |

Plus the same set for arm (b), TornadoVM, on the same host with the same prompts and the same CPU
control, since the pre-registration chooses between the arms on G4 and a comparison across hosts
would not be a comparison.
