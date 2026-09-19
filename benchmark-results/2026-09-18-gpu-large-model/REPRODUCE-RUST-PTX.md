# Reproducing the Rust PTX arm

Campaign: `benchmark-results/2026-09-18-gpu-large-model/PREREGISTRATION.md` (Amendment 1).
Arm (c): our own minimal Rust kernels compiled to PTX. Branch `feat/native-cuda-kernels`.

**Every command below has been executed.** Which host it was executed on is marked per step, and
that distinction is the point of this file. The previous revision of this guide contained commands
that had never been run — `compare --model ... --prompts ...` and `profile-decode --warmups ...
--iterations ...` accept none of those options — and a GPU host sat idle at $1.09/hr while that was
discovered by hand. Nothing goes in here now that has not been run somewhere.

| step | what it is | run on |
| --- | --- | --- |
| 0 | `:backend-cuda:check` | macOS 25.6.0 x86_64, no GPU |
| 1 | `ptxas` assembly of the emitted module | L40S pod (Linux) |
| 2 | `cuda-kernel-gate --mode capability` | L40S pod — **PASSED** |
| 3 | `cuda-kernel-gate --mode parity` (G1) | macOS, no GPU, as a self-test only |
| 4 | `cuda-kernel-gate --mode decode` (G4) | macOS, no GPU, as a self-test only |

Steps 3 and 4 have never run on a device. What has been shown off-device is that they run, that
they produce the report shape below, and that they say plainly that a CPU-only run is not gate
evidence. The numbers they will produce on a device do not exist yet.

## Toolchain pins

| | |
| --- | --- |
| Rust | `nightly-2026-09-17` (`rustc 1.100.0-nightly (330d31712 2026-09-17)`) |
| Target | `nvptx64-nvidia-cuda` |
| Flags | `-C target-cpu=sm_80`, `-Zbuild-std=core,compiler_builtins`, `-Zbuild-std-features=compiler-builtins-mem` |
| Components | `rust-src`, `llvm-bitcode-linker`, `llvm-tools` |
| JDK | 25 |
| CUDA driver | any shipping `libcuda.so.1` with a compute-capability >= 8.0 device |
| CUDA toolkit | **not required to build**; required only for the `ptxas` check below |

Pinned in `backend-cuda/src/main/rust/models-cuda-kernels/rust-toolchain.toml`, cross-checked by
the `verifyRustToolchain` Gradle task, written into the packaged artifact descriptor, and emitted
in the `kernel.toolchain` field of every report JSON.

## The gate command

One command, three modes, one report shape. `models-bench cuda-kernel-gate`:

| mode | gates | needs a model |
| --- | --- | --- |
| `capability` | G2 scaffolding, G5 | no |
| `parity` | G1 | yes |
| `decode` | G4, and the overhead term | yes |

Every mode takes `--report <path>` and `--models-revision <40-char sha>`, both required, and
accepts `--require-device true` to refuse to run at all without an eligible device.

Exit status is the verdict: `0` every requested gate passed, `1` a gate failed, `2` the run could
not be performed (bad usage, or no device when one was required).

**There is no `compare` parity command and no `profile-decode` GPU arm.** `compare`
(`BenchmarkComparisonCli`) diffs two existing benchmark report JSONs and takes `--models
<report.json> --llama-cpp <report.json> --json --markdown`; it cannot load a model.
`profile-decode` (`DecodeProfileCli`) profiles one backend's decode loop with `--model --prompt
--prompt-file --context --token-id --warmup-tokens --measure-tokens --output`; it has no arm
selection, emits a JFR recording rather than a report JSON, and repeats one token rather than
decoding a sequence. Both still exist and are still useful for what they do. Neither is a gate.

## Step 0. Off-device — run these anywhere, no GPU

```bash
./gradlew :backend-cuda:check :models-bench:check
```

Runs, on any host with no GPU and no CUDA toolkit:

- 23 Rust parity tests (`cargo test --release`) — Q4_K and Q6_K bit-exact against an oracle
  transcribed from the CPU path the accelerator is gated against
  (`PanamaVectorUtilSupport.ggufQ6_KQ8_KMatVecDot`), the warp decomposition bit-exact for 1–64
  lanes, both formats bit-exact at 1/2/3/32/33/48 super-blocks, the measured disagreement between
  the two CPU Q6_K reductions, `f16` over all 65,536 bit patterns, `expf` within 1e-6 of the
  platform `expf`, attention within its stated 2.0e-5 L2 contract;
- `cargo clippy -D warnings`;
- the PTX compile, plus structural assertions on the emitted module (three entry points, `sm_80`,
  `fma.rn.f32` present, scratch in `.shared` and not global);
- the backend-cuda Java tests — packaging and digest, absent-driver fallback, format and shape
  refusals, routing counters, the KV-sharing refusal, and Q8_K + block-order fold bit-identical to
  `VectorUtil`. `CudaQ6KDeviceParityTest` is in this module too but launches kernels, so it skips
  off-device and is only evidence when it runs on a GPU host — see step 2a;
- the gate command's own tests: argument parsing, the greedy decode loop, the parity comparison and
  its divergence report, the G4 threshold arithmetic, and the off-device self-test semantics.

Verified green on macOS 25.6.0 x86_64, 2026-09-18.

## Step 1. GPU host — assemble the module first, it is cheap

```bash
ptxas -arch=sm_80 -O3 \
  backend-cuda/build/generated/cuda-resources/META-INF/models/cuda/models-cuda-kernels.ptx \
  -o /dev/null
```

`ptxas` ships with the CUDA toolkit and is Linux-only, so the inline-PTX fragments in `device.rs`
cannot be assembled off-device. If this fails, nothing below is worth running.

Ran clean on the L40S pod, 2026-09-18.

## Step 2. Capability gate — device, kernel identity, readiness (G2 scaffolding, G5)

```bash
./gradlew :models-bench:installDist
models-bench/build/install/models-bench/bin/models-bench cuda-kernel-gate \
  --mode capability \
  --require-device true \
  --report benchmark-results/2026-09-18-gpu-large-model/rust-ptx-capability.json \
  --models-revision "$(git rev-parse HEAD)"
```

**Measured on the L40S pod, 2026-09-18: PASSED.** Compute capability 8.9, driver 595.91.07, kernel
id `8f6fcdb39837`, readiness 525 ms against a 120,000 ms ceiling. The module loads on the device,
all three kernels resolve.

Off-device this exits `0` and prints `SKIP ... accelerated=false`, which is the G3 fallback
evidence. With `--require-device true` it exits `2` instead, which is what CI should use on a host
that is supposed to have a GPU.

## Step 2a. The Q6_K device regression test — run it before any model

```bash
./gradlew :backend-cuda:test --tests '*CudaQ6KDeviceParityTest'
```

`CudaQ6KDeviceParityTest` launches the Q6_K projection kernel against the CPU control for rows of
1, 2, 3, 32, 33 and 48 super-blocks, on two real Q6_K super-blocks lifted from Granite's
`token_embd.weight`, with no model, no forward pass, no attention and no MoE routing in the path.
It asserts raw-bit equality; there is no tolerance, because G1 has none.

It costs seconds and it localises everything the model-scale gates can only report as "prompt 0,
token 0". Run it first. If it fails, nothing below is worth the GPU time.

Off-device every arm skips with an explicit reason, which is not evidence of anything.

## Step 3. Parity — G1

```bash
export GEMMA4_Q4KM_GGUF=/path/to/gemma-4-26b-a4b-it-Q4_K_M.gguf

models-bench/build/install/models-bench/bin/models-bench cuda-kernel-gate \
  --mode parity \
  --model "$GEMMA4_Q4KM_GGUF" \
  --prompts benchmark-results/2026-09-18-gpu-large-model/prompts.txt \
  --max-tokens 64 \
  --context 4096 \
  --require-device true \
  --report benchmark-results/2026-09-18-gpu-large-model/rust-ptx-parity.json \
  --models-revision "$(git rev-parse HEAD)"
```

Loads the one GGUF twice — once with the CUDA kernels injected, once with
`GgufBatchedMatrixKernel.none()` — greedily decodes 20 prompts × 64 tokens on each, and compares
the **full token id sequences**. Exit `0` only if every sequence is identical. No token-level
tolerance is available and none is permitted.

The arms run one after the other, not side by side: a 26B Q4_K_M is 16.8 GB and holding two copies
would double host residency for nothing, since the comparison is over recorded sequences.

**Sampling is argmax, ties broken by the lowest token id.** There is no `--temperature`, no
`--top-k` and no `--seed`, because there is nothing to seed. The old guide passed `--temperature 0
--top-k 1 --seed 42`, which described this same rule through options the command does not have.
The rule is recorded in the report as `configuration.sampling`.

End-of-generation is recorded (`parity.sequencesHittingEndOfGeneration`) and deliberately **not**
obeyed. Stopping early would shorten the compared sequence exactly where the arms agree.

### If G1 fails

The report's `parity.firstDivergence` carries the prompt index, the token index, both token ids,
the routing counter delta for that exact token, `attentionRouted`, `projectionRouted`, and a
`reading` naming what the evidence implicates.

The projections are bit-exact by construction. Attention carries a stated 2.0e-5 relative-L2
contract because of `expf` (`UPSTREAM.md` CU-005). So a divergence at a token where attention
routed and no projection did points at device `expf` disagreeing with the host libm, **and the fix
is to make device `expf` agree exactly. G1 is not to be loosened.** A divergence where projections
routed is a kernel defect, not a tolerance question.

### Off-device

This mode still runs with no GPU, and both arms are then the Vector API path. That is a self-test
of the harness, not evidence for G1, and the command says so: the console line reads `SELF-TEST`
rather than `PASS`, `parity.selfTest` is `true`, `qualified` is `false` whatever the sequences did,
and the G1 evidence string ends `Not G1 evidence.` A CPU-vs-CPU pass must never be filed as a gate
result.

Run off-device on macOS with Qwen3 0.6B Q4_0, 2026-09-18: 20 prompts × 64 tokens, both arms
identical, `qualified: false`, `selfTest: true`.

## Step 4. The 27B-class decode measurement — G4

```bash
models-bench/build/install/models-bench/bin/models-bench cuda-kernel-gate \
  --mode decode \
  --model "$GEMMA4_Q4KM_GGUF" \
  --prompts benchmark-results/2026-09-18-gpu-large-model/prompts.txt \
  --max-tokens 64 \
  --warmup-tokens 16 \
  --context 4096 \
  --arm both \
  --require-device true \
  --report benchmark-results/2026-09-18-gpu-large-model/rust-ptx-decode.json \
  --models-revision "$(git rev-parse HEAD)"
```

`--arm both` measures the accelerated arm and the CPU control **in the same process on the same
host**, one after the other, and that control is G4's denominator. Measuring the denominator
anywhere else would not be a comparison.

Model: Gemma 4 26B-A4B IT Q4_K_M, 16,796,015,136 bytes, pinned by revision and sha256 from the
ModelJars catalog. The report recomputes and records the sha256 of whatever file was actually
passed, so a mismatch is visible rather than assumed. Also run Qwen3 0.6B Q4_0 as the continuity
control against the published figures.

G4 needs decode >= **3.0x** the CPU control on the same host. `GEMMA4_QUALIFICATION.md` measured
12.91 tok/s for this artifact on 8 AMD EPYC-Milan vCPUs, which would put the bar near 38.7 tok/s —
but that figure is from a different host and does not transfer. The control this command measures
is the only denominator that counts.

The report records, per arm: prefill and decode tokens/s, wall clock, prompt and generated token
counts, decode steps, resident and heap bytes at the end of the arm, and for the accelerated arm
the peak device bytes.

### Decode tokens/s counts the tokens that cost a forward pass

The first token of each sequence is the argmax of the prefill logits and costs no decode step. So
64 requested tokens are 63 decode steps, and `decodeTokensPerSecond` is 63 / decode seconds.
Charging the prefill-produced token to the decode clock would inflate short runs, which is the
regime G4 is measured in.

### G3, separately

`-Dmodels.cuda.disabled=true` is a real switch — `CudaGgufBatchedMatrixKernel.DISABLED_PROPERTY`,
checked first thing in `open()`, which then reports `disabled by models.cuda.disabled` as its
refusal reason. G3 asks that a build with the module present but ineligible lands within 5% of a
build without it on the classpath, so it is two separate processes, not an arm of this command:

```bash
# module present, refusing
MODELS_BENCH_OPTS="-Dmodels.cuda.disabled=true" \
models-bench/build/install/models-bench/bin/models-bench cuda-kernel-gate \
  --mode decode --arm control \
  --model "$GEMMA4_Q4KM_GGUF" \
  --prompts benchmark-results/2026-09-18-gpu-large-model/prompts.txt \
  --max-tokens 64 --warmup-tokens 16 --context 4096 \
  --report benchmark-results/2026-09-18-gpu-large-model/cpu-control-module-present.json \
  --models-revision "$(git rev-parse HEAD)"

# and the same run from a build without :backend-cuda on the classpath
```

`--arm control` on its own leaves G4 undecided rather than reporting a zero speedup: the report's
`decode.speedup` is `null` and the G4 evidence reads `G4 is a ratio and was not measured`.

## Step 5. Read the overhead term out of the report

This is the point of the exercise beyond pass/fail. The `routing` block carries
`launchesPerDecodeStep`, `transfersPerDecodeStep` and `activationBytesPerDecodeStep`. The
large-model analysis identifies per-token launch and transfer latency as the term that decides G4
and records that it has no measurement; these fields plus the wall clock give it one. **Report it
whether or not G4 passes** — a failing run that pins the overhead term is more useful than a
passing one that does not.

Three things about how that block is computed, because each of them would otherwise produce a
number that looks like a measurement and is not:

- **A step is a generated token, not a projection.** The kernel is handed one projection at a time
  and cannot see token boundaries, so it counts `decodeProjections` and the measurement harness
  marks `decodeSteps`. Had the kernel marked steps itself, `launchesPerDecodeStep` would have been
  launches-per-projection — 1.0, for every model, forever.
- **The per-step terms exclude warmup and prefill.** The raw counters are cumulative from process
  start. The per-step figures are summed from the deltas of measured decode steps only, so weight
  uploads, the warmup sequence and every prefill launch stay out of the per-token figure. The
  cumulative counters are in the same block, so the two can be compared.
- **Zero steps means unmeasured, not zero.** `routing.decodeStepsMarked` says which, and the
  console prints `unmeasured` rather than `0.000`.

`backend-cuda/README.md` predicts, from arithmetic alone and at an assumed 5 µs per operation:
~3.4x with the current per-projection dispatch, ~9.8x with grouped expert dispatch and one
synchronise per layer. If the measurement disagrees with both, the assumed per-operation cost is
what was wrong, and the report should say so rather than the prediction being quietly dropped.

`weightUploads` is the other thing to read. It must be one per tensor for the life of the process.
A count that grows with tokens means the design regressed into per-token streaming, which the
memory arithmetic puts below the CPU path.

## What the memory figures do and do not cover

`memory.peakDeviceBytes` counts kernel-owned allocations — weights, the staging scratch, the
per-call attention buffers. The CUDA context, the loaded module and the driver's own reservations
are not included, so it is a **lower bound** on the process's device footprint, not a reading of
it. For the real figure, watch `nvidia-smi` alongside the run.

`memory.peakProcessRssBytes` is process-wide (`VmHWM`), so in an `--arm both` run it is not
separable per arm. Run `--arm accelerated` and `--arm control` as separate processes if per-arm
host peaks are needed.

## The prompt set

`prompts.txt`, 20 prompts, committed, fixed order, sha256 recorded in every report as
`configuration.promptsSha256`. The file carries its own selection criteria in `#` comments, which
the reader skips. In short: self-contained (no retrieval, no current date), ASCII only, each with
an answer that runs well past 64 tokens, ordered shortest to longest with ties broken
alphabetically, across nine task families. No chat template is applied — the gate measures kernels,
not chat behaviour — so nothing about answer quality should be read out of a gate run.

## What a complete result set looks like

| file | gate | status |
| --- | --- | --- |
| `rust-ptx-capability.json` | G2 scaffolding, G5 | measured on the L40S, PASSED |
| `rust-ptx-parity.json` | G1 | not yet run on a device |
| `rust-ptx-decode.json` | G4, and the overhead term | not yet run on a device |
| `cpu-control-module-present.json` | G3 | not yet run on a device |

Plus the same set for arm (b), TornadoVM, on the same host with the same prompts and the same CPU
control, since the pre-registration chooses between the arms on G4 and a comparison across hosts
would not be a comparison.

A report from before the parity and decode modes existed has `schemaVersion: 1` and a two-member
`gates` object. Current reports are `schemaVersion: 2` with `gates.tokenParity` and
`gates.decodeSpeedup`, `null` in the modes that do not decide them.
