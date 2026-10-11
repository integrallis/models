# Models CUDA backend

`backend-cuda` supplies Models-owned Rust kernels compiled to PTX. Java loads the NVIDIA driver
through the Foreign Function and Memory API, manages device resources and orchestrates inference.
The parser, tokenizer, graph, sampling and generation loop stay in Models. No external inference
server or vendor math library is embedded.

## Scope and activation

The packaged PTX exports three kernels:

| Kernel | Work |
| --- | --- |
| `models_q4k_decode_projection` | Q4_K projection |
| `models_q6k_decode_projection` | Q6_K projection |
| `models_gqa_decode_attention` | Grouped-query attention during single-token decode |

The accelerator is opt-in and has no `ServiceLoader` registration. Call
`CudaGgufBatchedMatrixKernel.open()` and inspect its status; absent or ineligible hardware is
reported rather than assumed usable. The PTX targets `sm_80`, and the loader checks device/driver
eligibility, ABI and the bundled PTX digest. Running requires a compatible NVIDIA device and
driver. Building the PTX does not require a CUDA toolkit or GPU.

Weights are uploaded from mapped segments without a second host-side weight buffer. KV-cache
ownership remains in Java. Attention over a non-contiguous shared-prefix window is declined so
the fallback preserves the sharing contract; routing counters make this refusal observable.

See the [GPU guide](../docs/content/modules/ROOT/pages/gpu-acceleration.adoc) for the public
opening API, switches, routing counters and qualification commands.

## What has actually been measured

These are retained October 7 measurements, not a fresh benchmark of every supported device.

| Evidence | Recorded profile and result |
| --- | --- |
| [G1 token parity](../benchmark-results/2026-10-07-g1-parity/README.md) | RTX 4090, Granite 4.1 3B Q4_K_M: 1,280 identical token IDs over 20 prompts. |
| [G4 paired decode](../benchmark-results/2026-10-07-g4-dualpath/README.md) | RTX 4090, Granite 4.1 3B Q4_K_M: 31.26 tokens/s accelerated versus 5.06 for the CPU control, 6.184x. |

The G4 investigation found missing FFN gate/up routing. Adding that dispatch, without changing
the PTX kernels, moved the earlier accelerated result from 9.02 to 31.26 tokens/s on the recorded
hardware class; its paired CPU controls were 5.05 and 5.06. That finding supersedes the older README's claim
that transfer overhead alone explained the observed ceiling. It does not show that transfers are
free, or establish a throughput guarantee for other models, shapes or devices.

The [original design discussion](DESIGN-HISTORY.md) retains its predictions and failed reasoning
with an explicit historical label. [UPSTREAM.md](UPSTREAM.md) records the Rust-to-PTX toolchain
investigation. Neither substitutes for device qualification.

## Build and verification

```bash
./gradlew :backend-cuda:check
./gradlew :backend-cuda:compilePtx
```

The host checks cover packed arithmetic, attention decomposition, PTX exports and digest,
fallback behavior, routing counters and shared-prefix refusal. They run without a GPU and do
not prove device execution. The real-device commands in the GPU guide cover that boundary.

The Rust build pins `nightly-2026-09-17` in
[`rust-toolchain.toml`](src/main/rust/models-cuda-kernels/rust-toolchain.toml), including the target
and required components. Rustup installs them when needed. The Java binding and Rust kernels
live under `src/main/java/` and `src/main/rust/models-cuda-kernels/` respectively.
