# GPU is slower than our CPU kernel on a 0.6B decision model

Measured 2026-09-21 on one RunPod A40 (sm_86, driver 595.91.07, 96 vCPU host), Qwen3-0.6B Q4_K_M,
418-token document, ten 23-token tails. Both arms on the same host, interleaved, two repeats each.
The CPU arm uses our rust-ffm kernel, built on that box for the comparison; the GPU arm uses our
own Rust->nvptx64 PTX module (`models-cuda-kernels`, no vendor kernel libraries).

| | CPU rust-ffm | GPU CUDA A40 |
| --- | ---: | ---: |
| prefill | 3.775 / 3.356 s | 6.818 / 6.908 s |
| prefill ms/token | 9.03 / 8.03 | **16.31 / 16.53** |
| 10 tails batched | 3.247 / 2.758 s | 4.027 / 4.003 s |
| **ten questions, batched** | **7.022 / 6.115 s** | 10.845 / 10.910 s |

The device genuinely ran: `cuda [NVIDIA A40, cc 86, ptx sm_80]`, weights resident, routing counters
returned. It is about **1.8x slower** than the CPU kernel on this workload, in both repeats, with no
overlap between the arms.

## Why

Qwen3-0.6B is 1024 wide, so its projections are 1024x1024. That is far too small to fill an A40.
Each one is a kernel launch plus a host-device round trip, and at that size the transfer and launch
overhead exceed the arithmetic. GPU utilisation sat at 9-11% during the run -- the device waiting on
PCIe rather than computing.

This is consistent with the existing GPU campaign, which qualified this machinery on **27B-class**
models where projections are 5000+ wide and each launch carries real work. The campaign notes even
record "27B gates". The premise was carried to a 0.6B model without being re-checked, and it does
not hold there.

## Consequence for the decisions work

Projection-only offload does not rescue small-model latency. Helping at this size needs the whole
forward pass resident on the device -- attention, norms and KV included -- so there is no per-layer
round trip. That is a far larger piece of work than a kernel swap, and "attention on device" is
already on the unbuilt list.

The batched-tails win survives on both substrates (1.54-1.56x on CPU, 1.09-1.16x on GPU), which is
itself informative: batching helps least exactly where the device is already overhead-bound.

## Measurement hazards hit while producing this

Three contaminated runs were discarded rather than reported:

- The pod had no rust-ffm kernel at first, so the "CPU" arm would have been the pure-Java fallback
  and the GPU would have looked about 8x better than it is.
- Two and then three probe processes ran concurrently, because `pkill` kept missing a parent shell
  that then spawned fresh children. Killed by explicit PID and relaunched under `setsid`.
- Absolute CPU numbers here (8.03 ms/token) are worse than the dedicated Hetzner ccx33 (5.18) since
  this is a shared 96-vCPU host. Only the within-host comparison is used.

The probe throws when `-Ddecisions.kernel=cuda` is requested and the device is unavailable, rather
than falling back, and prints device, compute capability, PTX target and routing counters on every
run. Every one of today's retractions began as a silent fallback producing a plausible number.
