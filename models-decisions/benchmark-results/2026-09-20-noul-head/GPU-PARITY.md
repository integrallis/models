# Q6_K device parity, verified on hardware

Run 2026-09-20 on Vultr `decisions-gpu-q6k-20260920`, instance
`c2c85dc0-679a-4dc5-a464-cc0061023ced`, `vcg-a16-3c-32g-8vram`, ewr.

| | |
| --- | --- |
| Device | NVIDIA A16-8Q, compute capability **8.6**, 8192 MiB, driver 550.90.07 |
| Host | Ubuntu 24.04.5, 3 vCPU, 31 GiB RAM, Temurin 25.0.4.1 |
| Toolchain | rustc nightly-2026-09-17, `ptxas` CUDA 12.0 V12.0.140 |
| Branch | `origin/fix/cuda-q6k-parity-defect` at `93b3feb9` (models PR #198) |

## Result

```
CudaQ6KDeviceParityTest > the two-super-block row that failed G1 is bit-exact again   PASSED
  ... bit-exact against the CPU control at 1, 2, 3, 32, 33, 48 super-blocks           PASSED
:backend-cuda:test  passed=41  skipped=0  failed=0   BUILD SUCCESSFUL
```

**`skipped=0` is the load-bearing part.** The device tests guard themselves with
`Assumptions.assumeTrue` and skip on a host with no CUDA device, so a green build proves nothing
on its own. Zero skips with seven device parity tests passing is what establishes the device was
actually exercised.

`ptxas -arch=sm_80` accepts the emitted module, closing step 0 of the GPU-host checklist (CU-007).
The PTX declares `.shared .align 4 .b8 models_cuda_scratch[1024]`.

## What this establishes, and what it does not

**Establishes:** the Q6_K device kernel, folded the way the CPU control folds it, is bit-exact
against that control at every width tried including the two-super-block row that failed G1 — and
on an **A16 (8.6)**, a different device from the A40 the defect was diagnosed on. The fix is not
specific to the machine that found the bug.

**Does not establish:** anything about the full G1 parity gate against a real model, which was not
run here, nor anything about throughput. No model was loaded.

## A caveat that matters for the energy plan

Installing `nvidia-cuda-toolkit` from the Ubuntu archive pulled NVML **580.178** against a loaded
kernel driver of **550.90.07**, and `nvidia-smi` now fails with a driver/library version mismatch.
The CUDA **driver API** is a separate library and is unaffected — the parity tests ran after the
toolkit install and used the device. But **NVML is how the pre-registration amendment proposes to
measure real watts**, so a future energy arm needs either a matched toolkit or a host image that
ships CUDA, and must verify `nvidia-smi` works *before* trusting any power figure.
