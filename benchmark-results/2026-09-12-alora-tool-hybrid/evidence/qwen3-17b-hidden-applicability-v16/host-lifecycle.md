# V16 feature-extraction host lifecycle

- Experiment: frozen BF16 feature extraction and affine-head fitting for
  `qwen3-17b-hidden-applicability-v16`
- Provider / region / exact plan: Vultr `ewr`, `vcg-a16-3c-32g-8vram`
- Hardware: three vCPU, 32 GB RAM, 170 GB disk, NVIDIA A16 8 GB vGPU, Ubuntu 24.04 x86-64
- Live availability at preflight: `ewr`, `atl`, `sjc`, `nrt`, `sgp`, and `blr`
- Live price at preflight: USD 0.236/hour
- Expected use: under two hours for setup, exact corpus reconstruction, batch-one feature
  extraction, CPU head fitting, evidence transfer, and teardown
- Maximum authorized spend: USD 0.71; three-hour hard ceiling
- Created no earlier than: `2026-09-14T01:17:53Z`
- Hard deletion deadline: `2026-09-14T04:17:53Z`
- State before creation: zero Vultr instances and zero Hetzner servers/resources
- Resolved image: Ubuntu 24.04 LTS x64, Vultr OS ID `2284`
- Verified SSH key: Vultr key `87e6aabf-dc20-48ea-993d-06e12e5d6941`, whose key material
  exactly matches local Ed25519 fingerprint
  `SHA256:fseILVYhZFEBXRXkKm3KPUaoat9GGIZiXnpb/WnYDuM`
- Exact V16 trainer revision: `5fd6d09a981e667176e0c324098b057bb4934598`
- Training-only contract: Transformers/PEFT execution may extract features and fit the small head;
  it is not a production inference path
- Correctness stop conditions: delete immediately if the source/model/adapter/version identity,
  representation, finite-value, save/reload, or frozen validation gate fails
- Prohibited work: Java qualification, generation, performance claims, sealed-case access, and any
  unrelated experiment
- Local evidence destination: this directory

The exact instance ID, address, host-key fingerprint, measured GPU/toolchain, output hashes,
deletion timestamps, cost, and final provider inventory are appended after those events occur.

- Created: `2026-09-14T01:19:14Z`
- Instance ID / label: `3c443bfe-310b-4321-a332-5f0af81da304` /
  `modeljars-alora-v16-a16-20260914`
- Address: `45.77.76.246`
- First observed SSH host-key fingerprint: Ed25519
  `SHA256:4SI7Nazbij57oCLZ8e1qlcI25GhOz6l4PVMKL+tM4vQ`, retained in the experiment-specific
  `known_hosts`; strict checking is enabled
- Measured GPU: NVIDIA A16-8Q, 8,192 MiB, compute capability 8.6, driver 550.90.07, CUDA 12.4
- Measured host: three single-thread Cascade Lake vCPUs, 31 GiB visible RAM, 8 GiB swap, 159 GiB
  root filesystem with 139 GiB initially available
- Exact host checkout: `b63a55a5d99b572310fa498d4e7df1f679f5d6ec`
- Environment gate: all 127 experiment tests passed; Python 3.12.3, Torch 2.6.0+cu124,
  Transformers 4.53.3, PEFT 0.18.1, Accelerate 1.10.1, and Safetensors 0.5.3 match the frozen
  environment; CUDA was live with no competing GPU process
- Watchdog: local launchd job `org.modeljars.alora-v16-watchdog` is armed to validate and delete
  only this exact instance at the hard deadline
- Corpus reconstruction: the historical V9 parent split, frozen hard negatives, and merged V13
  train/validation bytes reproduced every frozen SHA-256. The committed historical manifests were
  retained because current formatter diagnostics intentionally differ while the selected row bytes
  remain exact.
- Host CPU preflight: passed all 3,311 usable rows, exact tokenizer/decision alignment, deterministic
  objective checks, and zero overlap with the 735 static BFCL evaluation queries. Report SHA-256:
  `22d00cebf9797a35d2f04e868201aed3246a2345e55c92d63f7bbe397ffefebf`.
- Feature-extraction attempt 1: stopped before writing any feature row when deterministic PyTorch
  rejected cuBLAS without `CUBLAS_WORKSPACE_CONFIG`. The failed log/output path is retained. A
  regression test and fail-fast requirement now freeze `CUBLAS_WORKSPACE_CONFIG=:4096:8`; retry 2
  uses a new output path and changes no data, representation, model, or gate.
- Feature-extraction retry 2: completed all 2,840 training and 471 untouched validation rows with
  frozen cuBLAS determinism, BF16, SDPA, batch size one, and TF32 disabled. Train/validation feature
  SHA-256 values are `b2a385073021edc4fd6a6ba9d229610e2e8bb3e8e6e5a476df39e4624e1b2b29`
  and `dcead1d83a46c3b785eb07b59634c6957178e20ec9d0c6257a42b96a3613eea6`.
- Frozen affine-head result: selected `lambda=1e-5`; untouched validation passed with 389/390
  calls, 79/81 no-calls, and 0.986372269705603 balanced accuracy. Artifact file SHA-256 is
  `67b1f5fd231f924554d02daf450ce049fbccd24c084f7730d4ce2724ca234d1e`; canonical artifact
  SHA-256 is `acece5d8e0188b874ef8a919597ba7a359d876bc20fd828af5fbfab00984445a`.
- Evidence transfer: both feature tensors, manifests, head, result, successful logs, failed-attempt
  log, and host preflight were copied locally and independently hashed before deletion.
- Deleted: `2026-09-14T02:20Z`; the exact instance endpoint returned HTTP 404 and the provider-wide
  Vultr inventory returned zero instances. Runtime was about 1.02 hours; estimated charge was USD
  0.24 at the bound USD 0.236/hour rate.
- Cleanup: the launchd watchdog was unloaded; its exact plist and deletion script were moved to the
  macOS Trash and remain recoverable.
