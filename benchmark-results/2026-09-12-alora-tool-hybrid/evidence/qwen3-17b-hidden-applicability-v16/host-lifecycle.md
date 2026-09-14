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
- Exact Models revision at creation: `5fd6d09a981e667176e0c324098b057bb4934598`
- Training-only contract: Transformers/PEFT execution may extract features and fit the small head;
  it is not a production inference path
- Correctness stop conditions: delete immediately if the source/model/adapter/version identity,
  representation, finite-value, save/reload, or frozen validation gate fails
- Prohibited work: Java qualification, generation, performance claims, sealed-case access, and any
  unrelated experiment
- Local evidence destination: this directory

The exact instance ID, address, host-key fingerprint, measured GPU/toolchain, output hashes,
deletion timestamps, cost, and final provider inventory are appended after those events occur.
