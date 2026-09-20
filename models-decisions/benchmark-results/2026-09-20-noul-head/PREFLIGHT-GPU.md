# Preflight control record — GPU parity host

Written 2026-09-20T13:27:12Z, before any resource was created.

```text
Experiment:              Re-run the Q6_K device parity gate against the corrected CPU contract
                         (vectors#76 + models#200), then the full G1 parity on Granite 4.1 3B,
                         then the decisions GPU arm with NVML power sampling.
Provider / region / plan: Vultr, ewr, vcg-a16-3c-32g-8vram
                         NVIDIA A16 (GA107), 8 GiB VRAM, 3 vCPU, 32 GiB RAM
                         Ubuntu 24.04 LTS x64 (image 2284)
Compute capability:      8.6, against the sm_80 floor CudaDriver enforces
                         (MINIMUM_COMPUTE_CAPABILITY = 80). AWS g4dn was rejected on this
                         ground: T4 is 7.5 and the driver would refuse it.
Expected hourly charge:  0.236 gross (live API read 2026-09-20T13:27:12Z); billed hourly
Maximum authorized spend: USD 2.00 (6 h at this rate is about USD 1.42)
UTC creation deadline:   2026-09-20T13:27:12Z
UTC deletion deadline:   2026-09-20T19:27:12Z  (hard)
Correctness gate:        CudaQ6KDeviceParityTest green against the one-fma kernel; then G1 parity
                         bit-exact against the single canonical CPU answer
Performance gate:        none on this host for the parity work. The decisions GPU arm, if reached,
                         inherits the pre-registered gates unchanged.
Local evidence dest:     projects/models-decisions/models-decisions/benchmark-results/2026-09-20-noul-head/
SSH key:                 Vultr 87e6aabf-dc20-48ea-993d-06e12e5d6941, verified by comparing the
                         public key material against the local id_ed25519.pub, not by its label
Known environmental step: ptxas ships with the CUDA toolkit and Vultr's images are plain Ubuntu,
                         so the toolkit is installed on boot. Recorded as CU-007, not a defect.
No external inference runtime will be installed or used.
```

## Created resources (record immediately)

| Resource | ID | Detail |
| --- | --- | --- |
| Vultr instance | `c2c85dc0-679a-4dc5-a464-cc0061023ced` | `decisions-gpu-q6k-20260920`, `vcg-a16-3c-32g-8vram`, ewr |

Tagged `delete-by-2026-09-20T19-27-12Z`. Vultr has no separate firewall resource attached here;
teardown is the instance alone, confirmed absent by its exact id afterwards.

## Decommission

Deleted 2026-09-20, after the label was validated against the expected one rather than the id
alone. Post-delete audit: the exact instance id returns HTTP 404, the account holds zero
instances and zero block-storage volumes. Lifetime about 35 minutes at 0.236/hr, roughly USD 0.14
against the USD 2.00 ceiling. Evidence copied to `GPU-PARITY.md` before deletion.
