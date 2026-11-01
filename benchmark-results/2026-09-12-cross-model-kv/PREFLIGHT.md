# Qwen3 cross-model KV qualification preflight

- Experiment: complete the Qwen3 0.6B to 1.7B cross-model KV transfer ladder: paper-scale
  calibration, held-out mapping fidelity, fail-closed Java cache import, task correctness, TTFT,
  throughput, and memory.
- Provider / region / exact plan: Vultr `ewr`, `vcg-a40-4c-20g-8vram`; four vCPU, 20 GiB
  RAM, 360 GB local storage, and one 8 GiB NVIDIA A40 slice. The live API listed only `ewr` for
  this plan.
- Image: Vultr Ubuntu 24.04 LTS x64, image ID `2284`, resolved from the live API immediately
  before creation.
- Price verified: USD 0.288/hour from the live plan API on 2026-09-12. Network transfer is
  additional.
- Maximum authorized spend: USD 4.00 total; delete earlier on completion or a decisive gate
  failure.
- UTC creation deadline: 2026-09-12T16:40:00Z.
- UTC deletion deadline: 2026-09-13T02:22:04Z.
- Instance: Vultr `f7e4869a-a5d0-4efa-bab7-0f857954a875`, label
  `modeljars-kv-a40-20260912`, requested at 2026-09-12T16:22:39Z. Its address and hardware
  readiness must be recorded after provisioning completes.
- Ready at 2026-09-12T16:27:22Z: `64.176.196.146`, SSH ED25519 fingerprint
  `SHA256:a6UDyPCGt01hQzILTqwBQDz9omRhcTCka3zCBdA8qKI`, Ubuntu kernel 6.8.0-138, four
  Cascade Lake vCPU, 19 GiB visible RAM plus 8 GiB swap, 338 GiB filesystem with 310 GiB free,
  and NVIDIA A40-8Q compute capability 8.6 with 8,192 MiB VRAM, driver 550.90.07/CUDA 12.4.
- Calibration toolchain preflight found that the pinned `vllm==0.26.0` wheel installs
  `torch==2.11.0+cu130`. That wheel cannot initialize CUDA through the provider's fixed
  550.90.07 vGPU guest driver (`found version 12040`). Replacing a vGPU guest driver without the
  provider's matching host driver is not a valid remedy. The vLLM environment is retained only as
  a record of the failed compatibility check; it will not produce evidence. Calibration exports
  use a separate pinned PyTorch CUDA 12.4 environment and the standard Hugging Face model forward
  pass. External runtimes remain calibration/oracle tools only; production inference is Java.
- The first anonymous FineWeb-Edu streaming request resolved the pinned dataset files but was
  rejected by Hugging Face's content endpoint with HTTP 429. No calibration data was produced.
  The retry uses the project's read-only fine-grained Hugging Face token without printing or
  persisting it in evidence, while retaining the same immutable dataset revision and split
  parameters. This is a reproducibility and qualification-infrastructure gotcha, not a model
  result.
- The authenticated retry wrote and hash-verified all 600 requested sequences (500 calibration,
  50 mapper validation, 50 final test) at the pinned dataset revision. During interpreter
  shutdown, PyArrow then aborted in `PyGILState_Release`; because that happened after the writer's
  completion message, the two output files were independently parsed, counted, and SHA-256 hashed
  before use. Their hashes are retained with the evidence rather than treating process exit alone
  as proof of a valid dataset.
- Correctness gate: mapper identity and artifact hashes match; cache import fails closed; held-out
  K/V fidelity is reported; native-vs-transferred logits, perplexity, long-context retrieval,
  tool-call validity, argument correctness, and response correctness all meet predeclared gates.
- Predeclared downstream thresholds: at least 95% of the native target's normalized HellaSwag
  accuracy; mapped/native WikiText perplexity ratio at most 1.10; 100% retention of native-correct
  exact answers on synthetic long-context key/value and UUID retrieval; and no loss of native-valid
  tool selection or required arguments. Shorter prompts that do not cross the measured performance
  break-even point must use native target prefill and are not allowed to enter the mapped path.
- Performance gate: transferred-prefix TTFT must beat native target prefill at a declared context
  tier after including mapping and handoff time; report cold and warm paths separately.
- Local evidence destination:
  `benchmark-results/2026-09-12-cross-model-kv/`.
- Stop condition: evidence copied locally and hashes verified, any correctness gate fails
  decisively, no performance crossover exists in the measured context range, or the spend/deletion
  deadline is reached.

The host must be deleted and all experiment-owned Vultr resources inventoried before this run is
closed.

## Superseded AWS attempt

The first preflight selected AWS `us-east-1` `g5.8xlarge` at USD 2.448/hour with an USD 8.50
ceiling. Creation failed before an instance existed because the account's G-family vCPU quota was
zero (`VcpuLimitExceeded`). The temporary security group and key-pair records were deleted, and a
follow-up inventory confirmed zero matching instances or volumes. No AWS compute charge was
incurred.

The next attempt created Vultr A16 instance `757bf971-ce87-4e3b-82b2-cf3a83646434` in `blr` at
2026-09-12T16:10:40Z. Installation completed, but the instance remained powered off and two exact
API start requests timed out; port 22 never became reachable. The temporary instance was deleted
at approximately 2026-09-12T16:21Z, and the follow-up inventory confirmed zero Vultr instances.
The smaller A40 plan was chosen only after its live price, sole location, 360 GB local disk, image,
and matching SSH public key were revalidated. Its 8 GiB device requires sequential source and
target trace collection; it is not sufficient evidence for a two-model concurrent deployment.
