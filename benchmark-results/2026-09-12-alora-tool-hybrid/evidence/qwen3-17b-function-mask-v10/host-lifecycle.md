# Bounded V10 training host lifecycle

- Experiment: Qwen3 1.7B rank-32 function-masked activated-LoRA
- Intended provider / region / plan: Vultr `ewr`, `vcg-a40-4c-20g-8vram`
- Hardware: NVIDIA A40 8 GiB vGPU, 4 vCPU, 20 GiB RAM
- Maximum authorized spend: USD 1.75
- Correctness stop condition: reject immediately if identity or the fixed development smoke fails
- Performance work: prohibited on this GPU host
- Local evidence destination: this directory
- State before creation: Vultr account inventory returned zero instances
- Created: `2026-09-13T14:50:09Z`
- Instance ID: `07c712ce-78ba-4d7c-baf8-742098e65bfd`
- Label: `modeljars-alora-mask-v10-20260913`
- Initial provider state: `pending`
- Hard deletion deadline: `2026-09-13T20:45:00Z`
- Environment installation completed with PyTorch `2.6.0+cu124`, Transformers `4.53.3`, PEFT
  `0.18.1`, Accelerate `1.10.1`, and safetensors `0.5.3`
- V10 training PID: `3442`
- Startup gate: aborted after step 1 of 681 because long opaque identifiers reduced usable rows by
  approximately 8%; PID termination was verified and no evaluation ran
- Host reuse: retained under the same cost ceiling and deletion deadline for the separately frozen
  V11 short-identifier experiment

The host must be deleted, not stopped, after evidence is copied or at the hard deadline. A
provider-wide inventory check must confirm absence.
