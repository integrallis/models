# Bounded V11 training host lifecycle

- Experiment: Qwen3 1.7B rank-32 short function-masked activated-LoRA
- Provider / region / plan: Vultr `ewr`, `vcg-a40-4c-20g-8vram`
- Hardware: NVIDIA A40 8 GiB vGPU, 4 vCPU, 20 GiB RAM
- Live price: USD 0.288/hour
- Shared host created: `2026-09-13T14:50:09Z`
- Instance ID: `07c712ce-78ba-4d7c-baf8-742098e65bfd`
- Label: `modeljars-alora-mask-v10-20260913`
- Maximum authorized spend across V10 startup and V11: USD 1.75
- Hard deletion deadline: `2026-09-13T20:45:00Z`
- Local identity-checked deletion watchdog: armed
- Input and code hashes: verified on host
- Pinned tokenizer retention preflight: passed with 11,827/979 usable rows
- Training started: approximately `2026-09-13T15:05:48Z`
- Training PID: `4373`
- Frozen command: `train_alora.py --prepared prepared --out run --candidate qwen3-1.7b
  --max-length 1024 --rank 32 --alpha 64 --epochs 1 --learning-rate 1e-4
  --gradient-accumulation 16 --seed 20260917`
- Startup verification: 740 total optimizer steps, matching V9; progress reached step 3 with
  6,565 MiB GPU memory in use and an approximately 2 hour 45 minute remaining estimate

The instance is reused only because V10 stopped at its first optimizer step without candidate
evaluation. It remains subject to the original cumulative cost ceiling and hard deadline. It must
be deleted rather than stopped after V11 evidence is copied or when the deadline is reached.
