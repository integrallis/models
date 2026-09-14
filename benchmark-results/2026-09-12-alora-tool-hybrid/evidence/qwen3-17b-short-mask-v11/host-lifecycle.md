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
- Exposed-smoke continuation PID: `32418`
- Continuation behavior: wait for training PID `4373` to exit; run the frozen 25-per-kind smoke
  only when `run/adapter/adapter_model.safetensors` and `run/training-manifest.json` both exist;
  never open the sealed 300-case window automatically
- Frozen command: `train_alora.py --prepared prepared --out run --candidate qwen3-1.7b
  --max-length 1024 --rank 32 --alpha 64 --epochs 1 --learning-rate 1e-4
  --gradient-accumulation 16 --seed 20260917`
- Startup verification: 740 total optimizer steps, matching V9; progress reached step 3 with
  6,565 MiB GPU memory in use and an approximately 2 hour 45 minute remaining estimate
- Live verification at `2026-09-13T16:00:40Z`: step 272/740, training PID still live, 6,755 MiB
  GPU memory in use, and an approximately 1 hour 28 minute remaining estimate
- Training completed at approximately `2026-09-13T17:36:41Z`: 740/740 optimizer steps,
  training loss `0.09141710738877992`, validation loss `0.04099120944738388`
- Adapter SHA-256: `f3238e4f9b997c643dc9c446d8554280be4bc14987445ab74dadbc30a35c9128`
- Training manifest SHA-256: `7c3572d7ee00042ee969de940533943b81e5e4b4422441f7f6784e303dec880c`
- Fixed exposed smoke: failed syntax, schema, and false-call gates; hidden set never opened
- Evidence copied locally and verified byte-for-byte before deletion
- Instance deletion requested and confirmed absent (`HTTP 404`) at `2026-09-13T17:44:25Z`
- Local launchd watchdog unloaded and its temporary script, plist, and log removed after deletion
- Provider inventory check: zero Vultr instances with the `modeljars-alora` label prefix

The instance is reused only because V10 stopped at its first optimizer step without candidate
evaluation. It remains subject to the original cumulative cost ceiling and hard deadline. It must
be deleted rather than stopped after V11 evidence is copied or when the deadline is reached. That
deletion is complete; the instance no longer exists.
