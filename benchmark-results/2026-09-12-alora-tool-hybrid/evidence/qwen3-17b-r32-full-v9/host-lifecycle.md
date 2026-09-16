# Bounded v9 training host lifecycle

- Experiment: Qwen3 1.7B rank-32 full-corpus activated-LoRA
- Provider / region / plan: Vultr `ewr`, `vcg-a40-4c-20g-8vram`
- Hardware: NVIDIA A40 8 GiB vGPU, 4 vCPU, 20 GiB RAM
- Live list price: USD 210/month, approximately USD 0.288/hour
- Maximum authorized spend: USD 1.75
- Creation not before: `2026-09-13T11:12:17Z`
- Hard deletion deadline: `2026-09-13T17:00:00Z`
- Correctness stop condition: reject immediately if training identity, development syntax/schema,
  exact-call, or irrelevance gates fail
- Performance work: prohibited on this GPU host; a passing candidate must be packaged and tested
  through the JVM on a separate clean CPU host
- Local evidence destination: this directory
- State before creation: Vultr account inventory returned zero instances
- Created: `2026-09-13T11:13:13Z`
- Instance ID: `04543d5f-f823-4d62-b9e3-e89a3950642b`
- Label: `modeljars-alora-r32-v9-20260913`
- Initial address: `45.76.14.91`
- Provider state after creation: `active`
- Full-corpus training started: `2026-09-13T11:19:11Z`
- Training PID: `3861`
- Frozen training command: `train_alora.py --prepared prepared --out run --candidate qwen3-1.7b
  --max-length 1024 --rank 32 --alpha 64 --epochs 1 --learning-rate 1e-4
  --gradient-accumulation 16 --seed 20260917`
- Startup verification: process remained live, CUDA reported `5955` MiB in use, and progress reached
  step 9 of 740 with an approximately 2 hour 42 minute remaining estimate at
  `2026-09-13T11:21:35Z`
- Qualification evaluator uploaded and hash-verified at
  `d60e5f96f0dd3c65f0dc56001dda803026d7950dac92493670643ff5b9415e14`; it resolved the frozen
  manifest to exactly 300 cases and reproduced case-set SHA-256
  `94c951aa277cee1e3f5ba454323e28b81cbfa357eb3d25d23a2e6ca581d1f132` and manifest SHA-256
  `5fee70a0d18801ac9541b2cfa3b504c0fd06e98f5bf1adeb196adfc89dffa1d2`
- Training and validation completed at approximately `2026-09-13T14:08:40Z`; the adapter weights
  are 139,512,976 bytes with SHA-256
  `f1a4a1bc15e3942ed5a3fbe2c4750f39a8b24804e20566242ec740c5ce851c21`, and the training manifest
  SHA-256 is `d0620df4860c879f6d3f6e5573168bb08afc0b48d954040825e1317972f7de47`
- The fixed 25-case-per-kind smoke completed at approximately `2026-09-13T14:13:38Z` and rejected
  the candidate: 98.67% syntax, 98.67% schema, 88% exact tool calls versus 86% for the base, and
  24% false tool calls versus the 5% ceiling and 12% for the base
- Copied evidence was hash-verified against the host: report
  `8243d38cb501a2d4c93dcb22056e0a0245964cb5dd81f6cf2af12c7c8299e804`, records
  `944d137ba1e325e0f3daa922a73108de0da5ccf0292bcf4dbe4a565aa11fcb4c`, training log
  `8574fa81951206305fba3a6afb4317d9daaa2ce06a355204b95d1c91d59ca063`, and preparation manifest
  `c7da1f5f826bb1c6f10047e09ff1e7a69e3346dad2336400e5a277ef937090b9`
- Instance deletion requested and absence verified at `2026-09-13T14:14:22Z`; a provider-wide
  inventory then returned zero Vultr instances
- The local hard-deadline watchdog was unloaded after provider absence was verified

The instance was deleted rather than merely stopped. No GPU resource from this experiment remains.
