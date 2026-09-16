# Bounded V12 policy-screen host lifecycle

- Experiment: unchanged Qwen3 1.7B V9 adapter with an applicability system contract
- Provider / region / exact plan: Vultr `ewr`, `vcg-a40-4c-20g-8vram`
- Hardware: NVIDIA A40 8 GiB vGPU, 4 vCPU, 20 GiB RAM
- Live list price: USD 210/month, approximately USD 0.288/hour
- Expected use: 15–25 minutes including prompt parity and two 75-case screens
- Maximum authorized spend: USD 0.58
- Creation not before: `2026-09-13T18:46:59Z`
- Hard deletion deadline: `2026-09-13T20:46:59Z`
- Correctness stop condition: delete immediately on prompt/token-parity failure or exposed-gate failure
- Performance work: prohibited; this host is only for deterministic V12 correctness screens
- Local evidence destination: this directory
- State before creation: Vultr inventory returned zero instances
- Resolved plan availability: A40 8 GiB in `ewr`; Ubuntu 24.04 image `2284`
- Verified SSH key: `modeljars-qualification-20260830`, whose public material matches the local
  Ed25519 public key
- Created: `2026-09-13T18:47:29Z`
- Instance ID: `3b1d8bef-2747-4a13-93c2-999ced304cde`
- Label: `modeljars-alora-policy-v12-20260913`
- Address: `66.135.7.58`
- Provider state after creation: `active`, `running`, and `ok`
- First observed SSH host key: Ed25519 SHA-256 fingerprint
  `O8EPbus3ximUkOPIr6BZwWdddQFctyR+m1OH/5wBrR0`, recorded in this evidence directory's
  experiment-specific `known_hosts`; no global host-key bypass was used
- Exact evaluator environment: Python 3.12.3, Torch 2.6.0+cu124, CUDA 12.4, Transformers 4.53.3,
  PEFT 0.18.1, Safetensors 0.5.3, Accelerate 1.10.1
- Upload verification: policy, evaluator, oracle generator and helper, V9 adapter and training
  manifest, frozen window manifest, and all five pinned BFCL files matched their local SHA-256
  identities before use
- Prompt parity: accepted `policy-oracle-v2` matched the Java Qwen3 1.7B Q8_0 path for all 1,072
  prompt bytes and all 228 token IDs; activation tokens remained `[151644, 77091, 198]`
- Exposed screen: rejected at 42/50 adapter exact calls, 5/25 adapter false calls, 75/75 syntax,
  and 75/75 schema; the same-policy base reached 45/50 exact calls and also made 5/25 false calls
- Copied evidence was hash-verified against the host: report
  `07a1a7562f1aee34d022147680cd9b2eba164a8ee90ab5ff0a3a210c7513a3bc`, records
  `4b800809cbb7bd427c8649a2c35635dcb93ccee39d04ab75a3cbb9b2d9cda0fc`, and raw log
  `d9e9c6b26b398cf1285ee99737f901da1a994a7a8403c456c884f3c8bb0fc74a`
- Deleted: `2026-09-13T19:07:25Z`, immediately after exposed rejection and evidence verification
- Approximate lifetime / spend: 20 minutes / USD 0.10 at the listed hourly rate
- Provider verification: the exact instance returned HTTP 404 and the account-wide Vultr inventory
  returned zero instances after deletion
- Watchdog: unloaded after provider absence was verified; its exact script, plist, and empty logs
  were removed

The live-development and sealed 300-case qualification sets were not opened because the exposed
gates failed. The instance was deleted rather than stopped; no Vultr resource remains.
