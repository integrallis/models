# Bounded training host lifecycle

- Provider: Vultr
- Instance ID: `33011d59-2d07-4aa4-b611-5d3baee78381`
- Label: `modeljars-alora-v3-a40-20260913`
- Region: `ewr`
- Plan: `vcg-a40-4c-20g-8vram` (NVIDIA A40 8 GiB, 4 vCPU, 20 GiB RAM)
- Created: `2026-09-13T02:37:39Z`
- List price at creation: USD 210/month, approximately USD 0.288/hour
- Deletion deadline: `2026-09-13T12:00:00Z`. The original `08:30:00Z` deadline was first extended
  for the frozen 1.7B v4 and v5 runs. It was extended again after v5 failed so one bounded,
  provenance-bound continuation diagnostic could run from the stronger v4 checkpoint. The final
  one-hour extension is for one frozen hard-negative/positive continuation and its fixed GPU
  qualification only. At USD 0.288/hour, deletion at the revised deadline remains below USD 2.72
  from instance creation and below the USD 4 ceiling.
- Cost ceiling tag: USD 4
- GPU preflight: NVIDIA A40-8Q, 8,192 MiB, CUDA 12.4, compute capability 8.6
- Training environment: Python 3.12.3, PyTorch 2.6.0+cu124, Transformers 4.53.3,
  PEFT 0.18.1, Accelerate 1.10.1, Safetensors 0.5.3
- Training started: `2026-09-13T02:47:41Z`, remote PID `17137`
- Remote inputs verified before start: manifest `c7da1f5f...`, train `3fe555c1...`,
  validation `f12c4c34...`, trainer `b0a2bbb6...`
- Training completed: `2026-09-13T06:14:00Z`; adapter SHA-256
  `437c6e1e098768d6d40e256044fdcbcd8d26f2e785f0b4734b9d1d56aa203106`
- Fixed 25-case diagnostic completed: `2026-09-13T06:17:00Z`; adapter rejected at 98.67% syntax,
  94.67% schema, 76% exact calls, and 32% false calls
- Evidence copied locally and hashed: training manifest `5b5dc554...`, five-case report
  `c427a4dc...`, 25-case report `aa2966ad...`, 25-case records `e2988bd0...`
- Qwen3 1.7B base screen completed before adaptation: 86% exact calls, 12% false calls, and 65.33%
  raw syntax/schema on the fixed 25-per-kind slice; report and records copied locally and hashed
- Qwen3 1.7B one-step preflight completed at `2026-09-13T06:31:00Z`: the exact training path,
  checkpointing-to-evaluation transition, and adapter serialization all passed
- Frozen Qwen3 1.7B rank-16 v4 training started at `2026-09-13T06:34:09Z`, remote PID `28577`,
  and completed at `2026-09-13T07:48:30Z`; adapter SHA-256
  `dcaeab65a51ebf9d98c97f7203e5890cb3e24ae592ba1f2358cbc1d8e46895bd`
- The fixed v4 diagnostic completed at `2026-09-13T07:51:05Z` and rejected the adapter at 100%
  syntax, 98.67% schema, 88% exact calls, and 24% false calls
- The provenance-bound 50/50 call/no-call v5 population was prepared and frozen under
  `../qwen3-17b-balanced50-v5/`
- Frozen Qwen3 1.7B balanced v5 training started at `2026-09-13T07:57:59Z`, remote PID `31962`
- Balanced v5 training completed at `2026-09-13T09:03:00Z`; adapter SHA-256
  `007aa5dca9892f146239de50d9c2f30d7e17a4bfe17df8150d78a7970a36f173`
- The frozen v5 diagnostic completed at `2026-09-13T09:05:00Z` and rejected the adapter at 98.67%
  syntax, 98.67% schema, 82% exact calls, and 20% false calls
- The provenance-checked continuation path passed a one-step serialization preflight
- Frozen Qwen3 1.7B abstention continuation v6 started at `2026-09-13T09:14:00Z`, remote PID
  `35535`
- V6 completed at `2026-09-13T09:27:00Z`; adapter SHA-256
  `ae7f4813aea4c8eb47b7d82999ff4b85bc086e21633acf3657ed7cead887c536`
- The frozen v6 diagnostic rejected it at 100% syntax, 100% schema, 18% exact calls, and 0% false
  calls
- State: no GPU job running while the two adapter endpoints are calibrated on disjoint development
  data; deletion confirmation pending
- Four interpolation points between the v4 and v6 endpoints were evaluated only on disjoint live
  BFCL development data. None met a viable call/abstention operating point; no interpolation was
  exposed to the frozen qualification set.
- The frozen hard-negative/positive continuation v8 started at `2026-09-13T10:00:00Z`, remote PID
  `3057`, after its trainer, merged splits, initial v4 adapter, and both manifests matched the
  hashes recorded in its preflight.
- V8 training completed at `2026-09-13T10:43:00Z`; adapter SHA-256
  `5a7541ef7074a76b3b8a8018172729cc7eb10518192218b04873dea723e83d94`.
- The fixed v8 diagnostic completed at `2026-09-13T10:50:00Z` and rejected the adapter at 100%
  syntax, 100% schema, 86% exact calls versus the base's 86%, and 20% false calls versus the
  base's 12%. The report, all 150 records, and training manifest were copied locally and their
  SHA-256 values verified.

The host was deleted at `2026-09-13T10:54:30Z`, after the v8 adapter identity and evaluation
evidence were copied and hashed. A provider-wide instance listing confirmed that exact instance ID
was absent. No billed host from this experiment remains. Stopping the instance would not have
counted as decommissioning.
