# Qwen3 1.7B activated-adapter preflight

Status: **configuration frozen before the training result is known**.

- Base: `Qwen/Qwen3-1.7B`
- Base revision: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
- Purpose: replace independent-runtime routing with one base model, one tool adapter, and physically
  shared immutable KV-prefix blocks
- Trainer SHA-256: `02aa7335c936fec446a33b5a287d66195834a1f24132823b77e8b753e2238814`
  (the exact source is retained beside this preflight as `train_alora.py`)
- Prepared manifest SHA-256:
  `c7da1f5f826bb1c6f10047e09ff1e7a69e3346dad2336400e5a277ef937090b9`
- Train split SHA-256: `3fe555c1e4a68b65b6715341cd1d1cbf9995e549cbdb267f15896b0a9b7edf77`
- Validation split SHA-256:
  `f12c4c34c875d929b252d075749468f2c53d919744eca711e907927d9ecd83ca`
- Training selection: first 4,000 usable rows after deterministic preparation, including 760 no-call
  rows (19%) and 1,709 multi-call rows; 64 over-length rows encountered before the selection filled
- Validation selection: all usable validation rows
- Adapter: rank 16, alpha 32, zero dropout, all seven attention/MLP projection families
- Training: one epoch, learning rate `1e-4`, batch size 1, gradient accumulation 16, maximum length
  1,024, seed `20260912`, BF16, SDPA, gradient checkpointing with non-reentrant evaluation handoff
- Remote output: `/opt/modeljars-alora-v3/qwen17-r16-v4`
- Started: `2026-09-13T06:34:09Z`, remote PID `28577`

The 4,000-row bound is based on the measured 20.33-second optimizer step on the exact A40-8Q host:
250 steps plus full validation fit the recorded deletion deadline. It retains the prepared corpus's
19% no-call target. This is a qualification attempt, not a lowered gate: the fixed 100% syntax,
100% schema, at least 85% exact-call, no-worse-than-base, and at-most-5% false-call thresholds in
the parent protocol remain unchanged.

## Evidence before training

The exact one-step GPU preflight completed training, independent evaluation, and adapter
serialization. Its training-manifest SHA-256 is
`77bfa38a3066d634650c8878ff727f32a816827d98b22131accf7df304c503d4`; its deliberately
non-qualifying adapter SHA-256 is
`a0d36ab7d8a4baa2205f9da689d891c8619462631cafa5337f7a03a0e89e7d1e`.

The unadapted 1.7B base was screened first on the fixed 25-per-kind BFCL subset. It reached 86%
exact tool calls but made false calls on 12% of irrelevance cases; raw syntax and schema rates were
both 65.33%. This is promising enough to justify adapting but is not qualification. The report and
all 75 records are retained under `../qwen3-17b-base-screen/` with SHA-256 values
`1fc75ebc836e54da870b26d4a7ec5922b0558b73aa7fcd4c4a04f35aa8a42c30` and
`414c3863677f0008593c1e48e1e4099c68fd5f1808dddc883da222a44d6c00a7` respectively.

The real-weight Java gate also compares the Java-rendered tool prompt and every GGUF tokenizer ID
against the exact pinned Hugging Face tokenizer used for training. This prevents an adapter from
passing projection math while being invoked on a subtly different token sequence in production.
