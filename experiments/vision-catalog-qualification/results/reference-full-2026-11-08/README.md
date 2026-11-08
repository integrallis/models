# Frozen reference-arithmetic qualification, 2026-11-08

Measured runtime: `78528c14d640da2947a60cad7be4e4ba4f10228b`. Native SHA-256:
`fa7d6984fa05a2a47932168c605b3695837a4dd15efee1806a3a7e41271eeed7`.
The frozen manifest, corpus, thresholds, pinned external reference and execution settings
are unchanged. Each variant has its own m6a.4xlarge worker and complete same-host
candidate/F32/F16 evidence; these are not a fabricated combined-host receipt.

Both variants match every one of the 80 cases against the fresh F32 reference,
including all six earlier regressions and all blank-image controls. The scored
nonblank token agreement is 64/64 for each. Task scores equal the reference.

| Variant | Backend admission | VQAv2 | TextVQA | Worst task decode ratio | Confirmation F16 decode |
| --- | --- | --- | --- | --- | --- |
| Q4_K_S | PASS | 0.687500 | 0.415625 | 1.081988 | 1.083162 |
| Q4_K_M | FAIL | 0.675000 | 0.384375 | 1.120166 | 1.050024 |

Q4_K_S passes every frozen backend gate. Q4_K_M fails only the strongest-reference
performance gate: `textvqa-37476` has an F16 decode latency ratio of 1.120166 against
the 1.10 maximum. Its prefill and total latency pass, as do every other task and the
longer integration confirmation. This failed result is retained without changing the gate.
The initial M integration F16 decode miss is also retained. The original finalizer
uses the preplanned 3-warmup/7-repeat confirmation for admission; that policy was not
changed after seeing these results.

These are backend admission results, not catalog publication or released-runtime
consumer verification. Q4_K_M needs a measured runtime improvement and a fresh full
qualification. The separate GEMV experiment is not part of this payload or verdict.
The public ModelJars API, generated demo and subsequent Central-only release repeat
remain distinct checks. Measurements cover Linux x86-64 with AVX2/FMA/F16C; they do
not qualify other CPU/OS implementations.

Per-variant JSON files retain the final attempt, task comparison, immutable receipt,
initial and confirmation decoder reports, raw reference outputs, and resource data.
`evidence-sha256.json` binds these files. Full raw candidate/encoder results and all
worker diagnostics are collected in the owned fleet's checksummed archive before its
hard deletion deadline. The source runtime is available through the parent history.
