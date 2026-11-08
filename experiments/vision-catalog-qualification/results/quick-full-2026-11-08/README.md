# Complete retained-runtime repeat: not admitted

Measured runtime: `3f31646c743647449310de2c5ff57fbd55d9e957`, containing double
CLIP moments, upstream QuickGELU operation order, head-major F32 causal caches and
fixed Q4 block views. The frozen corpus, thresholds and original AUTO reference
remain unchanged. Both quantizations completed all 80 candidate/reference cases,
both integration schedules, and the separate faster F16 performance cohort.

| Variant | VQA, ours / reference | TextVQA, ours / reference | Exact sequences | Worst decode ratio |
| --- | --- | --- | --- | --- |
| Q4_K_S | 68.75% / 68.75% | 43.4375% / 41.5625% | 63/64 | 1.1081989206 |
| Q4_K_M | 67.5% / 67.5% | 38.4375% / 38.4375% | 63/64 | 1.1031121761 |

Both pass accuracy loss, dataset floors, image contribution, F32 task timing and
the longer integration confirmation. The confirmation decoder ratio against F16
is 1.0671287243 with identical greedy tokens. Each variant fails the unchanged
99% sequence gate and one of four short-answer decode ratios against the 1.10
limit. Every other prefill, decode and total-latency bucket passes.

The remaining sequence differences are Q4_K_S `textvqa-36979` (ours `Lugnuts`,
reference `Lug nuts`) and Q4_K_M `textvqa-38337` (ours `Movies`, reference
`Electronics`). The first earns greater VQA soft credit in this dataset; neither
answer in the second earns credit. No answer or token normalization was added to
the exact-sequence comparison. These are bounded subset scores, not publisher
benchmark reproductions.

The formal report is `campaign/results/admission-attempt.json`; both
`backendQualified` flags remain false. No catalog or released-runtime qualification
is claimed. The user has been asked whether the previously approved GLiNER-style
task-fidelity policy should also apply here; no answer or policy change is recorded
by this attempt. Performance thresholds remain unchanged regardless of that question.

This directory retains compact raw task inputs/outputs, timing distributions,
configuration, resource logs and both integration reports. The checksummed full
archive additionally contains images, arrays, executable payloads, sources and
worker logs; GGUF weights, caches and credentials are excluded. Its local location
is `audit-artifacts/pr380-vision-repeat-2026-11-08/pr380-repeat-raw.tar.gz`.
`pr380-repeat-raw.tar.gz.json` records its 1,252,663,513 bytes and SHA-256.

The archive and formal reports were verified before deleting the instance, key,
security group and temporarily retained root volume at 08:00:12 UTC. The temporary
volume safeguard did not extend the compute lifetime or interrupt measurements.
