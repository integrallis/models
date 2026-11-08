# Full normalization-only attempt

Runtime `54ed1649a6e17fec119083242a4843d3d9a49c98` on the original frozen corpus and AUTO reference. Neither quantization qualifies. Q4_K_S loses 1.25 VQA percentage points against its paired reference; both variants match 62/64 nonblank token sequences. The faster-reference decode ratios and caption confirmation also include failures. Every gate remains unchanged.

The follow-up combines head-major cache storage, fixed Q4 block views and the independently verified QuickGELU operation-order correction. Its full corpus repeat is separate evidence and cannot overwrite this attempt.

The local raw archive under `audit-artifacts/pr380-vision-fidelity-2026-11-08/` contains all raw measurements, vectors, images, diagnostics, worker scripts and invalid-payload logs. `archive-receipt.json` records the verified 2,228,879,348-byte archive SHA-256. It excludes model weights, environment caches and credentials. Later arithmetic diagnostic combinations have a separate supplemental archive.
