# Rejected Q6 fixed-block-view experiment

Source 812f6eda, against retained 3f31646c native kernels with the identical head-major Java decoder. One m6a.4xlarge, eight threads, original/candidate/original; three warmups and seven measured caption generations per arm. All 30 token sequences match; 63 native tests and Clippy pass.

Decode medians are 3854.500 / 3859.642 / 3859.894 ms. Ratios 1.001334 and 0.999935 show no meaningful improvement. Exclude the change from production. Prefill variation is disclosed in the raw results and is not evidence of decode acceleration. This is a component diagnostic, not model admission.

The supplemental archive receipt is retained in the adjacent attention-arithmetic-2026-11-08 directory. It includes the experiment source, binaries, payloads and all failures.
