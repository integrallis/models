# Vision performance closure

- Experiment: profile and replace the measured vision/decoder bottlenecks, preserving original weights and numerical gates.
- Branch: complete/pr380-vision-qualification; baseline fbd0d66f; freeze each deployed candidate commit and JAR hashes.
- Host: EC2 us-east-1a m6a.4xlarge, 16 vCPU/64 GiB, 80 GiB gp3.
- Expected duration: 1–2 hours; hard worker lifetime 4 hours; maximum spend $6.
- Creation: 2026-11-07T19:04:11.346462+00:00; deletion deadline: 2026-11-07T23:04:11.346462+00:00.
- Correctness: unchanged three-probe pixels/projected-vector gates; full decoder output checked against baseline and independent comparator.
- Performance: matched-host eight-thread runs, warmups and measured distributions. Target no material encoder slowdown (median ratio <=1.10) against PyTorch and matched end-to-end comparison before claiming decoder gap closed.
- Stop: terminate on completion, worker deadline or spend cap; preserve results before deletion.
- Secrets: ephemeral SSH public key only; no cloud/API credentials.
- Evidence: experiments/vision-qualification/results plus java-ai/audit-artifacts/pr380-vision-performance.
- Quota checked: 192 vCPU; zero instances holding standard quota at launch.

## Completed run

- Frozen candidate: `0f2bb7d6a4b30c6b5297d86456b9320388d86707`; all three probes and both decoder cache comparisons passed.
- Actual EC2 instance: `i-066d176fab20427fb`; launch `2026-11-07T19:04:15Z`.
- Final measurements: three warmups and seven measured runs, eight threads per arm, context 1024, 32 emitted greedy tokens.
- Evidence downloaded and all five verdicts recomputed successfully before termination.
- Archive: `java-ai/audit-artifacts/pr380-vision-performance/pr380-performance-evidence.tar.gz`.
- SHA-256: `dd6aa45f05741f3baea161a99451f03bbbd939e9313b43c46dc54722bc13319f`.
- Cleanup observation UTC: `2026-11-07T21:12:11.981887+00:00`; EC2 state confirmed `terminated`, ephemeral SSH private key removed, and cloud key-pair deletion confirmed (`Return: true`).
