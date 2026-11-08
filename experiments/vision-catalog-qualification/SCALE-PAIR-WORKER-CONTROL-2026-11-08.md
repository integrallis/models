# Bounded follow-up worker for the remaining decode gap

| Field | Value |
| --- | --- |
| Experiment | Compare the paired Q4_K half-scale conversion against the retained production runtime; repeat frozen admission only after a measured benefit |
| Baseline | Runtime `3f31646c743647449310de2c5ff57fbd55d9e957`; exact archived JARs and Linux native library |
| Candidate | `b782797dcc566a006ac1b54db6b5c31feff9a11e`; Java sources unchanged from baseline, Rust conversion pairing and its instruction-count test only |
| Provider / region / plan | EC2, us-east-1, one m6a.4xlarge, 16 vCPU / 64 GiB, 80 GiB gp3 |
| Expected duration | About 20 minutes for setup and diagnostics, then about 100 minutes if the complete repeat is justified |
| Maximum follow-up spend | USD 2.00 including estimated storage/transfer; cumulative declared campaign envelope USD 8.00 |
| Creation deadline | 2026-11-08 08:20 UTC |
| Deletion deadline | 140 minutes after launch, no later than 2026-11-08 10:40 UTC; independent root-owned shutdown terminates the instance |
| Correctness | All synthetic tests and Clippy; every A/B token identical; then all original accuracy, image-contribution, numerical and exact-sequence gates |
| Performance | Sequential baseline/candidate/baseline, 3 warmups / 7 repetitions for caption and original short-answer diagnostic; benefit must exceed bracketing baseline variation; unchanged full admission ratios at most 1.10 |
| Stop conditions | No benefit, output difference, incomplete native setup, nonfinite output, deadline or spend bound; preserve raw failed evidence and remove the worker |
| Secrets | No cloud/provider/service credentials on the instance; ephemeral SSH public key only |
| Local evidence | `audit-artifacts/pr380-scale-pair-2026-11-08/` |

All preceding workers and their keys/security groups/temporary volume were deleted
after archive verification. Re-read the live EC2 quota and active fleet before
provisioning; do not touch resources belonging to another task. SSH is restricted
to the controller IP. Root volume deletion on termination remains enabled.

The ordinary remote `ubuntu` user performs setup and measurements after the root
bootstrap; no interactive `sudo`. Do not run inference or download model weights
on the Mac. Record the actual native/JAR hashes, compiler and host alongside each
arm. No accepted gate or existing published model changes under this control.

The budget bounds this additional, explicitly recorded follow-up; it does not
retroactively alter the previous experiment's receipts or permit indefinite retries.
The release dry run remains a library check, and does not qualify these candidates.
