# LLaVA fidelity and decode follow-up — 2026-11-08

Written before provisioning or new inference. The rejected admission attempt remains immutable.

| Field | Value |
| --- | --- |
| Experiment | Layer-localize the three retained disagreements, profile short-answer decode, retain a general production improvement only after a matched A/B, then repeat frozen admission |
| Branch | `fix/llava-catalog-fidelity`, based on merged main e59dbc821af3ac3867be250f70d1865aa0e9cadd; each deployed payload records immutable revision and SHA-256 |
| Provider / region / plan | AWS us-east-1, one m6a.4xlarge, 16 vCPU / 64 GiB, 80 GiB gp3, Ubuntu 24.04 ami-0d6dd9e8265d48906 |
| Duration / spend | Expected 2 hours, maximum 4 hours; USD 6 total within existing qualification authorization |
| Creation / deletion deadlines | Create by 2026-11-08 05:00 UTC; delete by 2026-11-08 08:00 UTC and no later than four hours after launch; instance shutdown behavior terminate |
| Correctness | Original frozen manifest, independent reference, both public datasets, same weights/input tokens/image vectors; no gate changes |
| Performance | Same host, sequential candidate/reference arms, eight threads, original workload and warmup/repetition schedule; layer tracing is diagnostic and never timing evidence |
| Diagnostic order | Q4_K_M textvqa-38337 with identical reference embeddings first; then Q4_K_S textvqa-39587 and Q4_K_M vqav2-359592004; record per-layer final-prompt hidden states and full logits before selecting implementation changes |
| Acceptance of a fix | No case-specific branch or answer override; isolated operator fixture where possible; full original workload and short/long performance repeats before admission |
| Stop conditions | Corrupt inputs, nonfinite values, unsupported architecture, budget or deadline; archive failures and observations before termination |
| Secrets on box | No provider or service credentials; ephemeral SSH public key only |
| Local evidence | `audit-artifacts/pr380-vision-fidelity-2026-11-08/`; compact reproducible evidence committed under this experiment |

The live Standard EC2 quota was 192 vCPU and there were no pending, running,
stopping or shutting-down instances immediately before this record. Recheck on launch.
Only this experiment's instance, key and security group may be deleted. SSH is limited
to the controller IP. The root volume deletes on termination. No real inference,
model downloads or real-image preprocessing runs on the developer Mac.

The initial diagnostic reuses the measured ff627e32 JAR/kernel payload and retained
embeddings to reproduce the discrepancy. This is explicitly not a released-runtime
qualification. New implementation candidates and published 0.5.15 are identified by
separate receipts, never substituted into the original admission evidence.
