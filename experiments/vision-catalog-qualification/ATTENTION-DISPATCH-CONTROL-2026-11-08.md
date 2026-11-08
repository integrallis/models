# Attention span dispatch experiment — issues 388 and 389

Prepared before provisioning. No qualification threshold changes.

| Field | Value |
| --- | --- |
| Hypothesis | Moving native ISA dispatch outside the attention row loops reduces per-token overhead without changing arithmetic or cache layout |
| Branch | perf/vision-attention-dispatch, based on main f0813f8102954a0bf3e12c639a1b852782531cb1; exact candidate commit recorded before deployment |
| Baseline | Retained vision runtime 3f31646c743647449310de2c5ff57fbd55d9e957, unchanged Java artifacts; only native library varies |
| Provider/region/plan | AWS us-east-1, one m6a.4xlarge, Ubuntu 24.04 ami-0d6dd9e8265d48906, disposable 80 GiB gp3 root |
| Expected duration | 30–45 minutes for setup and A/B/A; at most two hours including conditional complete qualification |
| Maximum spend | USD 2 total for this new worker; verify live compute price no greater than USD 0.70/hour before creation |
| Creation deadline | 2026-11-08 18:30 UTC |
| Deletion deadline | Two hours after creation, no later than 2026-11-08 20:30 UTC; independent shutdown timer and terminate-on-shutdown |
| Correctness | Synthetic scalar/AVX2 bitwise comparisons including empty/tail/padded spans, Linux tests/Clippy, identical real-model baseline/candidate tokens |
| Diagnostic performance | Same-host sequential baseline/candidate/baseline; caption and short-answer cohorts, three warmups/seven measurements; decode benefit must exceed baseline drift on both, no prefill regression beyond drift |
| Full qualification | Conditional on diagnostic benefit: original manifest, corpus, quantizations, reference revisions and F32/F16 arms; every latency ratio <=1.10 and exact-token agreement >=0.99, unchanged task/image gates |
| Fidelity follow-up | Preserve existing failures; collect identical-input reference/candidate traces for remaining disagreements within the worker deadline. Diagnostic traces cannot replace full admission |
| Stop | Setup failure, changed tokens for this arithmetic-preserving optimization, nonfinite output, no diagnostic benefit, cap or deadline. Remove losing production experiment |
| Secrets | Ephemeral SSH key only; no provider credentials on worker |
| Evidence | audit-artifacts/pr380-attention-dispatch-2026-11-08, compact reports committed after collection |

Verify active fleet/quota before launch and touch only this worker's resources. Download and SHA-256 verify evidence before termination; delete experiment key/security group and verify volume deletion. Local work is source inspection, compilation and synthetic tests only.

The prior row-pair experiment is rejected: caption decode ratios 1.0124–1.0149 and short decode 1.0286–1.0301 against bracketed baseline, identical tokens. It remains out of this branch. Task fidelity remains a separate unsatisfied gate; performance success alone cannot publish either model.
