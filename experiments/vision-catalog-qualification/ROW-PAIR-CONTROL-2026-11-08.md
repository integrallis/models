# Q4 row-pair experiment — issue 388

Prepared before provisioning. The experiment is separate from Models 0.5.16
publication and does not admit a model or change qualification gates.

| Field | Value |
| --- | --- |
| Experiment | Share activation loads across two Q4_K decode rows; preserve each row's reduction order |
| Branch and source | `experiment/llava-q4-row-pair`, runtime `dcfca48a55ee6f2f73ded71522de6c564a3330ac` |
| Baseline | Retained runtime `3f31646c743647449310de2c5ff57fbd55d9e957`; same measured Java JARs in both arms |
| Provider and plan | AWS us-east-1, one m6a.4xlarge, Ubuntu 24.04 AMI ami-0d6dd9e8265d48906, disposable 80 GiB gp3 root |
| Duration | Expected 30 minutes for diagnostic; at most two hours including conditional full qualification and collection |
| Spend limit | USD 2 for this worker, within the existing USD 8 fidelity follow-up envelope; On-Demand compute price verified through AWS Pricing at USD 0.6912/hour |
| Creation deadline | 2026-11-08 16:00 UTC, after issue 387 publication verification |
| Deletion deadline | Within two hours of creation and no later than 2026-11-08 18:00 UTC; shutdown behavior terminate, independent automatic deadline |
| Correctness | Native tests/Clippy on Linux; all baseline/candidate token sequences identical; no nonfinite outputs |
| Diagnostic performance | Same-host sequential baseline/candidate/baseline, three warmups and seven measurements, caption and short-answer cohorts; candidate decode gain must exceed baseline drift in each cohort, with no prefill regression beyond baseline drift |
| Full qualification | Only after diagnostic benefit is confirmed: unchanged frozen manifest, both quantizations, original corpus and both reference precision arms, all median latency ratios at most 1.10 and original accuracy/token gates |
| Stop conditions | Failed setup, corrupted input, output mismatch, nonfinite output, failed benefit gate, spend cap or deadline; retain diagnostic evidence before termination |
| Secrets | Ephemeral SSH public key only; no service credentials on the worker |
| Evidence destination | `audit-artifacts/pr380-row-pair-2026-11-08/`, then compact reports committed to the repository |

The payload records hashes of every input and executable. Recheck live fleet and
quota before provisioning. Modify only resources created for this experiment.
Download and verify the evidence archive before deleting the instance, its key,
security group and root volume. The Mac performs compilation and synthetic tests
only. No model downloads, inference, oracle execution or image processing locally.

Local validation: 63 native tests passed, three existing tests ignored, on an
AVX2-capable x86 host; Clippy passed with warnings denied. The worker-range fixture
now covers full, odd-offset, one-row and empty ranges, checking untouched output
sentinels and bitwise parity across batch sizes. These tests establish synthetic
correctness and do not establish a performance gain.
