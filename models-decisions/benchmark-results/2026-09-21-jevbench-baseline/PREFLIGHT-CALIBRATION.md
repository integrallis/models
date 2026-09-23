# Preflight — held-out calibration run

Written 2026-09-21T09:22:49Z, before any resource was created.

```text
Experiment:        Score JevBench easy (48) + hard (111) with the same shared-prefix candidate
                   path, fit one temperature on those 159 held-out items, apply it to the stored
                   72-item original cohort, and report a properly calibrated ECE.
                   The three public splits are fully disjoint (verified: 0 shared ids), so the
                   fitted temperature never sees the cohort it is scored on.
Provider/plan:     Hetzner ccx33, ash -- same shape as the baseline host, so latency stays
                   comparable with the 2026-09-21 run.
Hourly (live):     0.2660 gross
Maximum spend:     USD 2.00
Deletion deadline: creation + 3 h (hard)
Correctness gate:  none -- this measures, it does not qualify. Reported beside the in-sample
                   bound of 0.1087 already recorded.
Expected result:   ECE worse than 0.1087, since that bound was fitted on the evaluation items.
                   Recorded before the run.
Evidence dest:     benchmark-results/2026-09-21-jevbench-baseline/
```
