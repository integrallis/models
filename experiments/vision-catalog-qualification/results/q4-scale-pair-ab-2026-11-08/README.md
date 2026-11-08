# Paired Q4 half-scale conversion: rejected

Candidate `b782797dcc566a006ac1b54db6b5c31feff9a11e` converts adjacent Q4_K
half scales with one F16C instruction. Baseline runtime is
`3f31646c743647449310de2c5ff57fbd55d9e957`. The Java JARs are identical;
the build receipt records the native binaries and toolchain. Linux synthetic tests
pass (63 active tests; three ignored), and Clippy passes.

One m6a.4xlarge ran sequential baseline/candidate/baseline, eight threads,
three warmups and seven measured repetitions per arm, using the original caption
and short-answer inputs. All 30 token sequences match within each cohort.

| Cohort | Baseline 1 decode | Candidate decode | Baseline 2 decode |
| --- | ---: | ---: | ---: |
| Caption | 3863.366 ms | 3921.200 ms | 3863.722 ms |
| Short answer | 249.079 ms | 248.056 ms | 243.943 ms |

Caption decode regresses about 1.5%; short-answer movement is within the 2.1%
baseline variation. The short reference F16 median is 225.255 ms, for a candidate
ratio of 1.101223. These are diagnostics, not a qualification repeat.

Reject the candidate and do not promote its runtime code. No complete admission
repeat was started because the predeclared benefit gate failed. Models 0.5.16
retains the validated production implementation. The report does not change the
original admission gates or either model's qualification state.

The complete archive was verified locally before deleting the worker, security
group and key at 08:31:28 UTC. Its checksum and cleanup receipt are retained here.
The archive excludes model weights, credentials and compiler caches; it is stored
at `audit-artifacts/pr380-scale-pair-2026-11-08/pr380-scale-pair-raw.tar.gz`.
