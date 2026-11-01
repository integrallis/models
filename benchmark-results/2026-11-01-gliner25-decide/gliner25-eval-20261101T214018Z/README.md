# Matrix and attention screening result

Complete CPU comparison on one m6a.4xlarge with eight inference threads per arm, F32 weights and 429 requests per arm (13 shapes, three warmups and 30 timed samples each). Zero inference errors. Runtime source `0b115ba1aad4f7a4254011bd10ebd9bdfa410883`; checkpoint `d1ca5a14b0b535987ce6333680d0de2c963100cbff91745620e8455ec731caf5` (general-v1).

**11 of 13 buckets pass.** This candidate is not qualified. The two 128-token failures are retained; every bucket completed before the verdict. Nine diagnostic and thirteen shape-parity cases pass, as does the existing F16 reranker regression.

| Shape | Models p50 ms | Publisher p50 ms | Throughput ratio | p95 ratio | Pass |
|---|---:|---:|---:|---:|---|
| tokens32-labels2-tasks1 | 185.4 | 180.2 | 0.9911 | 1.0766 | True |
| tokens32-labels8-tasks1 | 165.3 | 180.6 | 1.0687 | 0.9874 | True |
| tokens128-labels2-tasks1 | 346.6 | 280.8 | 0.8028 | 1.3044 | True |
| tokens128-labels8-tasks1 | 349.7 | 281.1 | 0.8044 | 1.2451 | True |
| tokens128-labels32-tasks1 | 353.4 | 281.0 | 0.7749 | 1.5086 | False |
| tokens256-labels2-tasks1 | 618.0 | 517.1 | 0.8112 | 1.1817 | True |
| tokens256-labels8-tasks1 | 582.9 | 513.7 | 0.8345 | 1.3861 | True |
| tokens256-labels32-tasks1 | 588.8 | 520.3 | 0.8642 | 1.2721 | True |
| tokens512-labels2-tasks1 | 1127.0 | 1325.1 | 1.1581 | 0.9208 | True |
| tokens512-labels8-tasks1 | 1195.3 | 1331.3 | 1.1089 | 0.8827 | True |
| tokens512-labels32-tasks1 | 1198.8 | 1327.4 | 1.1074 | 0.9275 | True |
| tokens128-labels8-tasks2 | 350.3 | 273.0 | 0.7677 | 1.5624 | False |
| tokens512-labels8-tasks2 | 1198.1 | 1321.9 | 1.1022 | 0.9127 | True |

Observed load: Models 1491.0 ms, publisher 4627.9 ms. These are one observation each, excluding download and process startup; caches were not cleared.

Peak process RSS: Models 3.584 GiB; publisher 4.044 GiB. All 512-token buckets exceed publisher throughput in this pair. Three final counterbalanced replications are still required after freezing the final runtime and checkpoint.

The next candidate prepares immutable projection weights once at load, removing repeated weight staging. Its public factory owns both native kernels and tokenizer; it must pass a new full comparison. These results do not establish the speed of that next candidate.

Cleanup: instance `i-0da340969d7e2d664` terminated and root volume `vol-0a0c8661cc57c2496` verified deleted. Rental upper bound: 877 seconds. The cumulative value in control.json includes the still-active training pod’s full reserved rental; the ledger recomputes actual settled receipts after deletion.
