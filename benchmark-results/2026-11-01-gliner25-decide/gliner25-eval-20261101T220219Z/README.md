# Prepared-weight public-runtime screening

One complete paired screening run, not final qualification. All 13 shapes pass throughput and p95 gates; three pairs on the final checkpoint/runtime and full quality are still required.

Runtime source: `4055e69f8824ac983d1ca191a9997a1c65052241`; candidate weights: `d1ca5a14b0b535987ce6333680d0de2c963100cbff91745620e8455ec731caf5` (general-v1). Public `RustGlinerClassificationModel` route, F32, 8 threads, m6a.4xlarge, Java 25.0.3. Publisher source and exact file identities are in the raw report.

Each arm completed 429 requests (3 warmups and 30 timed samples per shape) with zero inference errors. Numerical parity and the F16 reranker regression passed. Load was 2.079 s for Models and 4.638 s for the publisher; these are single observations, not controlled cold-load distributions.

| Shape | Models p50 ms | Publisher p50 ms | Throughput ratio | p95 ratio |
|---|---:|---:|---:|---:|
| tokens32-labels2-tasks1 | 128.3 | 179.5 | 1.493 | 0.540 |
| tokens32-labels8-tasks1 | 107.5 | 179.5 | 1.606 | 0.736 |
| tokens128-labels2-tasks1 | 278.5 | 278.9 | 0.951 | 1.213 |
| tokens128-labels8-tasks1 | 276.0 | 278.1 | 1.001 | 1.039 |
| tokens128-labels32-tasks1 | 295.6 | 277.1 | 0.934 | 1.221 |
| tokens256-labels2-tasks1 | 531.4 | 512.6 | 0.934 | 1.068 |
| tokens256-labels8-tasks1 | 533.8 | 521.0 | 0.956 | 1.129 |
| tokens256-labels32-tasks1 | 535.6 | 516.1 | 0.965 | 1.034 |
| tokens512-labels2-tasks1 | 1081.0 | 1351.1 | 1.254 | 0.758 |
| tokens512-labels8-tasks1 | 1092.3 | 1357.6 | 1.204 | 0.794 |
| tokens512-labels32-tasks1 | 1098.4 | 1350.0 | 1.227 | 0.831 |
| tokens128-labels8-tasks2 | 270.2 | 273.6 | 0.996 | 1.050 |
| tokens512-labels8-tasks2 | 1101.6 | 1366.9 | 1.213 | 0.807 |

The prepared immutable native weights remove per-request weight staging. This screening is distinct from the earlier 0b115ba1 run (11/13 passing) and the complete original-base measurement (12/13 failing). The Linux run does not establish platform-wide shutdown safety: Windows ARM CI exposed a separate asynchronous native-pool teardown defect being corrected before final replication.

Cleanup independently verified: instance terminated, root volume absent; see `cleanup-verification.json`.
