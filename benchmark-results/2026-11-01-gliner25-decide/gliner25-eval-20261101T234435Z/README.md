# Three complete original-weight CPU performance replications

All three pairs pass all 13 registered shapes, with 30 timed requests after 3 warmups per bucket.
Each arm completed 429 requests per pair: 2,574 total requests across six processes, zero errors.
Arm order is counterbalanced reference/models, models/reference, reference/models. Both arms use
8 threads on the same m6a.4xlarge host, the identical pinned original F32 checkpoint and request
corpus. Per-request raw JSONL, runtime hashes, manifest and host data are retained here.

| Shape | Models median p50 ms | Publisher median p50 ms | Lowest throughput ratio | Highest p95 ratio |
| --- | ---: | ---: | ---: | ---: |
| tokens32-labels2-tasks1 | 137.9 | 186.7 | 1.332 | 0.755 |
| tokens32-labels8-tasks1 | 130.3 | 188.6 | 1.414 | 0.761 |
| tokens128-labels2-tasks1 | 292.3 | 284.4 | 0.964 | 1.053 |
| tokens128-labels8-tasks1 | 292.1 | 290.7 | 0.965 | 1.190 |
| tokens128-labels32-tasks1 | 294.9 | 284.8 | 0.912 | 1.327 |
| tokens256-labels2-tasks1 | 532.6 | 519.7 | 0.936 | 1.080 |
| tokens256-labels8-tasks1 | 536.0 | 522.9 | 0.918 | 1.207 |
| tokens256-labels32-tasks1 | 514.1 | 527.1 | 0.974 | 1.116 |
| tokens512-labels2-tasks1 | 1113.2 | 1412.8 | 1.257 | 0.825 |
| tokens512-labels8-tasks1 | 1107.9 | 1437.4 | 1.249 | 0.816 |
| tokens512-labels32-tasks1 | 1114.3 | 1410.5 | 1.234 | 0.776 |
| tokens128-labels8-tasks2 | 288.0 | 285.4 | 0.947 | 1.260 |
| tokens512-labels8-tasks2 | 1119.2 | 1410.4 | 1.244 | 0.886 |

The p50 columns are the median of each arm's three per-pair p50 measurements. Ratio columns
show the worst of the three paired measurements, not a ratio computed from independent medians.
Every ratio clears the unchanged minimum 0.80 throughput and maximum 1.50 p95 latency gates.

This measures the committed development payload `ff7130e3da566c0dc15723ebbe6a5ba2119e43ef`.
The focused release contains identical GLiNER/DeBERTa implementation files, but a separate repeat
against the actual Central artifacts remains required before catalog admission. Numerical parity
and the existing F16 reranker regression also passed on this worker. The instance and root volume
were independently verified deleted; see cleanup-verification.json.
