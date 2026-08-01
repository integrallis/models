# MiniCPM5 session-stage refactor gate

This gate verifies that splitting the session-batch transformer loop into
allocation-free input, QKV/attention, FFN, observation, and output stages does
not sacrifice the qualified throughput. The baseline and candidate execute the
same kernels and differ only in Java method structure.

The controlled host has eight AMD EPYC-Milan vCPUs and runs Eclipse Temurin
25.0.3 with 256-bit Vector API species. Each fresh process uses a fixed 2 GiB
G1 heap, eight active processors, the profiled `two-query-block` Q6_K policy,
four sessions, 16 warmup steps, and 64 measured steps. Process order is
baseline, candidate, candidate, baseline.

| Metric | Baseline `b79acef` | Refactor `6bf21b0` | Change |
|---|---:|---:|---:|
| Aggregate throughput | 16.972 tok/s | 17.128 tok/s | +0.92% |
| Per-request TPOT | 235.691 ms | 233.531 ms | -0.92% |
| Process CPU | 114,165 ms | 113,505 ms | -0.58% |

Every run generated 256 tokens with SHA-256
`722fe1aa593780b2ff7da300037decad54be5d9d0ff6e026e7515329fd56ce01`,
and no measured process collected. The small favorable differences are not
treated as an optimization claim; this gate establishes no regression and
supports retaining the lower-complexity implementation.

`summary.json` contains the aggregate and exact controls. The four other JSON
files are the unmodified process reports.
