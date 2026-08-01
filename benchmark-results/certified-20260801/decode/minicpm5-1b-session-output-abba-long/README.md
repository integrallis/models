# MiniCPM5 session output-projection qualification

This gate isolates the optimization that projects the final hidden states for
four independent sessions in one quantized matrix pass. Both revisions use the
same session-batched transformer path; only the vocabulary projection differs.

The controlled host has eight AMD EPYC-Milan vCPUs and runs Eclipse Temurin
25.0.3 with 256-bit Vector API species. Each fresh process uses a fixed 2 GiB
G1 heap, eight active processors, the profiled `two-query-block` Q6_K policy,
16 warmup decode steps, and 64 measured steps for each of four sessions. The
process order is baseline, candidate, candidate, baseline.

| Metric | Baseline `293a9fa` | Candidate `b79acef` | Change |
|---|---:|---:|---:|
| Aggregate throughput | 14.025 tok/s | 16.834 tok/s | +20.03% |
| Per-request TPOT | 285.208 ms | 237.615 ms | -16.69% |
| Process CPU | 139,150 ms | 114,830 ms | -17.48% |

Every process generated 256 tokens with SHA-256
`722fe1aa593780b2ff7da300037decad54be5d9d0ff6e026e7515329fd56ce01`.
The baseline's second process collected 29 times and reached a higher RSS high
water mark; the candidate processes did not collect. RSS includes mmap page
residency, so this gate makes no capacity claim from those process samples.

`summary.json` contains the aggregate and exact controls. The four other JSON
files are the unmodified process reports.
