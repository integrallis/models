# MiniCPM5 independent-session batching qualification

This gate compares four independent generation sessions advanced sequentially
with the same sessions advanced through one batched transformer pass. Each
session owns its KV cache and position; model weights and projection work are
shared for the decode step.

The controlled host has eight AMD EPYC-Milan vCPUs and runs Eclipse Temurin
25.0.3 with 256-bit Vector API species. Each fresh process uses a fixed 2 GiB
G1 heap, eight active processors, the profiled `two-query-block` Q6_K policy,
16 warmup decode steps, and 64 measured steps for each of four sessions. The
process order is sequential, batched, batched, sequential.

| Metric | Sequential sessions | Batched sessions | Change |
|---|---:|---:|---:|
| Aggregate throughput | 12.025 tok/s | 16.984 tok/s | +41.24% |
| Per-request TPOT | 332.648 ms | 235.519 ms | -29.20% |
| Process CPU | 167,390 ms | 114,200 ms | -31.78% |
| Maximum observed peak RSS | 1,055,150,080 B | 1,654,734,848 B | +56.82% |

Every process generated 256 tokens with SHA-256
`722fe1aa593780b2ff7da300037decad54be5d9d0ff6e026e7515329fd56ce01`,
and no measured process collected. The repeatable RSS difference is a serving
capacity tradeoff that requires heap/native-memory and mapped-page attribution;
process RSS alone cannot identify its owner.

`summary.json` contains the aggregate and exact controls. The four other JSON
files are the unmodified process reports.
