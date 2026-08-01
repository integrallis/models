# MiniCPM5 Session Memory and JVM Gate

This gate attributes the resident-memory increase from four-session decode
batching and qualifies the JVM used by the MiniCPM5 EPYC-Milan performance
profile. Every process used the same 688,065,920-byte GGUF artifact, four
sessions, a 256-token context, 16 warmup steps, 64 measured steps per session,
a fixed 2 GiB G1 heap, eight processors, and 256-bit vectors. Retained rows use
the two-query Q6_K batch kernel unless the row names the one-query control.

## Retained Results

| JVM and mode | Aggregate | Per-request TPOT | CPU | Mean current RSS | Mean anonymous RSS | Mean file RSS | GC |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Temurin 25, sequential | 12.019 tok/s | 332.794 ms | 167,340 ms | 1.046 GB | 0.440 GB | 0.607 GB | 0 |
| Temurin 25, batched | 17.103 tok/s | 233.879 ms | 113,717 ms | 1.621 GB | 1.014 GB | 0.607 GB | 0 |
| GraalVM 25, sequential | 19.561 tok/s | 204.504 ms | 100,643 ms | 1.092 GB | 0.465 GB | 0.627 GB | 0 |
| GraalVM 25, batched, Q6 one-query | 24.660 tok/s | 162.232 ms | 77,760 ms | 1.794 GB | 1.167 GB | 0.627 GB | 0 |
| GraalVM 25, batched | 27.753 tok/s | 144.131 ms | 68,403 ms | 1.997 GB | 1.370 GB | 0.627 GB | 0 |

Batching raises aggregate throughput by 42.29% on Temurin and 41.87% on
GraalVM. On Temurin, the 574,033,920-byte resident increase is entirely
anonymous: file RSS differs by only 12,288 bytes and measured active heap is
8,030,864 bytes lower. Native Memory Tracking committed totals are also nearly
identical. JFR allocation sampling localized the transient anonymous pages to
Vector API conversion work before the giant four-query Q4_K method reached C2;
the hot method then ran without steady-state collections.

GraalVM is the retained JVM recommendation for this exact profile. Three
cooled-down fresh processes measured 27.617, 27.827, and 27.814 tok/s and all
produced the exact Temurin token hash. Relative to batched Temurin, mean
throughput improves 62.27%, per-request TPOT falls 38.37%, and process CPU
falls 39.85%. Mean current RSS is 23.24% higher, so deployments must qualify
both throughput and memory capacity. GraalVM was launched with:

```text
-Djdk.graal.MaximumInliningSize=10000
```

The Q6_K policy is independently retained on GraalVM. Three one-query controls
averaged 24.660 tok/s; two-query blocks improve aggregate throughput 12.54%,
reduce TPOT 11.16%, and reduce process CPU 12.03%. Maximum observed RSS rises
3.89%. Every corresponding run is token-exact and reports zero collections.
The raw one-query controls are under `q6-one-query-graal/`.

## Rejected Compiler Directives

Method-scoped C2 compilation thresholds and synchronous compilation were
screened but not retained. Each candidate remained token-exact, yet its third
fresh process slowed down, approached a 2.1 GB peak RSS, and collected:

| C2 policy | Runs | Mean aggregate | Collections |
| --- | --- | ---: | ---: |
| Control | 16.924 / 17.063 / 17.052 tok/s | 17.013 tok/s | 0 |
| Aggressive threshold plus synchronous compile | 17.130 / 17.043 / 16.092 tok/s | 16.755 tok/s | 59 |
| Moderate threshold plus synchronous compile | 17.083 / 17.091 / 15.532 tok/s | 16.569 tok/s | 102 |
| Synchronous compile only | 17.150 / 17.064 / 16.402 tok/s | 16.872 tok/s | 44 |

The raw rejected reports are retained under
`rejected-compiler-directives/`; none of those flags belongs in a production
profile.

`summary.json` is the machine-readable aggregate. The remaining JSON files are
the unmodified process reports, including host, JVM, NMT, buffer-pool,
diagnostic-plan, artifact-hash, and output-hash metadata.
