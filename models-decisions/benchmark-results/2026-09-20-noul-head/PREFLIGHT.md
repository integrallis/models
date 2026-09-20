# Preflight control record — Noul head harvest and calibration

Written 2026-09-20T09:20:44Z, before any resource was created.

```text
Experiment:              models-decisions cycles 2-3. Build the shared-prefix evaluator and the
                         harvester on the host; harvest (hidden state, verdict) pairs from the
                         qualified Granite 4.1 3B Q4_K_M base over the audited answerability
                         corpus; train and calibrate a Noul head; measure head-vs-decode
                         agreement, ECE/Brier against published labels, and per-decision latency.
Provider / region / plan: Hetzner Cloud, ash (Ashburn, VA, US), ccx33
                         8 dedicated AMD EPYC-Milan vCPU, 32 GB RAM, 240 GB disk
                         Image ubuntu-24.04 x86 (161547269)
                         Chosen to match the retired modeljars-bench-* evidence hosts so the
                         latency figures stay comparable with existing runs.
Expected hourly charge:  0.2660 gross (live API read 2026-09-20T09:20:44Z); billed hourly
Maximum authorized spend: USD 10.00 (self-imposed; 24 h at this rate is about USD 6.38)
UTC creation deadline:   2026-09-20T09:20:44Z
UTC deletion deadline:   2026-09-21T09:20:44Z  (hard; label delete-by=2026-09-21T09-20-44Z)
Correctness gate:        head verdict agrees with the decode verdict at a floor pre-registered
                         before the sealed split is opened; calibration reported with its
                         majority-class floor beside it; n>=3
Performance gate:        per-decision cost strictly below one decode turn on the same host,
                         with physical prefix sharing proven by sharesPrefixStorage rather than
                         inferred from timing
Local evidence dest:     projects/models-decisions/models-decisions/benchmark-results/2026-09-20-noul-head/
SSH key:                 Hetzner 114663461, verified by comparing public material
                         (MD5:60:4e:57:38:96:48:c5:45:73:10:9d:37:f7:4f:74:cf), not by label
Firewall:                TCP/22 from 68.227.243.139/32 only
No external inference runtime will be installed or used.
```

## Created resources (record immediately)

| Resource | ID | Detail |
| --- | --- | --- |
| Hetzner server | `166607808` | `decisions-bench-noul-20260920`, ccx33, ash, IPv4 5.161.100.228 |
| Hetzner firewall | `11651096` | `decisions-bench-noul-20260920-fw`, TCP/22 from 68.227.243.139/32 only |

Both must be deleted at teardown: the server first, then the firewall as a separate resource.

## Host baseline (captured over SSH before any work)

```text
Server        166607808  decisions-bench-noul-20260920  ccx33  ash
Kernel        6.8.0-138-generic          OS  Ubuntu 24.04.4 LTS
CPU           AMD EPYC-Milan, 8 vCPU presented as 4 cores x 2 threads
ISA           avx2  (NO avx512f on this host)
Memory        30 GiB visible
Disk          225 G total, 215 G free
JDK           absent at provision time
Firewall      11651096 attached, TCP/22 from 68.227.243.139/32 only
```

This is the same ccx33 shape as the retired `modeljars-bench-*` evidence hosts, so latency
figures here stay comparable with those runs. The host is AVX2 only: it cannot supply the
AVX-512 arm that vectors PR #74 is still waiting on.
