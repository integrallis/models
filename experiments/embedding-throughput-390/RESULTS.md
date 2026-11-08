# Issue 390: measured EmbeddingGemma 2 fixes

The retained runtime passes the existing embedding oracle gate and preserves
all tested vectors bit for bit. It fixes the dropped native kernel and partitions
independent attention heads through the existing Java GGUF executor at **128 or
more tokens**. Shorter inputs execute the same head arithmetic inline.

## Measurements and provenance

One dedicated Hetzner ccx33 (8 x86 vCPU, AMD EPYC-Milan), JDK 25.0.5, 8 Java/native
threads, Models 0.5.16 and Vectors 0.1.33. Each comparison used 5 warmups and 11
measured samples, with baseline / candidate / baseline ordering; a separate
always-parallel comparison was added to the scheduling experiments. Every sample,
command, effective classpath hash and vector is retained in `results/`.

| Experiment | Tokens | Bracketing baseline medians, ms | Candidate median, ms | 300M control, ms |
| --- | --- | --- | --- | --- |
| Routing only (`e9c1760e`) vs published Models | 32 | 261.77 / 258.83 | 158.40 | 445.73 |
| Routing only vs published Models | 128 | 1234.42 / 1239.10 | 950.25 | 1393.36 |
| Routing only vs published Models | 512 | 9076.43 / 9138.58 | 7918.97 | 6238.70 |
| Parallel heads (`18771163`) vs published Models | 128 | 1156.46 / 1279.74 | 708.70 | 1224.59 |
| Parallel heads vs published Models | 512 | 9078.63 / 9132.81 | 3339.60 | 6418.02 |
| Inline short heads (`131e83ed`) vs routing only | 32 | 158.99 / 156.53 | 142.96 | 499.87 |
| Final 128-token boundary (`6b44ca96`) vs routing only | 64 | 370.37 / 359.25 | 316.68 | 728.12 |

These are separate experiments and controls, not one final-version end-to-end
speedup table. At 32 tokens the final boundary keeps the inline route measured
under `131e83ed`; at 128 and 512 it keeps the parallel route measured under
`18771163`. The 64-token boundary change was measured explicitly on `6b44ca96`.
Every retained gain exceeds that comparison's baseline drift. The 512-token gap
to the 300M control is closed in this dedicated-host experiment: 3.34 versus
6.42 seconds. A released ModelJars consumer repeat remains required.

The source has no Qwen hidden-state change: candidate `16a4410f` showed no gain
beyond baseline drift at any tested length and was reverted. Its raw reports
remain under `results/2026-11-08-qwen-head`.

## Correctness and observed execution

- The public-loader regression originally observed **zero** injected matrix calls;
  the fix observes 19 eligible synthetic calls, while preserving unsupported-type
  and unsupported-shape fallback, exact outputs and single kernel ownership.
- Real EG2 trace runs observe **0 -> 217** native projections per embedding.
- All eight frozen llama.cpp oracle probes and every timed input match the
  published runtime's output **bit for bit**. Stored vector hashes and unit norms
  were recomputed independently after retrieval.
- Minimum oracle cosine is **0.999808716201502**. This passes the existing
  **0.999** Models/ModelJars gate; it does **not** pass memory's proposed
  **0.99999** gate. No admission rule or existing catalog entry was changed.
- Attention tests cover full/symmetric windows, grouped heads, learned query
  scaling, nonuniform values, scratch growth/shrinkage, output sentinels, and both
  sides of the 127/128 scheduling boundary.
- The local unit suite, targeted final-boundary tests, formatting and SpotBugs
  pass. CI must validate the final PR head before merge.

## Profile and remaining limits

The routing-only candidate's warmed 128-token CPU JFR attributes 9,112 of 19,357
main-thread samples (47.1%) to serial attention. That finding motivated the head
partitioning. Java and native FFM frames were collected separately from timing;
internal Rust symbols are stripped, so unresolved native samples cannot be
assigned reliably to a particular kernel or spin loop. JFR hashes and archive
receipts are retained beside the raw reports; binary archives remain in the
workspace's `audit-artifacts/models-390-embedding-2026-11-08/`.

The measured 32-token p95 for inline scheduling is **180.93 ms**, above the proposed
50 ms default-tier target. Eleven samples are a diagnostic, not an SLA estimate.
True multi-text batching and bulk-ingest throughput were not measured or fixed.
Memory's LME and Hotpot retrieval acceptance remains separate work.

ModelJars **0.1.76** still uses the older automatic embedding loader and pins
Models 0.5.12 / Vectors 0.1.32. It cannot receive this fix merely because a new
Models artifact exists: release Models first, update and release ModelJars with
its native embedding loader, then repeat through the actual released API.

## Resource disposition

All five experiment archives were copied and SHA-verified. The exact temporary
server `175151018` and firewall `12000987` were deleted; the closing inventory at
2026-11-08T16:21:25Z reports zero servers, firewalls, primary IPs, volumes and
floating IPs. No provider credentials were placed on the worker.
