# Routing-only result (2026-11-08)

Measured on the dedicated ccx33 host described in `lscpu.json`. The candidate
is `e9c1760e`; the control uses published Models 0.5.16 / Vectors 0.1.33 with
explicit native-kernel injection. This is a direct Models experiment, not a
released ModelJars 0.1.76 consumer benchmark.

| Exact tokenizer length | Baseline medians, ms | Candidate median, ms | 300M control median, ms |
| --- | --- | --- | --- |
| 32 | 261.77 / 258.83 | 158.40 | 445.73 |
| 128 | 1234.42 / 1239.10 | 950.25 | 1393.36 |
| 512 | 9076.43 / 9138.58 | 7918.97 | 6238.70 |

Every candidate gain exceeds the observed bracketing baseline drift. Every timed
vector matches bit for bit. A separate check recomputed each stored vector hash,
verified normalization and confirmed only `backend-java-0.5.16.jar` differs among
the timed runtime dependency graphs. Full p95/sample data are in `summary.json`
and the raw reports. Eleven measured samples are not an SLA certification.

The 512-token candidate remains 26.9% slower than the 300M control. The proposed
50 ms short-input target is not met. Bulk ingest was not measured here; this
experiment does not establish a 10x ingest gain.

The eight-probe oracle minimum is **0.999808716201502** for both EG2 arms:
existing catalog floor **0.999** passes; memory's proposed **0.99999** floor does
not. No existing catalog qualification or admission threshold was changed.

Separate trace runs observe 0 native projections in the control and 217 per
embedding in the candidate. The warmed candidate profile attributes 9,112 of
19,357 main-thread samples to attention (47.1%). The JFR includes native FFM
frames, but the library's internal Rust symbols are stripped; do not attribute
those unresolved samples to a specific native kernel or spin loop.

The raw archive and both JFRs remain in the workspace audit-artifacts directory;
their SHA-256 values are in `summary.json`. Each report records effective runtime
JAR hashes. The archive's original bootstrap payload manifest predates harness
corrections; the per-report hashes are the effective executed classpath.
Three setup attempts are separately retained: absent cgroup-v2 metadata, an
incorrectly conflated oracle threshold, and a stale harness upload. None produced
accepted timing data. All tables above use only the completed canonical run.
