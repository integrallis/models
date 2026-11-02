# Supplied GLiNER2.5-Decide: completed VPS performance measurements

**The benchmark completed. The Models runtime failed the relative speed gate.**
Those are separate outcomes. Both runtimes successfully executed all 429 requests across
all 13 registered shapes, with zero inference errors. Each shape has 3 warmups and 30 timed
requests (390 timed requests per runtime). Recalculating from the raw logs exactly reproduces
the historical report. This audit does not rerun inference or incur another rental.

## Environment and artifact

- Date: 2026-11-01; EC2 m6a.4xlarge, AMD EPYC 7R13, 16 vCPU.
- Both arms ran serially on the same host, using 8 inference threads, F32, batch 1, concurrency 1.
- Supplied checkpoint: `fastino/GLiNER2.5-Decide`, revision `9d1bfb848cd16d93ee56ccbc59cd92bf0f284d90`.
- This is the original checkpoint, not the separately trained variant.
- Models source: `6add46ee740f8acd80295741fc8e22bae34f845f`; Java 25.0.3, Vectors 0.1.32.
- Publisher source: `ca4f68235b50fe4c3ecf3876851d748fe4b15c6c`; PyTorch 2.9.1+cpu.
- Models computes the encoder/head in Java and uses a Rust tokenizer. The later native projection experiment is not this run.

## Absolute timings and throughput

Times are end-to-end request wall times in milliseconds, including the local JSONL driver.
Throughput is timed request count divided by total timed wall time, not the reciprocal of p50.
Inputs are the frozen synthetic shape corpus; this measures speed, not answer accuracy.

| Tokens / labels / tasks | Publisher p50 ms | Models p50 ms | Publisher p95 ms | Models p95 ms | Publisher req/s | Models req/s |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 / 2 / 1 | 177.8 | 223.0 | 178.7 | 249.6 | 5.623 | 4.373 |
| 32 / 8 / 1 | 178.6 | 221.9 | 197.8 | 243.5 | 5.364 | 4.432 |
| 128 / 2 / 1 | 275.6 | 753.2 | 280.6 | 814.4 | 3.625 | 1.310 |
| 128 / 8 / 1 | 278.6 | 753.9 | 282.2 | 864.8 | 3.585 | 1.307 |
| 128 / 32 / 1 | 279.0 | 759.3 | 288.3 | 783.3 | 3.575 | 1.304 |
| 256 / 2 / 1 | 510.2 | 1611.4 | 512.0 | 1677.8 | 1.961 | 0.616 |
| 256 / 8 / 1 | 511.4 | 1609.5 | 557.3 | 1642.9 | 1.941 | 0.618 |
| 256 / 32 / 1 | 511.3 | 1615.4 | 513.0 | 1812.1 | 1.956 | 0.613 |
| 512 / 2 / 1 | 1343.4 | 4031.6 | 1350.7 | 4251.1 | 0.742 | 0.246 |
| 512 / 8 / 1 | 1344.1 | 4032.1 | 1388.6 | 4284.5 | 0.740 | 0.246 |
| 512 / 32 / 1 | 1343.1 | 4035.4 | 1427.6 | 4279.5 | 0.740 | 0.246 |
| 128 / 8 / 2 | 269.5 | 773.8 | 270.4 | 839.7 | 3.711 | 1.276 |
| 512 / 8 / 2 | 1344.9 | 4044.6 | 1391.1 | 4275.9 | 0.739 | 0.245 |

## Load and memory observations

| Metric | Publisher | Models |
| --- | ---: | ---: |
| Worker-reported model load | 4.64 s | 22.91 s |
| Process peak RSS | 4.04 GiB | 5.46 GiB |
| Request errors | 0 / 429 | 0 / 429 |

Load times are one observation per runtime around their model-loading calls. They exclude
model downloads and process/interpreter startup; filesystem caches were not cleared.
They are not controlled cold-start measurements. Peak RSS is the process high-water mark,
including runtime overhead, not just tensor storage or per-request allocation.

## What failed, and what did not

- **Measurement completed:** all requested shapes and both runtimes are present.
- **Runtime performance failed:** 12 of 13 shapes miss at least one fixed relative gate
  (throughput >= 0.80x publisher and p95 latency <= 1.50x publisher).
- **Qualification is incomplete:** this is one measured pair; final qualification needs
  three passing pairs, passing accuracy, and released-runtime integration checks.
- The later native projection experiment stopped early and has its own partial report.
  It does not invalidate or replace this complete baseline.

## Reproduction and evidence

```sh
python3 experiments/gliner25-decide/report_base_performance.py
```

- [Audit and source digests](base-performance-audit.json)
- [Complete paired report](results-restart-1/performance-1-report.json)
- [Publisher raw requests](results-restart-1/performance-1-reference.jsonl)
- [Models raw requests](results-restart-1/performance-1-models.jsonl)
- [Runtime artifact hashes and worker cleanup](deployment.json)
