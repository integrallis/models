# mellum on Zen 3: the measurement the published qualification never had

**Frozen before any data was collected.**

## Why this run exists

mellum's qualification is published — `catalog/qualifications.json` in ModelJars/modeljars#258,
marker live on Maven Central. Every passing measurement behind it was taken on AVX-512 hardware:
the replication in `benchmark-results/2026-11-02-mellum-qualification-replication` ran on an
**AMD EPYC 9754** whose `/proc/cpuinfo` carries `avx512f`.

78 of this catalog's 82 qualification entries were measured on **Zen 3 EPYC without AVX-512**
(`EPYC 7R13`, `EPYC-Milan`) at 8 or 16 vCPU. On that hardware mellum has exactly one measurement:
**0.763, FAILED_RELATIVE_GATE**, in `benchmark-results/2026-11-01-mellum-replicated-verdict`, taken
on an **EPYC 7713** *before* the Q4_K per-block unpack fix (`dcd8b172`).

`0.763 x 1.181 = 0.901` has been cited repeatedly as the reason to believe it now passes. That is
arithmetic on two numbers from different runs, not a measurement, and it has been presented as a
caveat four times instead of being resolved. This resolves it.

## Question

With `dcd8b172` in the library, does mellum's `decodeThroughputRatio` clear 0.80 on a Zen 3,
AVX2-only EPYC — the hardware class this catalog is actually calibrated to?

## Design

Identical to the 2026-11-02 replication protocol, so the only intended difference is the host:
three complete paired runs on one box, strictly sequential, candidate then comparator within each
pair, one session per pair. Workload `coding`, template `chatml-no-think`, top-k 1,
`--warmups 1 --iterations 3`, `--threads 16`, context 2048, max-tokens 256, `COMPARATOR=ollama`,
`--artifact` passed so `artifactSha256` is non-null and the comparison is admissible. Verdicts come
from `RagQualificationCli` under `RagProductionQualificationPolicy`, never from dividing reports by
hand.

The box is a RunPod `cpu3c` at 16 vCPU. Its CPU was verified **before** the payload was uploaded:
`AMD EPYC 7713 64-Core Processor`, `nproc=16`, no `avx512` flag of any kind in `/proc/cpuinfo`.
That is the same CPU model as the 0.763 run, which is what makes the two directly comparable.

The native library is built from the merged source on the box itself, per the replication protocol,
and its sha256 is recorded in `HOST.txt` beside the result. Source revision is recorded too.

## Decision rule, fixed in advance

- **mean `decodeThroughputRatio` over the three paired runs >= 0.80** — the published entry is
  confirmed on the catalog's reference hardware class, and this record is committed as that
  confirmation.
- **mean < 0.80** — the published entry is **not** supported on the hardware 78 of 82 entries were
  measured on. It is withdrawn from the catalog and the marker's qualification claim retracted,
  whatever the Zen 5 runs said. A revert PR follows and this record is published as the null.

`endToEndLatencyRatio` and `timeToFirstTokenRatio` are reported per run whatever they do, and the
CLI's own verdict string is reported verbatim. The 2026-11-02 record's decision rule covered only
`decodeThroughputRatio` while the gate that actually failed was end-to-end; that omission is not
repeated here — if the policy returns anything other than QUALIFIED on the mean-passing runs, that
is reported as the headline, not filtered out.

No run is discarded. An environmental failure is recorded and the run repeated, with both kept.

## What this cannot settle

One box, one CPU model, one 16-vCPU slice of it. It does not establish behaviour at 8 vCPU, on
`EPYC 7R13` specifically, or on ARM. `/proc/loadavg` and `/proc/meminfo` in this container report
the host, not our cpuset, so neighbour load is recorded for the file and never used as a gate —
pairing both arms in one session on one box is what controls for it.
