# Soprano Q8 scheduling cross-hardware preflight

- Experiment: isolate whether Soprano's batch-one Q8 projection gap is caused by the Q8 format or
  by the production row-parallel scheduling threshold.
- Provider / region / exact plan: Vultr `ewr`, `vc2-4c-8gb`, Ubuntu 24.04 x64.
- Expected hourly and minimum charge: USD 0.055/hour, hourly billing.
- Maximum authorized spend: USD 2.00.
- UTC creation deadline: 2026-09-04T19:45:00Z.
- UTC deletion deadline: 2026-09-04T23:45:00Z.
- Correctness gate: production and forced-row Q8 outputs are bit-identical for every benchmark
  shape before measurements begin.
- Performance gate: forced row scheduling materially improves the 2304x512 and 512x2304 Soprano
  projections without changing their output. The 8192x512 output head remains the control because
  production already parallelizes it.
- Local evidence destination:
  `benchmark-results/audio/soprano-q8-scheduling-20260904/`.
- Vultr resource: instance `9c468503-e86b-4b96-bbb3-00d2853979e8`, created
  2026-09-04T19:41:05Z with label `modeljars-soprano-q8-20260904`.

The persistent `vectors-bench` host was rejected for this run: it was 97% full and an unrelated
Python model oracle was active. No process or retained data on that host was changed.
