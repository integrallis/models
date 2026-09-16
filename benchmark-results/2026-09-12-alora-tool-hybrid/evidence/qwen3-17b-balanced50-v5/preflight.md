# Qwen3 1.7B balanced-abstention activated-adapter preflight

Status: **trained and rejected by the frozen smoke gates**.

The rank-16 v4 diagnostic passed tool syntax and tool choice but made false calls on 24% of the
irrelevance slice. Its selected training rows contained 19% no-call examples. This next bounded
experiment changes the training population, not the qualification thresholds: it uses an exactly
balanced 50% call / 50% no-call population selected from the same provenance-bound preparation.
It does not select or synthesize rows from the BFCL evaluation files.

- Base: `Qwen/Qwen3-1.7B`
- Base revision: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
- Parent preparation manifest SHA-256:
  `c7da1f5f826bb1c6f10047e09ff1e7a69e3346dad2336400e5a277ef937090b9`
- Resampling script SHA-256:
  `4060543322676cc759c597f2766c6a2a7540df416e5e32e86ef3b83a5249cf4b`
- Training/formatting implementation SHA-256:
  `02aa7335c936fec446a33b5a287d66195834a1f24132823b77e8b753e2238814`
- Independent evaluator SHA-256:
  `74b41820632773f915f5e0f1c643bf51e139fcae989c3a3817040fa96b82f645`
- Resampled manifest SHA-256:
  `7b658bcedb7937b225ac916d0be7c63b599a729857224c05508974d0be2fb27b`
- Train: 4,000 rows, exactly 2,000 no-call; SHA-256
  `ca2be39cc755101469a4500e4bb711d69d608b4fa04b3f946f1dc1bb1ffc1cba`
- Validation: 500 rows, exactly 250 no-call; SHA-256
  `019d1c2c20c64598503b9cdba091b0f31d3bf5731176bceff9efc9dd01e77bdc`
- Adapter: rank 16, alpha 32, zero dropout, all seven attention/MLP projection families
- Training: one epoch, learning rate `1e-4`, batch size 1, gradient accumulation 16, maximum
  length 1,024, seed `20260913`, BF16, SDPA, gradient checkpointing
- Remote output: `/opt/modeljars-alora-v3/qwen17-balanced50-v5`

The fixed diagnostic gates remain 100% syntax, 100% schema, at least 85% exact tool calls, no worse
than the unadapted base, and at most 5% false calls. A diagnostic failure rejects this run without a
full qualification, Java packaging, catalog entry, or release. A diagnostic pass advances to the
previously fixed full evaluation and JVM gates; it does not itself qualify the model.

## Result

Training completed at `2026-09-13T09:03:00Z`. The adapter weight SHA-256 is
`007aa5dca9892f146239de50d9c2f30d7e17a4bfe17df8150d78a7970a36f173`; its training manifest
SHA-256 is `1d08db16698bfb2312351cb80d4a10b5802e7e2ec1ca2db2816f7e606f5681ef`.

The frozen 25-case-per-kind smoke completed at `2026-09-13T09:05:00Z` and rejected the adapter:

- syntax: 98.67% (required 100%);
- schema: 98.67% (required 100%);
- exact tool calls: 82% (required at least 85% and no worse than the base's 86%);
- irrelevant-prompt false calls: 20% (required at most 5%).

The full qualification, Java packaging, catalog entry, and release were not run. The immutable
smoke report and records are retained under `smoke25/` with SHA-256 values
`f86b83a5c69ce962a2ce982a914ff2f1ca692c97c5fdc53ce25809c99d6357d0` and
`01eeeb8aef24c1e40c7d5228284b5327d8d56ec3b1127ec263fceec75cbdb28e`.
