# Qwen3 1.7B abstention-continuation preflight

Status: **trained and rejected by the frozen smoke gates**.

The balanced-from-scratch v5 adapter regressed exact tool calls and still overcalled on 20% of the
fixed irrelevance smoke. The stronger v4 adapter reached 88% exact tool calls but overcalled on 24%.
This bounded experiment continues the exact v4 checkpoint on provenance-bound, independently
resampled no-call examples. It changes the adapter training recipe, not the qualification data or
thresholds.

- Base: `Qwen/Qwen3-1.7B`
- Base revision: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
- Initial v4 adapter SHA-256:
  `dcaeab65a51ebf9d98c97f7203e5890cb3e24ae592ba1f2358cbc1d8e46895bd`
- Initial v4 training-manifest SHA-256:
  `5d73180fd1bbc13ef7664133cb2c3e5762025c408e35f1aac0ee9fbcb62d63bc`
- Parent preparation manifest SHA-256:
  `c7da1f5f826bb1c6f10047e09ff1e7a69e3346dad2336400e5a277ef937090b9`
- Resampling script SHA-256:
  `4060543322676cc759c597f2766c6a2a7540df416e5e32e86ef3b83a5249cf4b`
- Continuation trainer SHA-256:
  `bcf5942e8689aa92c02998ce858b56580c728c1fce7e23451578b5c75e05ad8c`
- Independent evaluator SHA-256:
  `74b41820632773f915f5e0f1c643bf51e139fcae989c3a3817040fa96b82f645`
- Resampled manifest SHA-256:
  `e756bd305c071d1202bbb7d82e41bc64ce407915b79a2ca08075d5b45c2969e4`
- Train: 800 no-call rows; SHA-256
  `22fbc15b79e808fb967a5d2db4964a4a271575d2802a047f59a5520b136a3acf`
- Validation: 100 no-call rows; SHA-256
  `4b86380e69a308f8ca1db6989e2d3d99f2fca83fa33280f213ba72aa4cec3a54`
- Adapter contract: rank 16, alpha 32, zero dropout, all seven attention/MLP projection families,
  with the exact v4 adapter and manifest verified before loading
- Continuation: one epoch, learning rate `2e-5`, batch size 1, gradient accumulation 16, maximum
  length 1,024, seed `20260914`, BF16, SDPA, gradient checkpointing
- Remote output: `/opt/modeljars-alora-v3/qwen17-abstention-v6`

The fixed diagnostic gates remain 100% syntax, 100% schema, at least 85% exact tool calls, no worse
than the unadapted base, and at most 5% false calls. A diagnostic failure rejects this run without a
full qualification, Java packaging, catalog entry, or release.

## Result

The continuation completed at `2026-09-13T09:27:00Z`. The adapter weight SHA-256 is
`ae7f4813aea4c8eb47b7d82999ff4b85bc086e21633acf3657ed7cead887c536`; its training manifest
SHA-256 is `97035e81e492f7af18363eda2c1dcbd67e81c8389d4b5c1a2f594f61cb8cbcce`.

The frozen smoke rejected it. Syntax and schema validity reached 100%, and false calls fell to 0%,
but exact tool calls collapsed to 18% against the base's 86%. The full qualification, Java
packaging, catalog entry, and release were not run. The report and records are retained under
`smoke25/` with SHA-256 values
`351fb0e31c5da75a8061d06cec0681c6ab8790ea4c8e1790177e4796a839465d` and
`6a36674e7aa015fad6e87c89c2e88b8a6a46955adbbee69ace236440baa24e72`.
