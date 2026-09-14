# V18 result

Status: **final Phase 1 passed under the deterministic Java execution kernel; Phase 2 has not been
scored**.

The first calibration below remains historical evidence. A pre-score live-loader defect required a
new Models revision and a complete calibration rerun. That rerun passed, but comparison with the
first process exposed a different first cold score while the other 74 were bit-identical. Targeted
shared, recomputed, sequential, and base-only experiments localized the behavior to the cold
Activated-LoRA F32 path. Revision `4ebbc471553d5956970091d72f08fd79ad835a4a` added a test-first
prewarm that confirmed the localization; a further full diagnostic passed but is not qualification
evidence. The retained fix moves deterministic reduction into Vectors 0.1.21 and uses owned Java
F32 execution matrices in Models, with no load-time prewarm, external inference runtime, or Rust
kernel. See `cold-activated-determinism.md`.

The live screen remained untouched while calibration was repeated under the final Models revision
resolving Vectors 0.1.21 from Maven Central. That required rerun passed under Models revision
`55a624ca68bb62115563dfdd5b0f982149ab8396`:
threshold `2.4962309929993705`, calls `47/50`, no-calls `24/25`, balanced accuracy `0.95`, and
physical sharing `75/75`. Its report SHA-256 is
`74588d3431a7d81e9e9df2ce2eab748df7a8533ffdcd101793fff0be0f8d542b`. The report and hash
were committed before Phase 2.

The original Java 25 production-Q4 calibration completed on all 75 exposed cases. Independent
recomputation of the committed observations reproduced threshold `2.4962309929993705` and the
following result:

- positive calls: 47/50;
- no-calls: 24/25;
- balanced accuracy: 0.95; and
- physically shared immutable prefix: 75/75 cases.

The four decision errors are retained in `calibration.json`: false call `irrelevance_165`, and
missed calls `multiple_76`, `simple_326`, and `simple_94`. They are not removed or reinterpreted.
The result meets the exact predeclared Phase 1 floor without margin on either class.

Evidence SHA-256:

- `calibration.json`:
  `973aa08df9a4bde2969453ffd3d041e2eef79de6ef1be68468f37dd0c5f929e6`
- `calibration.log`:
  `9b1352791d421f17355035e360f2bae8137d2f47d995683b23e936eae06e977a`
- `clean-check.log`:
  `c25d165a87e9dbf44b28f14f439d6aad44d42cb2f04dbda192f99859df3f2ff6`

The separately frozen BFCL-live screen remains untouched at this commit. Its command must consume
the committed calibration report and its exact SHA-256 and may not change the threshold, head,
model, adapter, prompt construction, source records, or admission gates.

This is not model qualification. V18 can advance only if the live screen and every later generation,
sealed, product, framework, cache, memory, performance, packaging, clean-host, and published-artifact
gate pass.
