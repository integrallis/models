# General-v3 training result

Completed 10,000 full-parameter training steps with capped rare-label balancing. All 486,444,053 parameters were trainable. Selection used the minimum normalized development gate score, not qualification outcomes.

Selected checkpoint: step 9000. Exported F32 model SHA-256: `deecc20819ee76accd6ab5ac9bbb91b63aee9e70a6df8b1e9c4678052fc2575b`. Training time: 1450.8 seconds.

Development emotion micro-F1: 0.562701; macro-F1: 0.562993. BoolQ accuracy: 0.831650. These emotion metrics do not pass the registered 0.70/0.60 improvement targets. The untouched 1,000-case qualification set was not evaluated.

The worker exported and uploaded the snapshot successfully, then held its terminal state until the controller deleted the pod. Independent API listing confirms the owned pod is absent. All upload manifests, selected-checkpoint provenance, development history and cleanup receipts are retained. Merged bytes have not yet been independently hash-verified on a fresh worker.

This was a combined intervention (full-parameter adaptation, capped balancing, and development-gate selection). The result does not establish which component caused any change. The admission targets are local policy targets, not publisher-reported emotion performance; see ../EMOTION-BASELINE-CONTEXT.md.
