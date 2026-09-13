# Qwen3 1.7B rank-32 full-corpus aLoRA preflight

Status: **frozen before training; candidate rejected by the fixed development smoke**.

This run tests the PEFT aLoRA guidance that activated adapters often need more capacity than a
standard LoRA. The rejected rank-16 run consumed only the first 4,000 usable examples. V9 starts
from the unmodified base, doubles the rank, and consumes the complete reproducible preparation. It
does not continue from, interpolate with, or select checkpoints from any rejected adapter.

## Model and training contract

- Base: `Qwen/Qwen3-1.7B`
- Base revision: `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`
- Invocation string: `<|im_start|>assistant\n`
- Invocation tokens: `[151644, 77091, 198]`
- Target modules: `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, and
  `down_proj`
- Rank: 32
- Alpha: 64
- Dropout: 0
- Epochs: 1
- Learning rate: `1e-4`, cosine schedule, 3% warm-up
- Batch size: 1; gradient accumulation: 16
- Maximum sequence length: 1,024
- Seed: `20260917`
- Precision: BF16 with SDPA; TF32 disabled
- Gradient checkpointing: enabled
- Trainer SHA-256:
  `bcf5942e8689aa92c02998ce858b56580c728c1fce7e23451578b5c75e05ad8c`

## Reproduced training inputs

The source files were downloaded from their pinned Hugging Face revisions and the preparation was
regenerated locally. All three generated files reproduced the earlier evidence byte-for-byte.

- BFCL fine-tuning source:
  `915679b445c64676d7212e8953a0c3379c1024235e3c8168d1af2d2ef2ae7973`
- Hammer irrelevance source:
  `2e6f3d0adbd40248a592ea001e3f3a4f1624a5d50f434fa1e7d019e93e922327`
- Preparation script:
  `a66aaa2aacdecfaf71809085770ccd1ac11601e7e915e308353386579477b696`
- Preparation manifest:
  `c7da1f5f826bb1c6f10047e09ff1e7a69e3346dad2336400e5a277ef937090b9`
- Prepared training split, 12,000 rows:
  `3fe555c1e4a68b65b6715341cd1d1cbf9995e549cbdb267f15896b0a9b7edf77`
- Prepared validation split, 1,000 rows:
  `f12c4c34c875d929b252d075749468f2c53d919744eca711e907927d9ecd83ca`

The training script filters examples exceeding 1,024 tokens and records the exact consumed source
lines and hashes in the resulting training manifest.

## Unseen qualification window

The earlier runs exposed the 25 lowest-ranked cases of each BFCL v3 category. V9 will not reuse
those prompts. `qualification-window.json` freezes 100 cases per category after excluding every
query fingerprint from the exposed window and every duplicate query fingerprint across categories.
The preparation manifest independently proves that all three source files were excluded from
training before sampling.

- Window manifest SHA-256:
  `5fee70a0d18801ac9541b2cfa3b504c0fd06e98f5bf1adeb196adfc89dffa1d2`
- Selected case-set SHA-256:
  `94c951aa277cee1e3f5ba454323e28b81cbfa357eb3d25d23a2e6ca581d1f132`
- Freeze script SHA-256:
  `1c3ad7677b88eeb9a0d67adaedce9f80564a6278f6413e25102e4f08b370e7b1`
- Initial evaluator SHA-256:
  `8c5528f648e3459c276e7b49c811fd10f8bb5e6edaf9809df61ab738f371d9c5`
- Query-bound evaluator SHA-256:
  `d60e5f96f0dd3c65f0dc56001dda803026d7950dac92493670643ff5b9415e14`

Before training produced a candidate result, the evaluator was tightened to require the frozen
manifest for every qualifying run. It verifies the exact BFCL repository revision and file hashes,
recomputes the case-set hash, resolves the exact case IDs, verifies every query fingerprint, and
rejects duplicate IDs or queries. The change does not alter the frozen cases, expected answers, or
qualification thresholds. The manifest resolved to exactly 300 cases and reproduced both frozen
hashes on the training host; the evaluator, freezer, and development-leakage suite has 18 passing
tests.

The fixed gates remain 100% syntax, 100% schema, at least 85% exact tool calls, no worse than the
unadapted base on the same cases, and at most 5% false tool calls. The first screen reuses the
previously exposed 25 cases per category as development data; those queries are excluded from
training. The new frozen window remains unread until a candidate passes that screen. A failure
rejects the candidate without JVM packaging, catalog publication, or threshold changes.

## JVM shared-state gates frozen before candidate output

While training was still running and before it emitted an adapter checkpoint or any evaluation
result, the production-runtime gates were completed and frozen. In addition to exact physical
storage identity and complete memory accounting, the crossover gate now requires the shared and
independently recomputed branches to emit the same token at 256, 1,024, and 4,096 prefix tokens.

A separate eight-case workload places an opaque archive code near the beginning of an exact
4,096-token tool prompt. After tool selection and a fixed tool result, the exact base branch must
recover that early code and the result value. Qualification requires:

- physical KV sharing and an actual 4,096-token shared prefix in all eight cases;
- correct tool selection in all eight cases;
- at least six native-correct base answers;
- retention of every answer the native base gets right; and
- byte-identical shared and independently evaluated base output in all eight cases.

Frozen production-gate identities:

- Long-context suite:
  `19809744a2ce14ab09598cf12636d71219b7f7cba95e177174d0abd880813b6c`
- Long-context evaluator:
  `5d43ffca36ea3cb853934e41c41a42c7e73f71b7b08b6cca15a77ab4e868d485`
- Prefix-sharing crossover evaluator:
  `d64974a8a8ab591cba40ce0eba4b70ff907c5735602f2c47ed06d974c3ae0d20`
- Production tool-loop evaluator:
  `4fa9ad33a35ffd2c77819932ce64ab1524f3679082d0018596da4f99ec848efa`

These gates were added without inspecting candidate output. Their cases, thresholds, and scoring
must not change for V9. The production tool follow-up specifically executes the initial activated
selection, appends the tool result, asks the activated specialist whether another tool is needed,
and only then lets the exact base branch narrate the result. That is the same two-stage behavior
used by the Spring AI and LangChain4j adapters; a repeated call after the result fails the gate.
