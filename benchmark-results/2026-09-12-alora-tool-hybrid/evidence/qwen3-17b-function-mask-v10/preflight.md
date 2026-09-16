# Qwen3 1.7B function-masked rank-32 aLoRA preflight

Status: **aborted by the startup corpus-retention gate; no candidate was evaluated**.

V9 selected tools accurately enough to clear the exact-call threshold, but made false calls on
24% of the fixed irrelevance screen. Its six parseable false calls consistently followed
suggestive callable names into tools whose descriptions did not cover the requested operation.
V10 tests one bounded change: deterministic Hammer-style masking of callable and argument names in
two thirds of the frozen preparation. Descriptions, schemas, user requests, argument values, and
expected call relationships remain intact. The intended effect is to make the adapter learn the
declared capability boundary instead of matching surface words in API names.

## Frozen model and training contract

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

These values are identical to V9. V10 does not continue from, interpolate with, or select a
checkpoint from any rejected adapter.

## Frozen preparation

- Preparation seed: `20260912`
- Function-mask fraction: `0.6666666666666666`
- Mask selection: unsigned first 64 bits of
  `SHA-256(seed:function-mask-selection:record-fingerprint)`
- Identifier derivation:
  `SHA-256(seed:kind:record-fingerprint:function-name:source-name)`
- Masked training rows: 7,955 of 12,000
- Masked validation rows: 644 of 1,000
- No-tool share: 19% in each split
- BFCL fine-tuning source SHA-256:
  `915679b445c64676d7212e8953a0c3379c1024235e3c8168d1af2d2ef2ae7973`
- Hammer irrelevance source SHA-256:
  `2e6f3d0adbd40248a592ea001e3f3a4f1624a5d50f434fa1e7d019e93e922327`
- Preparation script SHA-256:
  `40431c7f53bf81d485a13d1c05612b2a3ae8c64df0a3bf2d3299695c7ad4ea8b`
- Preparation manifest SHA-256:
  `4468d20e3fea0cea1497a189f8d733c750016863ce07308a2713f12a294b33e4`
- Training split SHA-256:
  `36aaf299a126eefd190c69f3c433270f04245a24d2794d4985a8fb24c692dd05`
- Validation split SHA-256:
  `d8556d3ab982b27a16c0e30377b02959ceaaeba4cf48b3750977b9773feb7e89`

Two independent local preparations reproduced all three generated files byte-for-byte. The
validation population has the exact same 1,000 ordered source-line identities as V9. Preparation
now rejects 165 source records that declare the same callable name with incompatible schemas;
21 such selected training rows are replaced deterministically. Compatible duplicate declarations
remain in place so their V9 fingerprints and selection priorities are preserved. This hygiene fix
and identifier masking are the only preparation differences from V9.

The masking implementation is derived from the useful mechanism in MadeAgents/Hammer
`train/data_processing.py`, pinned at repository revision
`ff415d9998180b5a68bbfdda3309ec04b472fb49` and file SHA-256
`bf9b79c249b444a25a36d0d43ab031faf2ba45bf3041fa3cb0932f464db579d9`.
V10 deliberately does not copy Hammer's random names, duplicate expansion, data shuffling, prompt
format, or external runtime.

## Unchanged evaluation and stop rules

The fixed 25-case-per-kind development smoke remains the first result-bearing gate. It requires
100% syntax, 100% schema validity, at least 85% exact tool calls, exact-call performance no worse
than the unadapted base, and at most 5% false tool calls. Any failure rejects V10 immediately.

Only a smoke pass permits scoring the still-unscored 300-case qualification window frozen in
`../qwen3-17b-r32-full-v9/qualification-window.json`:

- Window manifest SHA-256:
  `5fee70a0d18801ac9541b2cfa3b504c0fd06e98f5bf1adeb196adfc89dffa1d2`
- Selected case-set SHA-256:
  `94c951aa277cee1e3f5ba454323e28b81cbfa357eb3d25d23a2e6ca581d1f132`
- Evaluator SHA-256:
  `d60e5f96f0dd3c65f0dc56001dda803026d7950dac92493670643ff5b9415e14`

Passing both Python gates still does not qualify the artifact. The exact Java oracle, real Spring
AI and LangChain4j tool loops, conversational synthesis, disabled-adapter no-op, physical shared-KV
identity, memory accounting, 4,096-token retention, recompute comparison, clean-host run, and
published-artifact resolution gates remain mandatory and unchanged.

## Startup result

The first optimizer step exposed that the 16-hex-character masked identifiers increased prompt
tokenization enough to reduce usable training rows from V9's 11,827 to approximately 10,896. PID
3442 was terminated immediately. No diagnostic or qualification prompt was run and no V10 adapter
was serialized. Changing corpus retention after seeing this input-only property would violate the
frozen V10 contract, so the corrected short-identifier preparation is a separate V11 experiment.
