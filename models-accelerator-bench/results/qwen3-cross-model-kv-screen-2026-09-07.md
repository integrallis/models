# Qwen3 cross-model KV calibration screen — 2026-09-07

## Question

Is the qualified Qwen3 0.6B chat model and Qwen3 1.7B tool model pair structurally suitable for a
calibrated cross-model KV transfer experiment?

This is a metadata screen. It does **not** claim that the models' raw KV values are compatible, and
it does not measure transfer quality or speed.

## Immutable inputs

| Role | ModelJars coordinate | SHA-256 |
| --- | --- | --- |
| source/chat | `org.modeljars.huggingface:ggml-org.qwen3-0.6b-gguf.q4_0:3.0.0-q4_0.1` | `da2572f16c06133561ce56accaa822216f2391ef4d37fba427801cd6736417d4` |
| target/tools | `org.modeljars.huggingface:qwen.qwen3-1.7b-gguf.q8_0:3.0.0-q8_0.2` | `061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a` |

The Java 25 `KvTransferCompatibility` experiment parsed both GGUFs through Models' own reader and
fingerprinted the complete tokenizer vocabulary with SHA-256.

## Result

| Property | Qwen3 0.6B | Qwen3 1.7B |
| --- | ---: | ---: |
| layers | 28 | 28 |
| embedding width | 1,024 | 2,048 |
| query heads | 16 | 16 |
| KV heads | 8 | 8 |
| key width per head | 128 | 128 |
| value width per head | 128 | 128 |
| RoPE frequency base | 1,000,000 | 1,000,000 |
| vocabulary entries | 151,936 | 151,936 |
| vocabulary SHA-256 | `91c36cef6dcab160378c3ca60bd4f807ac7afbc93c1738d5e4dcb2d4931acf90` | `91c36cef6dcab160378c3ca60bd4f807ac7afbc93c1738d5e4dcb2d4931acf90` |

The tool reports:

```text
calibrationCandidate=true
directReuseSafe=false
```

The pair clears the prerequisites for the first aligned-token, per-head calibration experiment.
Its different hidden widths confirm why matching KV tensor shapes is not an activation-equivalence
proof.

## Preliminary linear-mapping screen

The next cheap rung used 256 calibration tokens and a separate 80-token holdout. It removed NeoX
RoPE from keys, treated each position/head pair as a sample, and compared:

- one pooled ridge map from the strongest raw-correlated source layer;
- one ridge map per KV head; and
- one pooled ridge map over the two strongest raw-correlated source layers.

The code and linear solver are Java-only and unit-tested against synthetic known transformations.
This small same-author text split is a triage screen, not the paper's 500 by 1,024-token calibration
corpus.

| target layer | top-two key cosine | key relative L2 | top-two value cosine | value relative L2 |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0.989217 | 0.146909 | 0.546728 | 0.851051 |
| 6 | 0.942339 | 0.335507 | 0.408476 | 0.942060 |
| 13 | 0.823192 | 0.571584 | 0.576366 | 0.819078 |
| 20 | 0.877302 | 0.482389 | 0.446710 | 0.922121 |
| 27 | 0.961383 | 0.275469 | 0.366539 | 0.987472 |

The keys contain strong linear structure. Values do not clear a handoff gate: the top-two mapping
stays at 0.37–0.58 cosine and 0.82–0.99 relative error. Per-head maps did not repair that result,
and selecting a second layer by raw correlation was only a marginal key improvement and sometimes
worsened values.

On the local Intel i7-9750H with Java 25 and Vectors 0.1.20, native prefill took 20.522 seconds for
the target calibration input and 6.128 seconds for its holdout input. Individual partial-map fits
took 17–120 milliseconds, but this experiment neither maps every layer nor installs translated KV;
those timings are not an end-to-end speedup claim.

**Decision:** reject the cheap single-layer and raw-correlation top-two mappers. Keep exact,
member-owned KV in the runtime. Matching geometry remains sufficient to reproduce the paper's
larger predictive layer-selection method later, but not to expose a cache-transfer API now.

## Next gate if this pair is revisited

1. Capture unrotated K/V from both models for the same pinned calibration prompts without adding a
   public cache-import API. This capture seam now exists only in the unpublished workbench.
2. Reproduce predictive top-k source-layer selection on a substantially larger, independent
   calibration corpus rather than selecting by raw correlation.
3. Compare a complete mapped state with native target prefill over held-out prompts: logits, perplexity, greedy
   continuations, five-turn tool fidelity, mapper time, target-prefill time, and peak memory.
4. Reject the pair if quality falls outside the declared tolerance or total handoff time does not
   improve. Only a passing result can justify a separately observable runtime transfer SPI.

The method follows the prerequisites described in [Cross-Model KV Cache Transfer in LLM
Families](https://arxiv.org/abs/2608.03893). The paper is a research lead, not evidence for this
specific pair.
