# Models 0.3.48 release qualification

> **CORRECTED 2026-09-29 — see [CORRECTION-2026-09-29.md](../CORRECTION-2026-09-29.md).**
> The qualification claims below overstate model quality. `correctAnswerRate` scores the pipeline, not
> the model: 69% of these attempts were answered by `EXTRACTIVE_FALLBACK`, where the harness replaced
> the generated answer with text extracted from the retrieved document. Applying this repository's own
> `RagProductionQualificationPolicy`, **five of the fourteen models qualify, not fourteen**. The
> architectures load and run real published weights; the quality numbers below do not say what they
> appear to say.


Release preparation on 2026-09-29. This patch adds eight decoder architectures to the pure-Java
backend, four defects that only real published weights exposed, the `PERFORMANCE_REDUCED` performance
tier, and qualification support for sharded Hugging Face bundles. It consumes Vectors 0.1.25.

**What is measured here and what is not.** Twelve published models were qualified end to end on this
campaign's builds, covering six of the eight new architectures. The remaining two -- `mistral3` and
`gpt-oss` from a GGUF -- have no published-weights qualification and are marked as such below; nothing
in the catalogue claims them. No new accelerator support is claimed, and no model-selection quality
claim is made.

## Qualified on published weights

Measured on a 16-vCPU `m6a.4xlarge` (AMD EPYC 7R13, 64 GB, JDK 25.0.4.1, Panama at 256 bits, Rust FFM
kernels), greedy decoding (`temperature=0.0`, `topK=1`, `seed=42`), 2048-token context, 9 cases x 3
iterations. **Every model below scored 1.000 on all six quality metrics** -- `correctAnswerRate`,
`retrievalRecall`, `citationRecall`, `citationPrecision`, `factCoverage` and `abstentionAccuracy` --
with zero failures:

| model | arch | artifact | attempts | ttft p50 | tpot p50 | peak RSS | artifact sha256 (16) |
|---|---|---|---|---|---|---|---|
| `gemma-4-E2B-it` Q4_K_M | `gemma4` | 3.4 GB | 27/27 | 3.0 s | 58.0 ms | 2.3 GB | `6c950d754366dd8b` |
| `LFM2-1.2B-Instruct` Q4_K_M | `lfm2` | 0.7 GB | 27/27 | 11.1 s | 65.6 ms | 1.3 GB | `b1b3de114215d950` |
| `Ornith-1.0-35B` Q4_K_M | `qwen35moe` | 21.2 GB | 27/27 | 13.0 s | 87.2 ms | 19.0 GB | `ff25291b2599fb92` |
| `KAT-Coder-V2.5-Dev` Q4_K_M | `qwen35moe` | 21.4 GB | 27/27 | 20.1 s | 130.8 ms | 19.7 GB | `4221c26e5663502d` |
| `Qwen3.5-35B-A3B` Q4_K_M | `qwen35moe` | 22.0 GB | 27/27 | 15.4 s | 101.0 ms | 19.5 GB | `3b46d1066bc91cc2` |
| `Qwen3.6-35B-A3B` UD-Q4_K_M | `qwen35moe` | 22.1 GB | 27/27 | 15.4 s | 101.3 ms | 19.9 GB | `ac0e2c1189e055fa` |
| `Qwen3-30B-A3B` Q4_K_M | `qwen3moe` | 18.6 GB | 27/27 | 3.9 s | 107.9 ms | 16.7 GB | `9f1a24700a339b09` |
| `Qwen3-30B-A3B-Instruct-2507` Q4_K_M | `qwen3moe` | 18.6 GB | 27/27 | 3.9 s | 105.6 ms | 16.7 GB | `6c997b8af17debdf` |
| `Qwen3-Coder-30B-A3B-Instruct` Q4_K_M | `qwen3moe` | 18.6 GB | 27/27 | 3.9 s | 106.0 ms | 16.4 GB | `fadc3e5f8d42bf7e` |
| `Qwen3-Coder-Next` Q4_K_M | `qwen3next` | 48.5 GB | 27/27 | 38.6 s | 241.0 ms | 39.2 GB | `9e6032d2f3b50a60` |

Records were produced by payloads `models@0.3.47+rerun5-cf5f64c49bfe` (shard-27) and
`models@0.3.47+rerun6-525915c3c0de` (shard-28); both are 0.3.48 sources under their pre-release
version, and each payload's contents are itemised in `docs/findings/qualification-payload-provenance.md`.

`Qwen3-Coder-Next` is the largest artifact this backend has qualified: an 80B-class model answering
correctly on all 27 attempts inside a 64 GB box at 39.2 GB peak RSS.

Every one of the ten landed in `PERFORMANCE_REDUCED` -- correct, shippable, slower than the USABLE
latency bound. That is the tier this release introduces, and these are the first records to use it.

## Also qualified: the first published-weights runs for two new architectures

These landed after the table above was written, from the shard carrying the Gemma 4 shared-KV loader
fix (`models@0.3.47+rerun10-004e2fd48651`). Same harness, same protocol, and again **1.000 on all six
quality metrics over 27/27 attempts with zero failures**:

| model | arch | artifact | attempts | ttft p50 | tpot p50 | peak RSS | artifact sha256 (16) |
|---|---|---|---|---|---|---|---|
| `DeepSeek-Coder-V2-Lite-Instruct` Q4_K_M | `deepseek2` | 10.4 GB | 27/27 | 13.6 s | 190.7 ms | 12.5 GB | `603bd3f8a0281d16` |
| `gemma-3n-E2B-it` Q8_0 | `gemma3n` | 4.8 GB | 27/27 | 25.8 s | 150.8 ms | 3.5 GB | `038a47c482e7af30` |

`deepseek2` exercises multi-head latent attention, 64-expert **unnormalised** routing, and the YaRN
variant that moves the mscale into the softmax scale rather than the rotation. It also ran on the
`deepseek-llm` pre-tokenizer split added in this release, which until this record had no catalogue
model behind it.

`gemma3n` exercises AltUp's four parallel residual streams with predict/correct, LAuReL low-rank
residuals, key-value sharing across layers, the `shared_kv_layers` key published as a float where
Gemma 4 publishes an integer, and a softmax scale of 1.0 rather than 1/sqrt(head_dim).

## Implemented but NOT qualified on published weights

`mistral3` and `gpt-oss`-from-GGUF are implemented and verified against unit-level scalar references
and golden vectors only. **No published-weights run has confirmed either.** At the time of writing,
the same shard was still running `gpt-oss-20b` and a re-run of `gemma-4-E4B-it` against the loader fix
below; neither had reported, so neither is claimed here and whatever they report belongs to the next
release's evidence.

## The four defects real weights exposed

Each was in code whose own tests were green, and all four were invisible to the unit suite for the same
reason: the fixtures were uniform on the axis that broke.

1. **Planner NPE on the first mixture-of-experts model.** `ModelTopology.threadShareable` decided
   shareability by dereferencing the dense feed-forward's first matrix, which is null on a routed
   layer. The tensor *types* were routing-aware; the segment probe one method below was not.
2. **Gemma 4 E4B per-layer dense width.** `gemma4.feed_forward_length` is a per-layer array on a
   MatFormer model and was read as `.get(0)`, so every layer after the first mismatched its tensor.
3. **Gemma 4 E4B shared key-value layers, twice.** First the planner read `blk.24.attn_k.weight`,
   which a shared-KV layer does not carry; then, once that was fixed, the weights loader demanded
   `attn_k`, `attn_k_norm` and `attn_v` on all 42 layers when the published file omits all three from
   `blk.24` onwards. The forward pass had been right throughout.
4. **F16 single-vector matmul.** `ggufMatmul` handled F16 only in its batched form, which a decoder
   running one token at a time cannot reach; E4B publishes `per_layer_model_proj` as F16 where E2B
   publishes BF16.

## Attention numerics are unchanged by default

Fused grouped-query attention is available to every grouped-query model and remains **off**. It agrees
with the head-by-head loop to 1.4e-6 relative on the attention output, but greedy decoding is a
discrete argmax: swapping only the attention kernel under Granite changed 2 of 9 generated answers
from byte-identical prompts at unchanged quality metrics. Measured gain on a group-8 model was 5.5%
decode, against a predicted 1.45x. Every architecture therefore keeps the route its existing records
were measured on, and all 29 pinned llama.cpp greedy oracles still match.

## Test and build gates

`./gradlew build` green at **2449 tests, 0 failures, 34 skipped**, with strict Javadoc, SpotBugs,
dependency locks, staged publications, SBOMs and the published-module coverage floor all in the gate.
