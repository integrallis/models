# Decoder architectures — ALL IMPLEMENTED as of 2026-09-29

**Closed out 2026-09-29.** Seven of the eight are now qualified on published weights: `lfm2`,
`qwen3moe`, `qwen35moe`, `qwen3next`, `deepseek2`, `gemma3n` and `gpt-oss` from a GGUF, alongside
`gemma4` E2B and E4B. Only `mistral3` has no published-weights run behind it. Fourteen models in total,
each at 1.000 on all six quality metrics over 27/27 attempts with zero failures.

Every architecture below is now implemented and unit-verified. `qwen3next` is **qualified** on
`Qwen3-Coder-Next` Q4_K_M (48.5 GB, 27/27 attempts, every quality metric 1.000) from shard-28.
`gemma3n`, `deepseek2` and the `gpt-oss` GGUF are queued as shard-30 on payload
`models-rag-bench-0.3.47-rerun9.tar` (sha `c19c4b41bf4a`), together with a re-run of gemma 4 E4B
after the shared-KV weights-loader fix that shard-28 exposed.

The scoping notes are kept below because they record what was measured from the real headers and which
details were load-bearing.

---

# Original scoping, from the published GGUF headers

Read on 2026-09-29 by range-fetching the real files (14 MB prefix each). Everything below is
*measured from the header*, not inferred from the architecture name.

---

## gemma3n — `gemma_3n_e2b_it_q8_0` (1 model)

30 layers, 727 tensors. Structurally **gemma4 E-series plus two new mechanisms**, so the existing
Gemma 4 decoder already covers per-layer input embeddings (`inp_gate`/`proj`/`post_norm`), shared KV
layers, and the sliding-window pattern.

New work:

1. **AltUp** — `altup.num_inputs = 4`, `altup.active_idx = 0`. Per layer: `altup_router.weight`
   `[2048, 4]`, `altup_router_norm.weight`, `altup_predict_coef.weight` `[4, 16]`,
   `altup_correct_coef.weight` `[4, 4]`, `altup_correct_scale.weight` `[2048]`. Four parallel
   residual streams; one is "active" and the others are predicted from it and then corrected.
2. **LAuReL** — `laurel_l.weight` `[2048, 64]`, `laurel_r.weight` `[64, 2048]`,
   `laurel_post_norm.weight`. A low-rank learned residual added alongside attention.
3. **Per-layer activation sparsity** — `gemma3n.activation_sparsity_scale` is a 30-entry float array
   (1.6448533535003662 on the early layers). Applied as a threshold on the FFN activation.

**Trap already found:** `gemma3n.attention.shared_kv_layers = 10.0` is a **float**, where the gemma4
loader reads `gemma4.attention.shared_kv_layers` with `getUint32`. Reading it as an integer key will
fail or return empty.

`altup_router` and the two `laurel_*` matrices are **F16** (type 1) while the rest are Q8_0 — so
`Gemma4Weights.MATRIX_TYPES` needs F16, which the 2026-09-29 E4B fix already added.

---

## deepseek2 — `bartowski_deepseek_coder_v2_lite_instruct_gguf_q4_k_m` (1 model)

27 layers, 377 tensors. The largest of the remaining jobs: a new attention mechanism.

1. **Multi-head Latent Attention (MLA).** Not standard GQA:
   - `attn_q.weight` `[2048, 3072]` = 16 heads x 192 (`key_length = 192` = 128 no-rope + 64 rope)
   - `attn_kv_a_mqa.weight` `[2048, 576]` = `kv_lora_rank` 512 + 64 rope dims
   - `attn_kv_a_norm.weight` `[512]` — RMS norm on the compressed latent
   - `attn_kv_b.weight` `[512, 4096]` — decompresses the latent to 16 x (128 K-nope + 128 V)
   - The **KV cache holds the 512-wide latent plus the 64 rope dims**, not K and V. That is the point
     of MLA and it means a different cache layout, not just a different projection.
2. **A different MoE shape** — `expert_count = 64`, `expert_used_count = 6`,
   `expert_shared_count = 2` (two *always-on* shared experts, where Qwen3.5 has one),
   `expert_weights_scale = 1.0` (present but 1.0, so no scaling -- the reference applies a scale only
   when it is neither 0 nor 1), `leading_dense_block_count = 1` (layer 0 is dense, the rest routed).
3. **YaRN rope scaling** — `rope.scaling.type = yarn`, `factor = 40.0`,
   `original_context_length = 4096`, `yarn_log_multiplier = 0.0707`.

---

## gpt-oss GGUF loader — `unsloth_gpt_oss_20b_gguf_q4_k_m` (1 model)

**Corrected 2026-09-29.** An earlier note called this "loader plumbing rather than new mathematics".
That was wrong, and the header says why. The mathematics is indeed already written and tested
(`gptoss/GptOssForwardPass`, `GptOssMath`, `GptOssMxfp4Moe` -- attention sinks and expert biases
included), and `GgufTensorSource` already exists. But the GGUF **weight layout** is a different shape
from the safetensors one the existing loader reads, in three ways:

1. **gate and up are separate in GGUF** (`ffn_gate_exps.weight`, `ffn_up_exps.weight`) where Hugging
   Face fuses them into `gate_up_proj`. `GptOssMxfp4ExpertWeights.load` reads the fused form.
2. **MXFP4 is one interleaved blob in GGUF** (tensor type 39) where Hugging Face ships
   `*_blocks` and `*_scales` as separate tensors. The existing expert loader reads the split form.
3. **Attention matrices are K-quants in GGUF** (`attn_q`/`attn_k` Q5_0, `attn_output` Q4_K, `attn_v`
   Q8_0), while `GptOssWeights.Layer` is typed to hold `BFloat16Matrix`. So the record itself has to
   become type-agnostic, or a parallel GGUF weights class has to exist.

It also carries what the header confirms and the HF path already implements: `attn_sinks.weight`
`[64]` (one learned logit per head in the softmax denominator), biases on q/k/v/output, a router bias
`ffn_gate_inp.bias`, per-expert biases `ffn_*_exps.bias` `[2880, 32]`, and YaRN scaling
(`factor = 32.0`, `original_context_length = 4096`).

Scope is therefore comparable to gemma3n: a second weights layer, not a name mapping.

### Progress and the blocking detail (2026-09-29)

**Done:** `GptOssProjection` now abstracts "one weight matrix" behind `multiply` and `row`, so
`GptOssWeights.Layer` is format-agnostic and the 379-line forward pass does not have to be duplicated.
The safetensors path is wrapped at its loading boundary and all 24 gpt-oss tests pass unchanged.

**What remains, and why it is not a small job.** `GptOssMath.swigluOai` reads the fused projection
**interleaved per pair** -- `gateUp[2*i]` is the gate and `gateUp[2*i+1]` the up -- because that is how
Hugging Face stores `gate_up_proj`. GGUF stores `ffn_gate_exps` and `ffn_up_exps` as two separate
tensors. So one of:

- interleave the two GGUF tensors into one buffer per expert, which gives up the strict zero-copy
  mapping that `GptOssMxfp4ExpertWeights` exists to provide; or
- change `Expert` to hold gate and up separately and teach `GptOssMxfp4Moe` to read them separately,
  with the safetensors loader supplying stride-2 row views into its interleaved tensor. This needs
  strided row support in `Mxfp4Matrix` and changes the tested core of a decoder that currently works.

### Completed 2026-09-29

Resolved by the **second** option: `Expert` now carries either shape, and `GptOssMxfp4Moe` branches on
it. No weights are copied or reformatted, and `Mxfp4Matrix` (which lives in the `vectors` project and
cannot be changed from here) is untouched.

- `GptOssMath.swigluOaiSplit` is the split-input twin of `swigluOai`. Two copies of one formula, so
  `swigluOaiMatchesItsSplitForm` pins them equal.
- `GptOssMxfp4ExpertWeights.fromGguf` slices the three stacked tensors per expert and hands each over as
  a `GptOssProjection`, which routes to `ggufMatmul`'s MXFP4 case -- GGUF's interleaved block format
  needs no second reader.
- `GptOssConfig.fromGgufMetadata` fills the fields only `config.json` carries. The one that matters is
  `layerTypes`, which **is** read: GGUF publishes the window size but no pattern, so it is derived as
  "even layers slide" from `openai-moe.cpp`, where `swa_period = 2` and `set_swa_pattern` leaves
  `il % 2 < 1` sliding.
- `GptOssHuggingFaceConfig` was renamed `GptOssConfig`, since it now serves both formats and the old name
  would have been a lie.

**A test-quality note worth keeping.** Three mutations of this path were initially caught by nothing:
loading zero sinks, tying the head despite `output.weight`, and reading the up stack as the gate. The
first two needed perturbation tests; the third needed a **golden vector**, because every other test only
asserted that the output was finite or that it changed. The fused/split equivalence test was also
vacuous at first -- the existing MoE fixture fills every row of the gate/up tensor identically, so
splitting by row parity and splitting by halves produce the same two matrices.

---

## Verified non-issues

- Neither `qwen35moe` nor `qwen3next` publishes `expert_weights_scale`, `expert_shared_count` or
  `norm_topk_prob`. The Qwen3.5 routed implementation's assumption of "no expert-weight scale, one
  shared expert, renormalised top-k" is therefore correct for every published file in that family.
