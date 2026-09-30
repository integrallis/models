/*
 * Copyright 2025-2026 Integrallis Software, LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     https://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package com.integrallis.models.backend.purejava.deepseek2;

import com.integrallis.models.backend.purejava.gguf.GgufMetadata;
import java.util.Objects;

/**
 * Immutable execution shape for a DeepSeek-V2 family decoder.
 *
 * <p>Two things distinguish it from the routed decoders already here.
 *
 * <p><b>Multi-head Latent Attention.</b> Keys and values are not projected directly. One projection
 * produces a compressed latent plus a shared rotary part ({@code attn_kv_a_mqa}, {@code kvLoraRank
 * + ropeDim} wide), the latent is normalised, and a second projection ({@code attn_kv_b})
 * decompresses it into per-head keys and values. The query carries a no-rope part and a rope part
 * side by side. This decoder implements the <b>decompressing</b> form -- the one the published
 * DeepSeek-Coder-V2-Lite GGUF supports, since it carries only the unsplit {@code attn_kv_b} and
 * none of the {@code attn_k_b}/{@code attn_v_b} pair that the absorbed form needs.
 *
 * <p><b>Unnormalised expert weights.</b> The router softmaxes over all experts and the chosen
 * weights are used as they are. Qwen3.5-MoE renormalises; this family publishes no {@code
 * expert_weights_norm} key and so does not. See {@link
 * com.integrallis.models.backend.purejava.ops.ExpertRouting#selectExpertsWithoutRenormalisation}.
 *
 * @param embeddingDim model width
 * @param numLayers block count
 * @param numHeads attention heads
 * @param keyLength per-head key width, rope part included
 * @param valueLength per-head value width
 * @param kvLoraRank width of the compressed key-value latent
 * @param ropeDimension the rotary part of a key or query head
 * @param vocabSize vocabulary size
 * @param contextLength trained context
 * @param hiddenDim dense feed-forward width, used by the leading dense layers
 * @param expertHiddenDim one routed expert's width
 * @param numExperts routed expert count
 * @param numExpertsUsed how many a token selects
 * @param numSharedExperts how many always-on shared experts the layer holds. <b>Fused into one
 *     tensor</b> of {@code numSharedExperts * expertHiddenDim} width, so nothing downstream needs
 *     to know the count
 * @param expertWeightsScale a factor on the routing weights, applied only when it is neither 0 nor
 *     1
 * @param leadingDenseLayers how many leading layers are dense rather than routed
 * @param ropeTheta rope base
 * @param ropeScalingFactor the YaRN factor
 * @param ropeOriginalContext the context the model was trained at before YaRN
 * @param yarnLogMultiplier the published {@code rope.scaling.yarn_log_multiplier}
 * @param rmsNormEpsilon normalisation epsilon
 */
public record Deepseek2Config(
    int embeddingDim,
    int numLayers,
    int numHeads,
    int keyLength,
    int valueLength,
    int kvLoraRank,
    int ropeDimension,
    int vocabSize,
    int contextLength,
    int hiddenDim,
    int expertHiddenDim,
    int numExperts,
    int numExpertsUsed,
    int numSharedExperts,
    float expertWeightsScale,
    int leadingDenseLayers,
    float ropeTheta,
    float ropeScalingFactor,
    int ropeOriginalContext,
    float yarnLogMultiplier,
    float rmsNormEpsilon) {

  /** YaRN's ramp bounds, which the reference defaults rather than publishes. */
  public static final float BETA_FAST = 32.0f;

  public static final float BETA_SLOW = 1.0f;

  public Deepseek2Config {
    positive("embeddingDim", embeddingDim);
    positive("numLayers", numLayers);
    positive("numHeads", numHeads);
    positive("keyLength", keyLength);
    positive("valueLength", valueLength);
    positive("kvLoraRank", kvLoraRank);
    positive("ropeDimension", ropeDimension);
    positive("vocabSize", vocabSize);
    positive("contextLength", contextLength);
    positive("hiddenDim", hiddenDim);
    positive("expertHiddenDim", expertHiddenDim);
    positive("numExperts", numExperts);
    positive("numExpertsUsed", numExpertsUsed);
    positive("numSharedExperts", numSharedExperts);
    positive("ropeTheta", ropeTheta);
    positive("rmsNormEpsilon", rmsNormEpsilon);
    if (numExpertsUsed > numExperts) {
      throw new IllegalArgumentException(
          "numExpertsUsed must not exceed numExperts: " + numExpertsUsed + " > " + numExperts);
    }
    if (ropeDimension >= keyLength) {
      throw new IllegalArgumentException(
          "ropeDimension must leave a no-rope part: " + ropeDimension + " >= " + keyLength);
    }
    if (leadingDenseLayers < 0 || leadingDenseLayers >= numLayers) {
      throw new IllegalArgumentException(
          "leadingDenseLayers must leave at least one routed layer: " + leadingDenseLayers);
    }
  }

  /** Parses the supported DeepSeek-V2 GGUF variant. */
  public static Deepseek2Config fromMetadata(GgufMetadata metadata) {
    Objects.requireNonNull(metadata, "metadata");
    String architecture = metadata.getString("general.architecture").orElse("");
    if (!"deepseek2".equals(architecture)) {
      throw new IllegalArgumentException(
          "Expected general.architecture=deepseek2, found " + architecture);
    }
    return new Deepseek2Config(
        requiredInt(metadata, "deepseek2.embedding_length"),
        requiredInt(metadata, "deepseek2.block_count"),
        requiredInt(metadata, "deepseek2.attention.head_count"),
        requiredInt(metadata, "deepseek2.attention.key_length"),
        requiredInt(metadata, "deepseek2.attention.value_length"),
        requiredInt(metadata, "deepseek2.attention.kv_lora_rank"),
        requiredInt(metadata, "deepseek2.rope.dimension_count"),
        metadata
            .getUint32("deepseek2.vocab_size")
            .or(() -> metadata.getArraySize("tokenizer.ggml.tokens"))
            .orElseThrow(() -> new IllegalArgumentException("Missing DeepSeek-V2 vocabulary size")),
        requiredInt(metadata, "deepseek2.context_length"),
        requiredInt(metadata, "deepseek2.feed_forward_length"),
        requiredInt(metadata, "deepseek2.expert_feed_forward_length"),
        requiredInt(metadata, "deepseek2.expert_count"),
        requiredInt(metadata, "deepseek2.expert_used_count"),
        requiredInt(metadata, "deepseek2.expert_shared_count"),
        // Absent means 1.0, which is also "no scaling": the reference multiplies only when the
        // value
        // is neither 0 nor 1, so a 0 default and a 1 default behave identically.
        metadata.getFloat32("deepseek2.expert_weights_scale").orElse(1.0f),
        metadata.getUint32("deepseek2.leading_dense_block_count").orElse(0),
        requiredFloat(metadata, "deepseek2.rope.freq_base"),
        metadata.getFloat32("deepseek2.rope.scaling.factor").orElse(1.0f),
        metadata
            .getUint32("deepseek2.rope.scaling.original_context_length")
            .orElse(requiredInt(metadata, "deepseek2.context_length")),
        metadata.getFloat32("deepseek2.rope.scaling.yarn_log_multiplier").orElse(0.0f),
        requiredFloat(metadata, "deepseek2.attention.layer_norm_rms_epsilon"));
  }

  /** The no-rope part of a query or key head: what is left of {@link #keyLength}. */
  public int noRopeDimension() {
    return keyLength - ropeDimension;
  }

  public int queryDim() {
    return Math.multiplyExact(numHeads, keyLength);
  }

  /** The key-value latent projection's width: the latent plus the shared rotary part. */
  public int compressedKeyValueDim() {
    return Math.addExact(kvLoraRank, ropeDimension);
  }

  /** What {@code attn_kv_b} produces: a no-rope key and a value for every head. */
  public int decompressedKeyValueDim() {
    return Math.multiplyExact(numHeads, Math.addExact(noRopeDimension(), valueLength));
  }

  /** Full per-head keys, rope part included, as stored in the cache. */
  public int cachedKeyDim() {
    return Math.multiplyExact(numHeads, keyLength);
  }

  public int cachedValueDim() {
    return Math.multiplyExact(numHeads, valueLength);
  }

  /** The fused shared expert's width: the published tensors stack all of them into one. */
  public int sharedExpertHiddenDim() {
    return Math.multiplyExact(numSharedExperts, expertHiddenDim);
  }

  public boolean usesMixtureOfExperts(int layer) {
    requireLayer(layer);
    return layer >= leadingDenseLayers;
  }

  /** Whether the routing weights are multiplied by {@link #expertWeightsScale}. */
  public boolean scalesExpertWeights() {
    return expertWeightsScale != 0.0f && expertWeightsScale != 1.0f;
  }

  /**
   * The multiplier YaRN applies to the rotation's cosine and sine.
   *
   * <p>{@code 1 / (1 + 0.1 * ln(factor))}, which is below one. The reference arrives at it by
   * pre-dividing the rotary magnitude so that the magnitude YaRN would otherwise contribute is
   * moved into {@link #attentionScale()} instead -- see the comment at {@code deepseek2.cpp:162},
   * "we have to pre-scale kq_scale and attn_factor to make the YaRN RoPE work correctly". Taking
   * the generic YaRN magnitude here and the generic softmax scale there would be wrong twice over.
   */
  public float ropeAttentionFactor() {
    if (ropeScalingFactor <= 1.0f) {
      return 1.0f;
    }
    return (float) (1.0 / (1.0 + 0.1 * Math.log(ropeScalingFactor)));
  }

  /**
   * The attention softmax scale, which YaRN makes substantially larger than {@code
   * 1/sqrt(keyLength)}.
   *
   * <p>{@code mscale^2 / sqrt(keyLength)} where {@code mscale = 1 + yarnLogMultiplier *
   * ln(factor)}. For the published DeepSeek-Coder-V2-Lite -- factor 40, multiplier 0.0707, key
   * length 192 -- that is 0.114721, where the plain scale would be 0.072169: a factor of 1.59.
   * Using the plain scale would narrow every attention distribution.
   *
   * <p>The reference computes the multiplier as {@code 0.1 * rope_yarn_log_mul} after having
   * divided the published value by 0.1, so the two cancel and the published number is used
   * directly.
   */
  public float attentionScale() {
    double mscale = 1.0;
    if (ropeScalingFactor > 1.0f) {
      mscale = 1.0 + yarnLogMultiplier * Math.log(ropeScalingFactor);
    }
    return (float) (mscale * mscale / Math.sqrt(keyLength));
  }

  private void requireLayer(int layer) {
    if (layer < 0 || layer >= numLayers) {
      throw new IllegalArgumentException("layer out of range: " + layer);
    }
  }

  private static int requiredInt(GgufMetadata metadata, String key) {
    return metadata
        .getUint32(key)
        .orElseThrow(() -> new IllegalArgumentException("Missing " + key));
  }

  private static float requiredFloat(GgufMetadata metadata, String key) {
    return metadata
        .getFloat32(key)
        .orElseThrow(() -> new IllegalArgumentException("Missing " + key));
  }

  private static void positive(String name, int value) {
    if (value <= 0) {
      throw new IllegalArgumentException(name + " must be > 0: " + value);
    }
  }

  private static void positive(String name, float value) {
    if (!(value > 0.0f) || !Float.isFinite(value)) {
      throw new IllegalArgumentException(name + " must be finite and > 0: " + value);
    }
  }
}
