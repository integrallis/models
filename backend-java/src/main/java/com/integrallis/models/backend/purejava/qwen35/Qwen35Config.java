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
package com.integrallis.models.backend.purejava.qwen35;

import com.integrallis.models.backend.purejava.gguf.GgufMetadata;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Immutable execution shape for a Qwen3.5-family hybrid-attention decoder.
 *
 * <p>Covers the dense, routed and Qwen3-Next variants; see {@link #ARCHITECTURES}.
 *
 * @param architecture the {@code general.architecture} this was parsed from. Carried rather than
 *     discarded because it is what the model reports as its family. Derived from the feed-forward
 *     shape instead -- routed means qwen35moe -- it would call every Qwen3-Next a Qwen3.5-MoE, and
 *     qualification records results against the reported name.
 */
public record Qwen35Config(
    String architecture,
    int embeddingDim,
    int numLayers,
    int numHeads,
    int numKvHeads,
    int attentionHeadDim,
    int vocabSize,
    int contextLength,
    int hiddenDim,
    float ropeTheta,
    int ropeDimension,
    float rmsNormEpsilon,
    int gdnConvKernel,
    int gdnHeadDim,
    int gdnKeyHeads,
    int gdnValueHeads,
    int gdnInnerDim,
    int fullAttentionInterval,
    int numExperts,
    int numExpertsUsed,
    int expertHiddenDim,
    int sharedExpertHiddenDim) {

  /**
   * The architectures this contract covers.
   *
   * <p>One decoder, three names. They publish the same hybrid keys under their own prefix and
   * differ only in the feed-forward half and, for Qwen3-Next, in how the Gated DeltaNet packs beta
   * and alpha. Verified from the published GGUF headers of all three rather than assumed from the
   * naming.
   */
  static final List<String> ARCHITECTURES = List.of("qwen35", "qwen35moe", "qwen3next");

  public Qwen35Config {
    if (!ARCHITECTURES.contains(architecture)) {
      throw new IllegalArgumentException(
          "architecture must be one of " + ARCHITECTURES + ": " + architecture);
    }
    positive("embeddingDim", embeddingDim);
    positive("numLayers", numLayers);
    positive("numHeads", numHeads);
    positive("numKvHeads", numKvHeads);
    positive("attentionHeadDim", attentionHeadDim);
    positive("vocabSize", vocabSize);
    positive("contextLength", contextLength);
    // A fully routed Qwen3.5 publishes no feed_forward_length at all -- every layer is expert -- so
    // a
    // zero dense width is a shape, not a malformed file. It stays required for the dense variant.
    if (numExperts == 0) {
      positive("hiddenDim", hiddenDim);
    } else if (hiddenDim < 0) {
      throw new IllegalArgumentException("hiddenDim must not be negative, was " + hiddenDim);
    }
    finitePositive("ropeTheta", ropeTheta);
    positive("ropeDimension", ropeDimension);
    finitePositive("rmsNormEpsilon", rmsNormEpsilon);
    positive("gdnConvKernel", gdnConvKernel);
    positive("gdnHeadDim", gdnHeadDim);
    positive("gdnKeyHeads", gdnKeyHeads);
    positive("gdnValueHeads", gdnValueHeads);
    positive("gdnInnerDim", gdnInnerDim);
    positive("fullAttentionInterval", fullAttentionInterval);
    // All four expert fields together or none. A half-declared mixture would be treated as dense
    // and
    // would silently skip the routed feed-forward, which is the whole of the model's capacity.
    if (!(numExperts == 0
        && numExpertsUsed == 0
        && expertHiddenDim == 0
        && sharedExpertHiddenDim == 0)) {
      positive("numExperts", numExperts);
      positive("numExpertsUsed", numExpertsUsed);
      positive("expertHiddenDim", expertHiddenDim);
      positive("sharedExpertHiddenDim", sharedExpertHiddenDim);
      if (numExpertsUsed > numExperts) {
        throw new IllegalArgumentException(
            "numExpertsUsed must not exceed numExperts: " + numExpertsUsed + " > " + numExperts);
      }
    }
    if (numHeads % numKvHeads != 0) {
      throw new IllegalArgumentException("numHeads must be divisible by numKvHeads");
    }
    if (ropeDimension > attentionHeadDim || (ropeDimension & 1) != 0) {
      throw new IllegalArgumentException("ropeDimension must be even and fit an attention head");
    }
    if (gdnValueHeads % gdnKeyHeads != 0) {
      throw new IllegalArgumentException("Gated DeltaNet value heads must divide by key heads");
    }
    int valueHeadDimension = gdnInnerDim / gdnValueHeads;
    if (gdnInnerDim % gdnValueHeads != 0 || valueHeadDimension != gdnHeadDim) {
      throw new IllegalArgumentException("Gated DeltaNet key and value head dimensions must match");
    }
  }

  /** Whether the feed-forward is routed to experts rather than dense. */
  public boolean usesMixtureOfExperts() {
    return numExperts > 0;
  }

  /** Recurrent state width per expert slice, for the routed feed-forward. */
  public int expertSliceElements() {
    return expertHiddenDim * embeddingDim;
  }

  public static Qwen35Config fromMetadata(GgufMetadata metadata) {
    Objects.requireNonNull(metadata, "metadata");
    String architecture = metadata.getString("general.architecture").orElse("");
    if (!ARCHITECTURES.contains(architecture)) {
      // A List rather than a Set: this text is a message a user reads, and Set.of iterates in an
      // unspecified order, so the same refusal would list its alternatives differently per run.
      throw new IllegalArgumentException(
          "Expected general.architecture one of " + ARCHITECTURES + ", found " + architecture);
    }
    // Both variants publish the same hybrid keys under their own prefix; only the feed-forward half
    // differs. Verified against a Kwaipilot KAT-Coder-V2.5-Dev header, which carries the full ssm.*
    // set, full_attention_interval and the attention keys exactly as qwen35 does.
    String prefix = architecture + ".";

    return new Qwen35Config(
        architecture,
        requiredInt(metadata, prefix + "embedding_length"),
        requiredInt(metadata, prefix + "block_count"),
        requiredInt(metadata, prefix + "attention.head_count"),
        requiredInt(metadata, prefix + "attention.head_count_kv"),
        requiredInt(metadata, prefix + "attention.key_length"),
        metadata
            .getUint32(prefix + "vocab_size")
            .or(() -> metadata.getArraySize("tokenizer.ggml.tokens"))
            .orElseThrow(() -> new IllegalArgumentException("Missing Qwen3.5 vocabulary size")),
        requiredInt(metadata, prefix + "context_length"),
        // Absent on a fully routed model, where the routed width replaces it.
        metadata.getUint32(prefix + "feed_forward_length").orElse(0),
        requiredFloat(metadata, prefix + "rope.freq_base"),
        requiredInt(metadata, prefix + "rope.dimension_count"),
        requiredFloat(metadata, prefix + "attention.layer_norm_rms_epsilon"),
        requiredInt(metadata, prefix + "ssm.conv_kernel"),
        requiredInt(metadata, prefix + "ssm.state_size"),
        requiredInt(metadata, prefix + "ssm.group_count"),
        requiredInt(metadata, prefix + "ssm.time_step_rank"),
        requiredInt(metadata, prefix + "ssm.inner_size"),
        requiredInt(metadata, prefix + "full_attention_interval"),
        metadata.getUint32(prefix + "expert_count").orElse(0),
        metadata.getUint32(prefix + "expert_used_count").orElse(0),
        metadata.getUint32(prefix + "expert_feed_forward_length").orElse(0),
        metadata.getUint32(prefix + "expert_shared_feed_forward_length").orElse(0));
  }

  public int attentionQueryDim() {
    return Math.multiplyExact(numHeads, attentionHeadDim);
  }

  public int attentionKeyDim() {
    return Math.multiplyExact(numKvHeads, attentionHeadDim);
  }

  int gdnKeyDim() {
    return Math.multiplyExact(gdnKeyHeads, gdnHeadDim);
  }

  int gdnValueDim() {
    return gdnInnerDim;
  }

  int gdnConvDim() {
    return Math.addExact(Math.multiplyExact(2, gdnKeyDim()), gdnValueDim());
  }

  boolean usesFullAttention(int layer) {
    requireLayer(layer);
    return (layer + 1) % fullAttentionInterval == 0;
  }

  List<Integer> fullAttentionLayers() {
    List<Integer> result = new ArrayList<>();
    for (int layer = 0; layer < numLayers; layer++) {
      if (usesFullAttention(layer)) {
        result.add(layer);
      }
    }
    return List.copyOf(result);
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

  private static void finitePositive(String name, float value) {
    if (!(value > 0.0f) || !Float.isFinite(value)) {
      throw new IllegalArgumentException(name + " must be finite and > 0: " + value);
    }
  }
}
