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

/** Immutable execution shape for a dense Qwen3.5 hybrid-attention decoder. */
public record Qwen35Config(
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
    int fullAttentionInterval) {

  public Qwen35Config {
    positive("embeddingDim", embeddingDim);
    positive("numLayers", numLayers);
    positive("numHeads", numHeads);
    positive("numKvHeads", numKvHeads);
    positive("attentionHeadDim", attentionHeadDim);
    positive("vocabSize", vocabSize);
    positive("contextLength", contextLength);
    positive("hiddenDim", hiddenDim);
    finitePositive("ropeTheta", ropeTheta);
    positive("ropeDimension", ropeDimension);
    finitePositive("rmsNormEpsilon", rmsNormEpsilon);
    positive("gdnConvKernel", gdnConvKernel);
    positive("gdnHeadDim", gdnHeadDim);
    positive("gdnKeyHeads", gdnKeyHeads);
    positive("gdnValueHeads", gdnValueHeads);
    positive("gdnInnerDim", gdnInnerDim);
    positive("fullAttentionInterval", fullAttentionInterval);
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

  public static Qwen35Config fromMetadata(GgufMetadata metadata) {
    Objects.requireNonNull(metadata, "metadata");
    String architecture = metadata.getString("general.architecture").orElse("");
    if (!"qwen35".equals(architecture)) {
      throw new IllegalArgumentException(
          "Expected general.architecture=qwen35, found " + architecture);
    }

    return new Qwen35Config(
        requiredInt(metadata, "qwen35.embedding_length"),
        requiredInt(metadata, "qwen35.block_count"),
        requiredInt(metadata, "qwen35.attention.head_count"),
        requiredInt(metadata, "qwen35.attention.head_count_kv"),
        requiredInt(metadata, "qwen35.attention.key_length"),
        metadata
            .getUint32("qwen35.vocab_size")
            .or(() -> metadata.getArraySize("tokenizer.ggml.tokens"))
            .orElseThrow(() -> new IllegalArgumentException("Missing Qwen3.5 vocabulary size")),
        requiredInt(metadata, "qwen35.context_length"),
        requiredInt(metadata, "qwen35.feed_forward_length"),
        requiredFloat(metadata, "qwen35.rope.freq_base"),
        requiredInt(metadata, "qwen35.rope.dimension_count"),
        requiredFloat(metadata, "qwen35.attention.layer_norm_rms_epsilon"),
        requiredInt(metadata, "qwen35.ssm.conv_kernel"),
        requiredInt(metadata, "qwen35.ssm.state_size"),
        requiredInt(metadata, "qwen35.ssm.group_count"),
        requiredInt(metadata, "qwen35.ssm.time_step_rank"),
        requiredInt(metadata, "qwen35.ssm.inner_size"),
        requiredInt(metadata, "qwen35.full_attention_interval"));
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
