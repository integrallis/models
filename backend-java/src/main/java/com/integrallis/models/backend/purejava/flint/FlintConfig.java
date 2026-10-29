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
package com.integrallis.models.backend.purejava.flint;

import com.integrallis.models.backend.purejava.gguf.GgufMetadata;
import java.util.List;

/** Validated Flint execution dimensions, including independent attention and rotary patterns. */
public record FlintConfig(
    int hiddenSize,
    int numLayers,
    int numHeads,
    int numKvHeads,
    int headDim,
    int vocabSize,
    int maxPosition,
    int denseHiddenSize,
    int expertHiddenSize,
    int sharedHiddenSize,
    int numExperts,
    int expertsPerToken,
    int leadingDenseLayers,
    int slidingWindow,
    float ropeTheta,
    float rmsNormEps,
    float expertWeightsScale,
    boolean normalizeExpertWeights,
    List<Boolean> slidingPattern,
    List<Boolean> ropePattern) {

  public FlintConfig {
    int[] positive = {
      hiddenSize,
      numLayers,
      numHeads,
      numKvHeads,
      headDim,
      vocabSize,
      maxPosition,
      denseHiddenSize,
      expertHiddenSize,
      sharedHiddenSize,
      numExperts,
      expertsPerToken,
      slidingWindow
    };
    for (int value : positive) {
      if (value <= 0) throw new IllegalArgumentException("Flint dimensions must be positive");
    }
    if (numHeads % numKvHeads != 0
        || headDim % 2 != 0
        || expertsPerToken > numExperts
        || leadingDenseLayers < 0
        || leadingDenseLayers >= numLayers) {
      throw new IllegalArgumentException(
          "Invalid Flint head, expert, or dense-layer configuration");
    }
    for (float value : new float[] {ropeTheta, rmsNormEps, expertWeightsScale}) {
      if (!Float.isFinite(value) || value <= 0) {
        throw new IllegalArgumentException("Flint scales and epsilon must be finite and positive");
      }
    }
    slidingPattern = List.copyOf(slidingPattern);
    ropePattern = List.copyOf(ropePattern);
    if (slidingPattern.size() != numLayers || ropePattern.size() != numLayers) {
      throw new IllegalArgumentException(
          "Flint attention patterns must contain one flag per layer");
    }
    if ((long) numHeads * headDim > Integer.MAX_VALUE
        || (long) numKvHeads * headDim > Integer.MAX_VALUE) {
      throw new IllegalArgumentException("Flint attention widths exceed Java array limits");
    }
  }

  public static FlintConfig fromMetadata(GgufMetadata metadata) {
    if (!metadata.getString("general.architecture").orElse("").equals("flint")) {
      throw new IllegalArgumentException("Expected flint architecture");
    }
    if (integer(metadata, "expert_gating_func") != 2) {
      throw new IllegalArgumentException("Flint requires sigmoid expert gating (2)");
    }
    int head = integer(metadata, "attention.key_length");
    if (integer(metadata, "attention.value_length") != head
        || metadata.getUint32("flint.rope.dimension_count").orElse(head) != head) {
      throw new IllegalArgumentException(
          "Flint requires equal, fully rotary key/value head widths");
    }
    String scaling = metadata.getString("flint.rope.scaling.type").orElse("none");
    if (!scaling.equals("none")
        || metadata.getFloat32("flint.rope.scaling.factor").orElse(1f) != 1f) {
      throw new IllegalArgumentException("Flint scaled RoPE is not supported: " + scaling);
    }
    int shared = integer(metadata, "expert_shared_feed_forward_length");
    int expert = integer(metadata, "expert_feed_forward_length");
    if (Math.multiplyExact(integer(metadata, "expert_shared_count"), expert) != shared) {
      throw new IllegalArgumentException("Flint shared expert width does not match its count");
    }
    return new FlintConfig(
        integer(metadata, "embedding_length"),
        integer(metadata, "block_count"),
        integer(metadata, "attention.head_count"),
        integer(metadata, "attention.head_count_kv"),
        head,
        metadata
            .getArraySize("tokenizer.ggml.tokens")
            .orElseThrow(() -> new IllegalArgumentException("Missing Flint vocabulary")),
        integer(metadata, "context_length"),
        integer(metadata, "feed_forward_length"),
        expert,
        shared,
        integer(metadata, "expert_count"),
        integer(metadata, "expert_used_count"),
        integer(metadata, "leading_dense_block_count"),
        integer(metadata, "attention.sliding_window"),
        floating(metadata, "rope.freq_base"),
        floating(metadata, "attention.layer_norm_rms_epsilon"),
        metadata.getFloat32("flint.expert_weights_scale").orElse(1f),
        metadata
            .getBool("flint.expert_weights_norm")
            .orElseThrow(() -> new IllegalArgumentException("Missing flint.expert_weights_norm")),
        pattern(metadata, "attention.sliding_window_pattern"),
        pattern(metadata, "attention.rope_pattern"));
  }

  public int queryDimension() {
    return numHeads * headDim;
  }

  public int keyValueDimension() {
    return numKvHeads * headDim;
  }

  public boolean usesSlidingAttention(int layer) {
    return slidingPattern.get(layer);
  }

  public boolean usesRope(int layer) {
    return ropePattern.get(layer);
  }

  public int attentionStartPosition(int layer, int position) {
    return usesSlidingAttention(layer) ? Math.max(0, position - slidingWindow + 1) : 0;
  }

  private static int integer(GgufMetadata metadata, String name) {
    return metadata
        .getUint32("flint." + name)
        .orElseThrow(() -> new IllegalArgumentException("Missing flint." + name));
  }

  private static float floating(GgufMetadata metadata, String name) {
    return metadata
        .getFloat32("flint." + name)
        .orElseThrow(() -> new IllegalArgumentException("Missing flint." + name));
  }

  private static List<Boolean> pattern(GgufMetadata metadata, String name) {
    return metadata
        .getBoolArray("flint." + name)
        .orElseThrow(() -> new IllegalArgumentException("Missing flint." + name));
  }
}
