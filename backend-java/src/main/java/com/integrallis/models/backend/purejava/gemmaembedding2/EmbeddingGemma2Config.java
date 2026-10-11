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
package com.integrallis.models.backend.purejava.gemmaembedding2;

import com.integrallis.models.backend.purejava.gguf.GgufMetadata;
import java.util.Collections;
import java.util.List;

/** Text encoder metadata, including its per-layer attention shapes and projected output width. */
public record EmbeddingGemma2Config(
    int embeddingDim,
    int outputDim,
    int numLayers,
    int numHeads,
    List<Integer> kvHeads,
    int fullHeadDim,
    int slidingHeadDim,
    int hiddenDim,
    int perLayerDim,
    int vocabSize,
    int contextLength,
    int slidingWindow,
    List<Boolean> slidingLayers,
    float fullRopeTheta,
    float slidingRopeTheta,
    float epsilon) {
  private static final String PREFIX = "gemma-embedding2.";

  public EmbeddingGemma2Config {
    for (int value :
        new int[] {
          embeddingDim,
          outputDim,
          numLayers,
          numHeads,
          fullHeadDim,
          slidingHeadDim,
          hiddenDim,
          perLayerDim,
          vocabSize,
          contextLength,
          slidingWindow
        }) {
      if (value <= 0) throw new IllegalArgumentException("encoder dimensions must be positive");
    }
    for (float value : new float[] {fullRopeTheta, slidingRopeTheta, epsilon}) {
      if (!(value > 0) || !Float.isFinite(value)) {
        throw new IllegalArgumentException(
            "encoder floating parameters must be finite and positive");
      }
    }
    kvHeads = List.copyOf(kvHeads);
    slidingLayers = List.copyOf(slidingLayers);
    if (kvHeads.size() != numLayers || slidingLayers.size() != numLayers) {
      throw new IllegalArgumentException("attention arrays must contain one entry per layer");
    }
    for (int heads : kvHeads) {
      if (heads <= 0 || numHeads % heads != 0) {
        throw new IllegalArgumentException("KV head count must divide query head count");
      }
    }
    if ((fullHeadDim & 1) != 0 || (slidingHeadDim & 1) != 0) {
      throw new IllegalArgumentException("rotary head widths must be even");
    }
  }

  /** Reads the text encoder contract; other pooling, KV sharing and partial RoPE are rejected. */
  public static EmbeddingGemma2Config fromMetadata(GgufMetadata metadata) {
    if (!metadata.getString("general.architecture").orElse("").equals("gemma-embedding2")) {
      throw new IllegalArgumentException("Expected general.architecture=gemma-embedding2");
    }
    if (metadata.getBool(PREFIX + "attention.causal").orElse(false)
        || integer(metadata, "pooling_type") != 1
        || metadata.getUint32(PREFIX + "attention.shared_kv_layers").orElse(0) != 0) {
      throw new IllegalArgumentException(
          "EmbeddingGemma 2 requires non-causal mean pooling without KV sharing");
    }
    int layers = integer(metadata, "block_count");
    int full = integer(metadata, "attention.key_length");
    int sliding = integer(metadata, "attention.key_length_swa");
    if (full != integer(metadata, "attention.value_length")
        || sliding != integer(metadata, "attention.value_length_swa")
        || full != integer(metadata, "rope.dimension_count")
        || sliding != integer(metadata, "rope.dimension_count_swa")) {
      throw new IllegalArgumentException(
          "EmbeddingGemma 2 requires equal Q/K/V widths and full-head RoPE");
    }
    List<Integer> heads =
        metadata
            .getInt32Array(PREFIX + "attention.head_count_kv")
            .orElseGet(
                () -> Collections.nCopies(layers, integer(metadata, "attention.head_count_kv")));
    return new EmbeddingGemma2Config(
        integer(metadata, "embedding_length"),
        integer(metadata, "embedding_length_out"),
        layers,
        integer(metadata, "attention.head_count"),
        heads,
        full,
        sliding,
        integer(metadata, "feed_forward_length"),
        integer(metadata, "embedding_length_per_layer_input"),
        metadata.getArraySize("tokenizer.ggml.tokens").orElseThrow(),
        integer(metadata, "context_length"),
        integer(metadata, "attention.sliding_window"),
        metadata.getBoolArray(PREFIX + "attention.sliding_window_pattern").orElseThrow(),
        floating(metadata, "rope.freq_base"),
        floating(metadata, "rope.freq_base_swa"),
        floating(metadata, "attention.layer_norm_rms_epsilon"));
  }

  public int headDim(int layer) {
    return slidingLayers.get(layer) ? slidingHeadDim : fullHeadDim;
  }

  private static int integer(GgufMetadata metadata, String key) {
    return metadata
        .getUint32(PREFIX + key)
        .orElseThrow(() -> new IllegalArgumentException("Missing " + PREFIX + key));
  }

  private static float floating(GgufMetadata metadata, String key) {
    return metadata
        .getFloat32(PREFIX + key)
        .orElseThrow(() -> new IllegalArgumentException("Missing " + PREFIX + key));
  }
}
