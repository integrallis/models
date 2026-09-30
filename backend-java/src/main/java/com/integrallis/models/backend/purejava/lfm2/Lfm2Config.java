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
package com.integrallis.models.backend.purejava.lfm2;

import com.integrallis.models.backend.purejava.gguf.GgufMetadata;
import java.util.List;
import java.util.Objects;

/**
 * The LFM2 decoder contract, read from GGUF metadata.
 *
 * <p>LFM2 is a <b>hybrid</b>: most layers replace attention with a gated short convolution, and
 * which layers do what is published per layer in {@code attention.head_count_kv} -- a zero there
 * means that layer has no attention at all. LFM2.5-1.2B is six attention layers of sixteen, at
 * indices 2, 5, 8, 10, 12 and 14, read from the file rather than assumed to follow a pattern.
 *
 * <p>Every layer, attention or convolution, carries the same gated feed-forward and the same two
 * norms, so only the middle of the block differs.
 */
public record Lfm2Config(
    int embeddingDim,
    int numLayers,
    int numHeads,
    List<Integer> kvHeadsByLayer,
    int headDim,
    int vocabSize,
    int contextLength,
    int hiddenDim,
    float ropeTheta,
    float rmsNormEpsilon,
    int shortConvCache) {

  public Lfm2Config {
    positive("embeddingDim", embeddingDim);
    positive("numLayers", numLayers);
    positive("numHeads", numHeads);
    positive("headDim", headDim);
    positive("vocabSize", vocabSize);
    positive("contextLength", contextLength);
    positive("hiddenDim", hiddenDim);
    finitePositive("ropeTheta", ropeTheta);
    finitePositive("rmsNormEpsilon", rmsNormEpsilon);
    if (shortConvCache < 2) {
      throw new IllegalArgumentException(
          "shortconv.l_cache must be at least 2, was " + shortConvCache);
    }
    Objects.requireNonNull(kvHeadsByLayer, "kvHeadsByLayer");
    if (kvHeadsByLayer.size() != numLayers) {
      throw new IllegalArgumentException(
          "attention.head_count_kv must carry one entry per layer: expected "
              + numLayers
              + ", got "
              + kvHeadsByLayer.size());
    }
    kvHeadsByLayer = List.copyOf(kvHeadsByLayer);
    boolean anyAttention = false;
    for (int layer = 0; layer < numLayers; layer++) {
      int kvHeads = kvHeadsByLayer.get(layer);
      if (kvHeads < 0) {
        throw new IllegalArgumentException("kv head count must not be negative at layer " + layer);
      }
      if (kvHeads > 0) {
        anyAttention = true;
        if (numHeads % kvHeads != 0) {
          throw new IllegalArgumentException(
              "numHeads must be divisible by the kv head count at layer " + layer);
        }
      }
    }
    // A model with no attention at all would be a different architecture, and silently accepting
    // one
    // would mean never allocating a key-value cache and never noticing.
    if (!anyAttention) {
      throw new IllegalArgumentException("LFM2 must carry at least one attention layer");
    }
  }

  /** Parses the text-only LFM2 GGUF contract. */
  public static Lfm2Config fromMetadata(GgufMetadata metadata) {
    Objects.requireNonNull(metadata, "metadata");
    String architecture = metadata.getString("general.architecture").orElse("");
    if (!"lfm2".equals(architecture)) {
      throw new IllegalArgumentException(
          "Expected general.architecture=lfm2, found " + architecture);
    }
    int numLayers = requiredInt(metadata, "lfm2.block_count");
    int embeddingDim = requiredInt(metadata, "lfm2.embedding_length");
    int numHeads = requiredInt(metadata, "lfm2.attention.head_count");
    List<Integer> kvHeads =
        metadata
            .getInt32Array("lfm2.attention.head_count_kv")
            .orElseThrow(
                () ->
                    new IllegalArgumentException(
                        "LFM2 publishes attention.head_count_kv per layer; a scalar cannot say"
                            + " which layers are convolutional"));
    return new Lfm2Config(
        embeddingDim,
        numLayers,
        numHeads,
        kvHeads,
        metadata.getUint32("lfm2.attention.key_length").orElse(embeddingDim / numHeads),
        metadata
            .getUint32("lfm2.vocab_size")
            .or(() -> metadata.getArraySize("tokenizer.ggml.tokens"))
            .orElseThrow(() -> new IllegalArgumentException("Missing LFM2 vocabulary size")),
        requiredInt(metadata, "lfm2.context_length"),
        requiredInt(metadata, "lfm2.feed_forward_length"),
        requiredFloat(metadata, "lfm2.rope.freq_base"),
        requiredFloat(metadata, "lfm2.attention.layer_norm_rms_epsilon"),
        requiredInt(metadata, "lfm2.shortconv.l_cache"));
  }

  /** Whether this layer attends; the alternative is a short convolution. */
  public boolean usesAttention(int layer) {
    requireLayer(layer);
    return kvHeadsByLayer.get(layer) > 0;
  }

  public int numKvHeads(int layer) {
    requireLayer(layer);
    int kvHeads = kvHeadsByLayer.get(layer);
    if (kvHeads == 0) {
      throw new IllegalArgumentException(
          "layer " + layer + " is convolutional and has no kv heads");
    }
    return kvHeads;
  }

  public List<Integer> kvHeadsByLayer() {
    return List.copyOf(kvHeadsByLayer);
  }

  /** How many layers attend, which is how many key-value caches are needed. */
  public int attentionLayers() {
    int count = 0;
    for (int layer = 0; layer < numLayers; layer++) {
      if (usesAttention(layer)) {
        count++;
      }
    }
    return count;
  }

  /** Recurrent state width per convolutional layer: {@code (l_cache - 1) * embeddingDim}. */
  public int shortConvStateSize() {
    return (shortConvCache - 1) * embeddingDim;
  }

  public int queryDim() {
    return numHeads * headDim;
  }

  public int keyDim(int layer) {
    return numKvHeads(layer) * headDim;
  }

  private void requireLayer(int layer) {
    if (layer < 0 || layer >= numLayers) {
      throw new IllegalArgumentException("layer out of range: " + layer);
    }
  }

  private static void positive(String name, int value) {
    if (value <= 0) {
      throw new IllegalArgumentException(name + " must be > 0, was " + value);
    }
  }

  private static void finitePositive(String name, float value) {
    if (!(value > 0.0f) || !Float.isFinite(value)) {
      throw new IllegalArgumentException(name + " must be finite and > 0, was " + value);
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
}
