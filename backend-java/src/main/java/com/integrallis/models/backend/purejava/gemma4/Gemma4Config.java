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
package com.integrallis.models.backend.purejava.gemma4;

import com.integrallis.models.backend.purejava.gguf.GgufMetadata;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** Immutable execution shape for the text-only Gemma 4 mixture-of-experts decoder. */
public record Gemma4Config(
    int embeddingDim,
    int numLayers,
    int numHeads,
    List<Integer> kvHeadsByLayer,
    int fullKeyLength,
    int slidingKeyLength,
    int fullValueLength,
    int slidingValueLength,
    int vocabSize,
    int contextLength,
    int sharedHiddenDim,
    int expertHiddenDim,
    int numExperts,
    int numExpertsUsed,
    float fullRopeTheta,
    float slidingRopeTheta,
    int fullRopeDimension,
    int slidingRopeDimension,
    float rmsNormEps,
    int slidingWindow,
    List<Boolean> slidingWindowByLayer,
    float finalLogitSoftcap) {

  public Gemma4Config {
    positive("embeddingDim", embeddingDim);
    positive("numLayers", numLayers);
    positive("numHeads", numHeads);
    kvHeadsByLayer = immutableSized("kvHeadsByLayer", kvHeadsByLayer, numLayers);
    slidingWindowByLayer = immutableSized("slidingWindowByLayer", slidingWindowByLayer, numLayers);
    positive("fullKeyLength", fullKeyLength);
    positive("slidingKeyLength", slidingKeyLength);
    positive("fullValueLength", fullValueLength);
    positive("slidingValueLength", slidingValueLength);
    positive("vocabSize", vocabSize);
    positive("contextLength", contextLength);
    positive("sharedHiddenDim", sharedHiddenDim);
    positive("expertHiddenDim", expertHiddenDim);
    positive("numExperts", numExperts);
    positive("numExpertsUsed", numExpertsUsed);
    positive("fullRopeDimension", fullRopeDimension);
    positive("slidingRopeDimension", slidingRopeDimension);
    positive("slidingWindow", slidingWindow);
    finitePositive("fullRopeTheta", fullRopeTheta);
    finitePositive("slidingRopeTheta", slidingRopeTheta);
    finitePositive("rmsNormEps", rmsNormEps);
    finitePositive("finalLogitSoftcap", finalLogitSoftcap);

    if (numExpertsUsed > numExperts) {
      throw new IllegalArgumentException("numExpertsUsed must not exceed numExperts");
    }
    if (fullKeyLength != fullValueLength || slidingKeyLength != slidingValueLength) {
      throw new IllegalArgumentException("Gemma 4 key and value head lengths must match");
    }
    if (fullRopeDimension > fullKeyLength || slidingRopeDimension > slidingKeyLength) {
      throw new IllegalArgumentException("RoPE dimensions must fit their attention head lengths");
    }
    if ((fullRopeDimension & 1) != 0 || (slidingRopeDimension & 1) != 0) {
      throw new IllegalArgumentException("RoPE dimensions must be even");
    }
    for (int layer = 0; layer < numLayers; layer++) {
      int kvHeads = kvHeadsByLayer.get(layer);
      positive("kvHeadsByLayer[" + layer + "]", kvHeads);
      if (numHeads % kvHeads != 0) {
        throw new IllegalArgumentException(
            "numHeads must be divisible by kvHeadsByLayer[" + layer + "]=" + kvHeads);
      }
    }
  }

  /** Parses the supported text-only Gemma 4 GGUF variant. */
  public static Gemma4Config fromMetadata(GgufMetadata metadata) {
    Objects.requireNonNull(metadata, "metadata");
    String architecture = metadata.getString("general.architecture").orElse("");
    if (!"gemma4".equals(architecture)) {
      throw new IllegalArgumentException(
          "Expected general.architecture=gemma4, found " + architecture);
    }

    int numLayers = requiredInt(metadata, "gemma4.block_count");
    List<Integer> kvHeads = requiredIntArray(metadata, "gemma4.attention.head_count_kv");
    requireLayerCount("gemma4.attention.head_count_kv", kvHeads, numLayers);
    List<Boolean> slidingPattern =
        metadata
            .getBoolArray("gemma4.attention.sliding_window_pattern")
            .orElseThrow(
                () ->
                    new IllegalArgumentException(
                        "Missing gemma4.attention.sliding_window_pattern"));
    requireLayerCount("gemma4.attention.sliding_window_pattern", slidingPattern, numLayers);

    int sharedKvLayers = metadata.getUint32("gemma4.attention.shared_kv_layers").orElse(0);
    if (sharedKvLayers != 0) {
      throw new IllegalArgumentException(
          "Unsupported Gemma 4 variant: shared_kv_layers=" + sharedKvLayers);
    }
    int perLayerEmbeddingLength =
        metadata.getUint32("gemma4.embedding_length_per_layer_input").orElse(0);
    if (perLayerEmbeddingLength != 0) {
      throw new IllegalArgumentException(
          "Unsupported Gemma 4 variant: embedding_length_per_layer_input="
              + perLayerEmbeddingLength);
    }

    int fullKeyLength = requiredInt(metadata, "gemma4.attention.key_length");
    int slidingKeyLength = requiredInt(metadata, "gemma4.attention.key_length_swa");

    return new Gemma4Config(
        requiredInt(metadata, "gemma4.embedding_length"),
        numLayers,
        requiredInt(metadata, "gemma4.attention.head_count"),
        kvHeads,
        fullKeyLength,
        slidingKeyLength,
        requiredInt(metadata, "gemma4.attention.value_length"),
        requiredInt(metadata, "gemma4.attention.value_length_swa"),
        metadata
            .getUint32("gemma4.vocab_size")
            .or(() -> metadata.getArraySize("tokenizer.ggml.tokens"))
            .orElseThrow(() -> new IllegalArgumentException("Missing Gemma 4 vocabulary size")),
        requiredInt(metadata, "gemma4.context_length"),
        requiredInt(metadata, "gemma4.feed_forward_length"),
        requiredInt(metadata, "gemma4.expert_feed_forward_length"),
        requiredInt(metadata, "gemma4.expert_count"),
        requiredInt(metadata, "gemma4.expert_used_count"),
        requiredFloat(metadata, "gemma4.rope.freq_base"),
        requiredFloat(metadata, "gemma4.rope.freq_base_swa"),
        metadata.getUint32("gemma4.rope.dimension_count").orElse(fullKeyLength),
        metadata.getUint32("gemma4.rope.dimension_count_swa").orElse(slidingKeyLength),
        requiredFloat(metadata, "gemma4.attention.layer_norm_rms_epsilon"),
        requiredInt(metadata, "gemma4.attention.sliding_window"),
        slidingPattern,
        requiredFloat(metadata, "gemma4.final_logit_softcapping"));
  }

  public boolean usesSlidingWindow(int layer) {
    requireLayer(layer);
    return slidingWindowByLayer.get(layer);
  }

  public List<Integer> kvHeadsByLayer() {
    return List.copyOf(kvHeadsByLayer);
  }

  public List<Boolean> slidingWindowByLayer() {
    return List.copyOf(slidingWindowByLayer);
  }

  public int numKvHeads(int layer) {
    requireLayer(layer);
    return kvHeadsByLayer.get(layer);
  }

  public int headDim(int layer) {
    return usesSlidingWindow(layer) ? slidingKeyLength : fullKeyLength;
  }

  public int queryDim(int layer) {
    return Math.multiplyExact(numHeads, headDim(layer));
  }

  public int keyDim(int layer) {
    return Math.multiplyExact(numKvHeads(layer), headDim(layer));
  }

  public int valueDim(int layer) {
    int valueLength = usesSlidingWindow(layer) ? slidingValueLength : fullValueLength;
    return Math.multiplyExact(numKvHeads(layer), valueLength);
  }

  public int attentionOutputDim(int layer) {
    int valueLength = usesSlidingWindow(layer) ? slidingValueLength : fullValueLength;
    return Math.multiplyExact(numHeads, valueLength);
  }

  public int ropeDimension(int layer) {
    return usesSlidingWindow(layer) ? slidingRopeDimension : fullRopeDimension;
  }

  public float ropeTheta(int layer) {
    return usesSlidingWindow(layer) ? slidingRopeTheta : fullRopeTheta;
  }

  public float embeddingScale() {
    return (float) Math.sqrt(embeddingDim);
  }

  public float attentionScale() {
    return 1.0f;
  }

  public int attentionStartPosition(int layer, int position) {
    requireLayer(layer);
    if (position < 0) {
      throw new IllegalArgumentException("position must be >= 0");
    }
    return usesSlidingWindow(layer) ? Math.max(0, position - slidingWindow + 1) : 0;
  }

  public List<Integer> fullAttentionLayers() {
    List<Integer> layers = new ArrayList<>();
    for (int layer = 0; layer < numLayers; layer++) {
      if (!slidingWindowByLayer.get(layer)) {
        layers.add(layer);
      }
    }
    return List.copyOf(layers);
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

  private static List<Integer> requiredIntArray(GgufMetadata metadata, String key) {
    return metadata
        .getInt32Array(key)
        .orElseThrow(() -> new IllegalArgumentException("Missing " + key));
  }

  private static void requireLayerCount(String key, List<?> values, int numLayers) {
    if (values.size() != numLayers) {
      throw new IllegalArgumentException(
          key + " must contain " + numLayers + " entries: " + values.size());
    }
  }

  private static <T> List<T> immutableSized(String name, List<T> values, int size) {
    Objects.requireNonNull(values, name);
    if (values.size() != size) {
      throw new IllegalArgumentException(
          name + " must contain " + size + " entries: " + values.size());
    }
    return List.copyOf(values);
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
