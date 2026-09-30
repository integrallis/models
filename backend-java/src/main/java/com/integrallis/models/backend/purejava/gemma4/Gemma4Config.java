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
    List<Integer> sharedHiddenDimByLayer,
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
    float finalLogitSoftcap,
    int sharedKvLayers,
    int perLayerEmbeddingDim) {

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
    // The E-series varies its dense feed-forward width per layer -- MatFormer -- so this is a
    // list, scalar-broadcast for the variants that publish one value. Reading element 0 and calling
    // it the model's width loaded Gemma 4 E2B as far as blk.15, whose ffn_gate is 12288 wide where
    // blk.0 is 6144, and then refused the file on a shape mismatch.
    sharedHiddenDimByLayer =
        immutableSized("sharedHiddenDimByLayer", sharedHiddenDimByLayer, numLayers);
    for (int layer = 0; layer < numLayers; layer++) {
      positive("sharedHiddenDimByLayer[" + layer + "]", sharedHiddenDimByLayer.get(layer));
    }
    // Zero across all three is the dense variant; positive across all three is mixture-of-experts.
    // A mix of the two is a malformed model rather than a shape we should try to serve, so it is
    // rejected here instead of failing later in the expert tensor layout.
    boolean dense = expertHiddenDim == 0 && numExperts == 0 && numExpertsUsed == 0;
    if (!dense) {
      positive("expertHiddenDim", expertHiddenDim);
      positive("numExperts", numExperts);
      positive("numExpertsUsed", numExpertsUsed);
    }
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

  /**
   * Compatibility constructor for the dense and mixture-of-experts variants.
   *
   * <p>Neither shares a key-value cache nor carries per-layer input embeddings, so both new values
   * are zero: {@code sharedKvLayers=0} means every layer owns its cache, and {@code
   * perLayerEmbeddingDim=0} means there is no second embedding table. Only the E-series sets them,
   * and it is built through {@link #fromMetadata}.
   */
  public Gemma4Config(
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
      float finalLogitSoftcap,
      int sharedKvLayers,
      int perLayerEmbeddingDim) {
    this(
        embeddingDim,
        numLayers,
        numHeads,
        kvHeadsByLayer,
        fullKeyLength,
        slidingKeyLength,
        fullValueLength,
        slidingValueLength,
        vocabSize,
        contextLength,
        java.util.Collections.nCopies(numLayers, sharedHiddenDim),
        expertHiddenDim,
        numExperts,
        numExpertsUsed,
        fullRopeTheta,
        slidingRopeTheta,
        fullRopeDimension,
        slidingRopeDimension,
        rmsNormEps,
        slidingWindow,
        slidingWindowByLayer,
        finalLogitSoftcap,
        sharedKvLayers,
        perLayerEmbeddingDim);
  }

  /** A model whose dense feed-forward is the same width on every layer. */
  public Gemma4Config(
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
    this(
        embeddingDim,
        numLayers,
        numHeads,
        kvHeadsByLayer,
        fullKeyLength,
        slidingKeyLength,
        fullValueLength,
        slidingValueLength,
        vocabSize,
        contextLength,
        java.util.Collections.nCopies(numLayers, sharedHiddenDim),
        expertHiddenDim,
        numExperts,
        numExpertsUsed,
        fullRopeTheta,
        slidingRopeTheta,
        fullRopeDimension,
        slidingRopeDimension,
        rmsNormEps,
        slidingWindow,
        slidingWindowByLayer,
        finalLogitSoftcap,
        0,
        0);
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
    List<Integer> kvHeads = perLayerInts(metadata, "gemma4.attention.head_count_kv", numLayers);
    List<Boolean> slidingPattern =
        metadata
            .getBoolArray("gemma4.attention.sliding_window_pattern")
            .orElseThrow(
                () ->
                    new IllegalArgumentException(
                        "Missing gemma4.attention.sliding_window_pattern"));
    requireLayerCount("gemma4.attention.sliding_window_pattern", slidingPattern, numLayers);

    int sharedKvLayers = metadata.getUint32("gemma4.attention.shared_kv_layers").orElse(0);
    int perLayerEmbeddingLength =
        metadata.getUint32("gemma4.embedding_length_per_layer_input").orElse(0);

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
        // The E-series publishes this per layer; the dense and routed variants publish one value,
        // which perLayerInts broadcasts.
        perLayerInts(metadata, "gemma4.feed_forward_length", numLayers),
        // Absent means DENSE, not malformed. Gemma 4 ships in two shapes: a mixture-of-experts
        // variant that publishes these three keys, and a dense variant (12B, 31B) that omits them
        // because it has no experts at all -- verified from the GGUF, which carries zero tensors
        // matching *_exps and a plain blk.N.ffn_{gate,up,down}. Requiring them rejected every dense
        // Gemma 4 with "Missing gemma4.expert_feed_forward_length".
        metadata.getUint32("gemma4.expert_feed_forward_length").orElse(0),
        metadata.getUint32("gemma4.expert_count").orElse(0),
        metadata.getUint32("gemma4.expert_used_count").orElse(0),
        requiredFloat(metadata, "gemma4.rope.freq_base"),
        requiredFloat(metadata, "gemma4.rope.freq_base_swa"),
        metadata.getUint32("gemma4.rope.dimension_count").orElse(fullKeyLength),
        metadata.getUint32("gemma4.rope.dimension_count_swa").orElse(slidingKeyLength),
        requiredFloat(metadata, "gemma4.attention.layer_norm_rms_epsilon"),
        requiredInt(metadata, "gemma4.attention.sliding_window"),
        slidingPattern,
        requiredFloat(metadata, "gemma4.final_logit_softcapping"),
        sharedKvLayers,
        perLayerEmbeddingLength);
  }

  /**
   * Whether this is a dense Gemma 4 rather than a mixture-of-experts one.
   *
   * <p>Dense models (12B, 31B) publish no {@code expert_*} metadata and carry no {@code *_exps}
   * tensors; their feed-forward network is the plain gated one at {@code blk.N.ffn_{gate,up,down}}.
   * The routed half of the layer is skipped entirely for them.
   */
  public boolean isDense() {
    return numExperts == 0;
  }

  /** Whether this variant feeds a second, per-layer embedding into every layer (the E-series). */
  public boolean usesPerLayerEmbeddings() {
    return perLayerEmbeddingDim > 0;
  }

  /**
   * How many leading layers keep their own key and value cache.
   *
   * <p>Mirrors llama.cpp's {@code n_layer_kv_from_start = n_layer - shared_kv_layers}. Gemma 4 E2B
   * declares {@code shared_kv_layers=20} over 35 layers, so the first 15 own a cache and the rest
   * read one belonging to an earlier layer.
   */
  public int kvOwningLayers() {
    return sharedKvLayers == 0 ? numLayers : numLayers - sharedKvLayers;
  }

  /** Whether this layer computes and stores its own key and value projections. */
  public boolean ownsKvCache(int layer) {
    requireLayer(layer);
    return layer < kvOwningLayers();
  }

  /**
   * The layer whose key and value cache this layer attends to.
   *
   * <p>Transcribed from llama.cpp's layer-reuse callback: a sharing layer reads {@code
   * kvOwningLayers() - (usesSlidingWindow(layer) ? 2 : 1)}. For Gemma 4 E2B that is layer 13 for a
   * sliding layer and layer 14 for a full one -- the last owning layer of each attention type,
   * which is why the head dimensions line up: 13 is sliding at 256 and 14 is full at 512. A sharing
   * layer whose source had a different head dimension would be a shape error, so this is asserted
   * rather than assumed.
   *
   * @param layer the zero-based layer index
   * @return that layer itself when it owns a cache, otherwise the layer it reads
   */
  public int kvSourceLayer(int layer) {
    requireLayer(layer);
    if (ownsKvCache(layer)) {
      return layer;
    }
    int source = kvOwningLayers() - (usesSlidingWindow(layer) ? 2 : 1);
    if (source < 0 || source >= kvOwningLayers()) {
      throw new IllegalArgumentException(
          "layer "
              + layer
              + " shares a key-value cache with layer "
              + source
              + ", which does not own one");
    }
    if (headDim(source) != headDim(layer) || numKvHeads(source) != numKvHeads(layer)) {
      throw new IllegalArgumentException(
          "layer "
              + layer
              + " shares a cache with layer "
              + source
              + " but their attention shapes differ: headDim "
              + headDim(layer)
              + " vs "
              + headDim(source)
              + ", kvHeads "
              + numKvHeads(layer)
              + " vs "
              + numKvHeads(source));
    }
    return source;
  }

  public boolean usesSlidingWindow(int layer) {
    requireLayer(layer);
    return slidingWindowByLayer.get(layer);
  }

  public List<Integer> kvHeadsByLayer() {
    return List.copyOf(kvHeadsByLayer);
  }

  public List<Integer> sharedHiddenDimByLayer() {
    return List.copyOf(sharedHiddenDimByLayer);
  }

  public List<Boolean> slidingWindowByLayer() {
    return List.copyOf(slidingWindowByLayer);
  }

  public int numKvHeads(int layer) {
    requireLayer(layer);
    return kvHeadsByLayer.get(layer);
  }

  /** The dense feed-forward width of one layer. */
  public int sharedHiddenDim(int layer) {
    requireLayer(layer);
    return sharedHiddenDimByLayer.get(layer);
  }

  /**
   * The widest dense feed-forward across the model, for buffers that have to serve every layer.
   *
   * <p>Separate from {@link #sharedHiddenDim(int)} on purpose: a buffer sized from one layer's
   * width overflows on a MatFormer model, and a tensor shape checked against the maximum silently
   * accepts the wrong tensor.
   */
  public int maxSharedHiddenDim() {
    int widest = 0;
    for (int hidden : sharedHiddenDimByLayer) {
      widest = Math.max(widest, hidden);
    }
    return widest;
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

  /**
   * A per-layer integer list, accepting a scalar and broadcasting it over every layer.
   *
   * <p>Gemma 4 publishes some of these keys both ways. The dense and mixture-of-experts variants
   * write {@code attention.head_count_kv} as an array with one entry per block; the E-series writes
   * it as a single {@code uint32}, because every layer shares the value. Demanding the array form
   * rejected the E-series before any decoder ran, and broadcasting is what the reference does.
   *
   * <p>A scalar is only broadcast when the array form is absent, so a file that publishes the array
   * still gets its exact per-layer values.
   */
  private static List<Integer> perLayerInts(GgufMetadata metadata, String key, int numLayers) {
    List<Integer> array = metadata.getInt32Array(key).orElse(null);
    if (array != null) {
      requireLayerCount(key, array, numLayers);
      return List.copyOf(array);
    }
    int scalar =
        metadata
            .getUint32(key)
            .orElseThrow(
                () ->
                    new IllegalArgumentException("Missing " + key + " (neither array nor scalar)"));
    return java.util.Collections.nCopies(numLayers, scalar);
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
