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
package com.integrallis.models.backend.purejava.gemma3n;

import com.integrallis.models.backend.purejava.gguf.GgufMetadata;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/**
 * Immutable execution shape for a Gemma 3n decoder.
 *
 * <p>Structurally a Gemma 4 E-series -- per-layer input embeddings, shared key-value layers, an
 * alternating sliding-window pattern -- with AltUp and LAuReL on top. Separate from {@link
 * com.integrallis.models.backend.purejava.gemma4.Gemma4Config} rather than an extension of it
 * because AltUp changes the shape of the residual stream itself: the state is {@code altupInputs}
 * parallel streams, not one vector, so the graph cannot be shared even though most of the metadata
 * can.
 *
 * @param architecture the {@code general.architecture} this was parsed from
 * @param embeddingDim the model width
 * @param numLayers block count; 30 for E2B, 35 for E4B
 * @param numHeads query heads
 * @param numKvHeads key/value heads
 * @param headDim attention head width
 * @param vocabSize vocabulary size
 * @param contextLength trained context
 * @param hiddenDim feed-forward width
 * @param altupInputs how many parallel residual streams AltUp carries
 * @param altupActiveIndex which stream the layers actually compute
 * @param perLayerEmbeddingDim width of one layer's slice of the second embedding table
 * @param sharedKvLayers how many trailing layers read another layer's key-value cache
 * @param ropeTheta rope base, shared by sliding and full layers on the published files
 * @param rmsNormEpsilon normalisation epsilon
 * @param slidingWindow sliding-window span
 * @param slidingWindowByLayer which layers are sliding, straight from the published pattern
 * @param sparsityScaleByLayer the activation-sparsity multiplier per layer. <b>Non-finite on the
 *     layers that have no sparsity</b> -- see {@link #usesActivationSparsity(int)}
 */
public record Gemma3nConfig(
    String architecture,
    int embeddingDim,
    int numLayers,
    int numHeads,
    int numKvHeads,
    int headDim,
    int vocabSize,
    int contextLength,
    int hiddenDim,
    int altupInputs,
    int altupActiveIndex,
    int perLayerEmbeddingDim,
    int sharedKvLayers,
    float ropeTheta,
    float rmsNormEpsilon,
    int slidingWindow,
    List<Boolean> slidingWindowByLayer,
    List<Float> sparsityScaleByLayer) {

  /**
   * The attention softmax scale.
   *
   * <p><b>1.0, not 1/sqrt(headDim).</b> The reference hard-codes {@code f_attention_scale = 1.0f}
   * for this architecture and publishes no key for it; the query normalisation absorbs the usual
   * scale. Applying 1/sqrt(headDim) as well would narrow every attention distribution.
   */
  public static final float ATTENTION_SCALE = 1.0f;

  public Gemma3nConfig {
    Objects.requireNonNull(architecture, "architecture");
    if (!"gemma3n".equals(architecture)) {
      throw new IllegalArgumentException("architecture must be gemma3n: " + architecture);
    }
    positive("embeddingDim", embeddingDim);
    positive("numLayers", numLayers);
    positive("numHeads", numHeads);
    positive("numKvHeads", numKvHeads);
    positive("headDim", headDim);
    positive("vocabSize", vocabSize);
    positive("contextLength", contextLength);
    positive("hiddenDim", hiddenDim);
    positive("altupInputs", altupInputs);
    positive("perLayerEmbeddingDim", perLayerEmbeddingDim);
    positive("slidingWindow", slidingWindow);
    if (altupActiveIndex < 0 || altupActiveIndex >= altupInputs) {
      throw new IllegalArgumentException(
          "altupActiveIndex must be a stream index below " + altupInputs + ": " + altupActiveIndex);
    }
    if (sharedKvLayers < 0 || sharedKvLayers >= numLayers) {
      throw new IllegalArgumentException(
          "sharedKvLayers must leave at least one owning layer: " + sharedKvLayers);
    }
    // The shared-key-value mapping reads kvOwningLayers - 2 for a sliding layer, so fewer than two
    // owning layers has nowhere to read from. The reference asserts the same bound.
    if (numLayers - sharedKvLayers < 2) {
      throw new IllegalArgumentException(
          "at least two layers must own a key-value cache, found " + (numLayers - sharedKvLayers));
    }
    slidingWindowByLayer = immutableSized("slidingWindowByLayer", slidingWindowByLayer, numLayers);
    sparsityScaleByLayer = immutableSized("sparsityScaleByLayer", sparsityScaleByLayer, numLayers);
  }

  /** Parses the supported text-only Gemma 3n GGUF variant. */
  public static Gemma3nConfig fromMetadata(GgufMetadata metadata) {
    Objects.requireNonNull(metadata, "metadata");
    String architecture = metadata.getString("general.architecture").orElse("");
    if (!"gemma3n".equals(architecture)) {
      throw new IllegalArgumentException(
          "Expected general.architecture=gemma3n, found " + architecture);
    }
    int numLayers = requiredInt(metadata, "gemma3n.block_count");

    // A FLOAT, unlike Gemma 4's integer key. Reading it with getUint32 finds nothing, and the
    // fallback would silently make every layer own a cache -- which loads, and then reads keys and
    // values that were never written.
    float sharedKv =
        metadata
            .getFloat32("gemma3n.attention.shared_kv_layers")
            .orElseThrow(
                () -> new IllegalArgumentException("Missing gemma3n.attention.shared_kv_layers"));
    if (sharedKv < 0 || sharedKv != Math.rint(sharedKv)) {
      throw new IllegalArgumentException(
          "gemma3n.attention.shared_kv_layers must be a non-negative whole number: " + sharedKv);
    }

    List<Boolean> sliding =
        metadata
            .getBoolArray("gemma3n.attention.sliding_window_pattern")
            .orElseThrow(
                () ->
                    new IllegalArgumentException(
                        "Missing gemma3n.attention.sliding_window_pattern"));
    List<Float> sparsity =
        metadata
            .getFloat32Array("gemma3n.activation_sparsity_scale")
            .orElseThrow(
                () -> new IllegalArgumentException("Missing gemma3n.activation_sparsity_scale"));

    return new Gemma3nConfig(
        architecture,
        requiredInt(metadata, "gemma3n.embedding_length"),
        numLayers,
        requiredInt(metadata, "gemma3n.attention.head_count"),
        requiredInt(metadata, "gemma3n.attention.head_count_kv"),
        requiredInt(metadata, "gemma3n.attention.key_length"),
        metadata
            .getUint32("gemma3n.vocab_size")
            .or(() -> metadata.getArraySize("tokenizer.ggml.tokens"))
            .orElseThrow(() -> new IllegalArgumentException("Missing Gemma 3n vocabulary size")),
        requiredInt(metadata, "gemma3n.context_length"),
        requiredInt(metadata, "gemma3n.feed_forward_length"),
        metadata.getUint32("gemma3n.altup.num_inputs").orElse(4),
        metadata.getUint32("gemma3n.altup.active_idx").orElse(0),
        requiredInt(metadata, "gemma3n.embedding_length_per_layer_input"),
        (int) Math.rint(sharedKv),
        requiredFloat(metadata, "gemma3n.rope.freq_base"),
        requiredFloat(metadata, "gemma3n.attention.layer_norm_rms_epsilon"),
        requiredInt(metadata, "gemma3n.attention.sliding_window"),
        sliding,
        sparsity);
  }

  /** Query projection width. */
  public int queryDim() {
    return Math.multiplyExact(numHeads, headDim);
  }

  /** Key and value projection width. */
  public int keyDim() {
    return Math.multiplyExact(numKvHeads, headDim);
  }

  /** Total width of the second embedding table's per-layer slices. */
  public int perLayerTotalDim() {
    return Math.multiplyExact(perLayerEmbeddingDim, numLayers);
  }

  public boolean usesSlidingWindow(int layer) {
    requireLayer(layer);
    return slidingWindowByLayer.get(layer);
  }

  /**
   * How many leading layers own a key-value cache.
   *
   * <p>{@code numLayers - sharedKvLayers}, the reference's {@code n_layer_kv_from_start}. For E2B
   * that is 30 - 10 = 20, which is also the value the reference hard-codes.
   */
  public int kvOwningLayers() {
    return numLayers - sharedKvLayers;
  }

  public boolean ownsKvCache(int layer) {
    requireLayer(layer);
    return layer < kvOwningLayers();
  }

  /**
   * Which layer's key-value cache a sharing layer reads.
   *
   * <p>{@code kvOwningLayers - (sliding ? 2 : 1)}: a sliding layer reads the last sliding owner and
   * a full-attention layer the last full owner, because the two have different cache spans.
   * Identical to the Gemma 4 E-series mapping, and confirmed from the reference's own reuse
   * callback, which handles both architectures with one expression.
   */
  public int kvSourceLayer(int layer) {
    requireLayer(layer);
    if (ownsKvCache(layer)) {
      return layer;
    }
    int source = kvOwningLayers() - (usesSlidingWindow(layer) ? 2 : 1);
    if (source < 0 || source >= kvOwningLayers()) {
      throw new IllegalStateException(
          "layer " + layer + " has no key-value source within " + kvOwningLayers() + " owners");
    }
    if (usesSlidingWindow(source) != usesSlidingWindow(layer)) {
      // A sharing layer must read a cache with the same span, or it reads positions that were never
      // written for it. Asserted rather than assumed: the -2/-1 only lands correctly when the
      // sliding pattern has the period the published files use.
      throw new IllegalStateException(
          "layer " + layer + " would read layer " + source + ", whose sliding-window span differs");
    }
    return source;
  }

  /**
   * Whether this layer applies activation sparsity.
   *
   * <p><b>The published scale is {@code -inf}, not zero, on the layers without sparsity.</b>
   * Feeding that into {@code mean + deviation * scale} gives a cutoff of negative infinity, and
   * subtracting it saturates every activation to {@code +inf}. The reference never meets this
   * because it ignores the array and hard-codes "the first ten layers"; reading the array is the
   * faithful choice, and it has to treat a non-finite entry as "no sparsity here".
   */
  public boolean usesActivationSparsity(int layer) {
    requireLayer(layer);
    float scale = sparsityScaleByLayer.get(layer);
    return Float.isFinite(scale) && scale > 0.0f;
  }

  /** The sparsity multiplier for a layer that has one. */
  public float activationSparsityScale(int layer) {
    if (!usesActivationSparsity(layer)) {
      throw new IllegalArgumentException("layer " + layer + " applies no activation sparsity");
    }
    return sparsityScaleByLayer.get(layer);
  }

  public List<Boolean> slidingWindowByLayer() {
    return List.copyOf(slidingWindowByLayer);
  }

  public List<Float> sparsityScaleByLayer() {
    return List.copyOf(sparsityScaleByLayer);
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

  private static <T> List<T> immutableSized(String name, List<T> values, int size) {
    Objects.requireNonNull(values, name);
    if (values.size() != size) {
      throw new IllegalArgumentException(
          name + " must contain " + size + " entries: " + values.size());
    }
    return List.copyOf(new ArrayList<>(values));
  }
}
