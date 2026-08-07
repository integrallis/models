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
package com.integrallis.models.backend.purejava.llama;

import com.integrallis.models.backend.purejava.gguf.GgufMetadata;

/** Configuration for a Llama-family model, extracted from GGUF metadata. */
public record LlamaConfig(
    DecoderArchitecture architecture,
    int embeddingDim,
    int numLayers,
    int numHeads,
    int numKvHeads,
    int keyLength,
    int valueLength,
    int vocabSize,
    int contextLength,
    int hiddenDim,
    float ropeTheta,
    float ropeFrequencyScale,
    float slidingWindowRopeTheta,
    float rmsNormEps,
    int slidingWindow,
    int slidingWindowPattern,
    float finalLogitSoftcap) {

  public LlamaConfig {
    if (architecture == null) throw new IllegalArgumentException("architecture must not be null");
    if (embeddingDim <= 0) throw new IllegalArgumentException("embeddingDim must be > 0");
    if (numLayers <= 0) throw new IllegalArgumentException("numLayers must be > 0");
    if (numHeads <= 0) throw new IllegalArgumentException("numHeads must be > 0");
    if (numKvHeads <= 0) throw new IllegalArgumentException("numKvHeads must be > 0");
    if (numHeads % numKvHeads != 0) {
      throw new IllegalArgumentException("numHeads must be divisible by numKvHeads");
    }
    if (keyLength <= 0) throw new IllegalArgumentException("keyLength must be > 0");
    if (valueLength <= 0) throw new IllegalArgumentException("valueLength must be > 0");
    if (vocabSize <= 0) throw new IllegalArgumentException("vocabSize must be > 0");
    if (!(ropeFrequencyScale > 0.0f) || !Float.isFinite(ropeFrequencyScale)) {
      throw new IllegalArgumentException(
          "ropeFrequencyScale must be finite and > 0: " + ropeFrequencyScale);
    }
    if (!(slidingWindowRopeTheta > 0.0f) || !Float.isFinite(slidingWindowRopeTheta)) {
      throw new IllegalArgumentException(
          "slidingWindowRopeTheta must be finite and > 0: " + slidingWindowRopeTheta);
    }
    if (slidingWindow < 0) {
      throw new IllegalArgumentException("slidingWindow must be >= 0");
    }
    if (slidingWindowPattern <= 0) {
      throw new IllegalArgumentException("slidingWindowPattern must be > 0");
    }
    if (finalLogitSoftcap < 0.0f || !Float.isFinite(finalLogitSoftcap)) {
      throw new IllegalArgumentException(
          "finalLogitSoftcap must be finite and >= 0: " + finalLogitSoftcap);
    }
  }

  /** Query/key dimensions per attention head. */
  public int headDim() {
    return keyLength;
  }

  /** Total query projection dimension. */
  public int queryDim() {
    return keyLength * numHeads;
  }

  /** Total key-cache dimension per layer and position. */
  public int keyDim() {
    return keyLength * numKvHeads;
  }

  /** Total value-cache dimension per layer and position. */
  public int valueDim() {
    return valueLength * numKvHeads;
  }

  /** Concatenated attention value dimension before the output projection. */
  public int attentionOutputDim() {
    return valueLength * numHeads;
  }

  /**
   * Whether this architecture shares the Gemma 3 block layout.
   *
   * <p>EmbeddingGemma is Gemma 3's block repeated 24 times: same scaled embeddings, same GELU-gated
   * feed-forward, same pair of post-norms, same interleaved sliding window. It differs only in how
   * attention is masked and in what happens after the stack, so every structural flag below is
   * shared rather than duplicated per architecture.
   */
  private boolean isGemmaFamily() {
    return architecture == DecoderArchitecture.GEMMA3
        || architecture == DecoderArchitecture.GEMMA_EMBEDDING;
  }

  /**
   * Whether every position attends to the whole sequence rather than only to what precedes it.
   *
   * <p>True for encoders. This is not a tunable: a bidirectional model run causally produces a
   * vector that looks entirely reasonable and is wrong, so the two passes are kept separate and
   * this predicate is what routes between them.
   */
  public boolean usesBidirectionalAttention() {
    return architecture == DecoderArchitecture.GEMMA_EMBEDDING;
  }

  /** Whether the GGUF architecture uses the NeoX split-half rotary layout. */
  public boolean usesNeoxRope() {
    return architecture == DecoderArchitecture.QWEN2
        || architecture == DecoderArchitecture.QWEN3
        || isGemmaFamily();
  }

  /** Whether the zero-based transformer layer applies rotary position embeddings. */
  public boolean usesRope(int layer) {
    if (layer < 0 || layer >= numLayers) {
      throw new IllegalArgumentException("layer out of range: " + layer);
    }
    return architecture != DecoderArchitecture.SMOLLM3 || (layer + 1) % 4 != 0;
  }

  /** Model-specific token embedding multiplier. */
  public float embeddingScale() {
    return isGemmaFamily() ? (float) Math.sqrt(embeddingDim) : 1.0f;
  }

  /** Whether this architecture uses GELU-gated rather than SiLU-gated feed-forward layers. */
  public boolean usesGeluFfn() {
    return isGemmaFamily();
  }

  /** Whether attention output is normalized before its residual addition. */
  public boolean usesPostAttentionNorm() {
    return isGemmaFamily();
  }

  /** Whether feed-forward output is normalized before its residual addition. */
  public boolean usesPostFfnNorm() {
    return isGemmaFamily();
  }

  /** Whether this layer uses bounded sliding-window attention. */
  public boolean usesSlidingWindow(int layer) {
    requireLayer(layer);
    return slidingWindow > 0
        && isGemmaFamily()
        && layer % slidingWindowPattern < slidingWindowPattern - 1;
  }

  /** RoPE base for the selected layer. */
  public float ropeTheta(int layer) {
    return usesSlidingWindow(layer) ? slidingWindowRopeTheta : ropeTheta;
  }

  /** RoPE frequency scale for the selected layer. */
  public float ropeFrequencyScale(int layer) {
    requireLayer(layer);
    return usesSlidingWindow(layer) ? 1.0f : ropeFrequencyScale;
  }

  /**
   * First cache position visible to the selected attention layer.
   *
   * <p>The two window shapes are not the same width. A causal window of {@code n_swa} spans the
   * {@code n_swa} positions ending at the query. A bidirectional one is centred on the query and
   * reaches {@code n_swa / 2} in each direction — llama.cpp's {@code LLAMA_SWA_TYPE_SYMMETRIC},
   * which masks a key exactly when {@code |p1 - p0| > n_swa / 2}. Reading the causal bound as the
   * symmetric one would silently widen the backward reach to twice its size.
   */
  public int attentionStartPosition(int layer, int position) {
    requireLayer(layer);
    if (position < 0) {
      throw new IllegalArgumentException("position must be >= 0");
    }
    if (!usesSlidingWindow(layer)) {
      return 0;
    }
    return usesBidirectionalAttention()
        ? Math.max(0, position - slidingWindow / 2)
        : Math.max(0, position - slidingWindow + 1);
  }

  /**
   * Last position visible to the selected attention layer, bounded by the sequence.
   *
   * <p>Always the query position itself under causal attention: nothing later exists yet.
   *
   * @param layer zero-based transformer layer
   * @param position the querying position
   * @param lastPosition the final position of the sequence being encoded
   * @return the highest key position this query may attend to
   */
  public int attentionEndPosition(int layer, int position, int lastPosition) {
    requireLayer(layer);
    if (position < 0) {
      throw new IllegalArgumentException("position must be >= 0");
    }
    if (lastPosition < position) {
      throw new IllegalArgumentException("lastPosition must be >= position");
    }
    if (!usesBidirectionalAttention()) {
      return position;
    }
    return usesSlidingWindow(layer)
        ? Math.min(lastPosition, position + slidingWindow / 2)
        : lastPosition;
  }

  /** Whether Llama-only staged/pruned layer implementations preserve this architecture. */
  public boolean usesStandardLlamaLayerSemantics() {
    return !isGemmaFamily();
  }

  private void requireLayer(int layer) {
    if (layer < 0 || layer >= numLayers) {
      throw new IllegalArgumentException("layer out of range: " + layer);
    }
  }

  /**
   * Extracts a LlamaConfig from GGUF metadata. Supports Llama-family architectures including models
   * that declare themselves as "qwen2", "qwen3", or "llama" — they all use the same structural
   * layout.
   */
  public static LlamaConfig fromMetadata(GgufMetadata metadata) {
    // Determine the architecture prefix from general.architecture (e.g., "llama", "qwen2",
    // "qwen3")
    String arch = metadata.getString("general.architecture").orElse("llama");
    DecoderArchitecture architecture = DecoderArchitecture.parse(arch);

    int embeddingDim =
        getArchKey(metadata, arch, "embedding_length")
            .orElseThrow(
                () -> new IllegalArgumentException("Missing " + arch + ".embedding_length"));
    int numLayers =
        getArchKey(metadata, arch, "block_count")
            .orElseThrow(() -> new IllegalArgumentException("Missing " + arch + ".block_count"));
    int numHeads =
        getArchKey(metadata, arch, "attention.head_count")
            .orElseThrow(
                () -> new IllegalArgumentException("Missing " + arch + ".attention.head_count"));
    int numKvHeads =
        getArchKey(metadata, arch, "attention.head_count_kv")
            .orElseThrow(
                () -> new IllegalArgumentException("Missing " + arch + ".attention.head_count_kv"));
    int defaultHeadLength = embeddingDim / numHeads;
    int keyLength = getArchKey(metadata, arch, "attention.key_length").orElse(defaultHeadLength);
    int valueLength =
        getArchKey(metadata, arch, "attention.value_length").orElse(defaultHeadLength);
    int vocabSize =
        getArchKey(metadata, arch, "vocab_size")
            .or(() -> metadata.getArraySize("tokenizer.ggml.tokens"))
            .orElse(32000);
    int contextLength = getArchKey(metadata, arch, "context_length").orElse(2048);
    int hiddenDim = getArchKey(metadata, arch, "feed_forward_length").orElse(embeddingDim * 4);
    float ropeTheta = getArchFloatKey(metadata, arch, "rope.freq_base").orElse(10000.0f);
    float ropeFrequencyScale = ropeFrequencyScale(metadata, arch);
    float slidingWindowRopeTheta =
        getArchFloatKey(metadata, arch, "rope.freq_base_swa").orElse(10_000.0f);
    float rmsNormEps =
        getArchFloatKey(metadata, arch, "attention.layer_norm_rms_epsilon").orElse(1e-5f);
    int slidingWindow = getArchKey(metadata, arch, "attention.sliding_window").orElse(0);
    int slidingWindowPattern =
        getArchKey(metadata, arch, "attention.sliding_window_pattern").orElse(6);
    float finalLogitSoftcap =
        getArchFloatKey(metadata, arch, "final_logit_softcapping").orElse(0.0f);

    return new LlamaConfig(
        architecture,
        embeddingDim,
        numLayers,
        numHeads,
        numKvHeads,
        keyLength,
        valueLength,
        vocabSize,
        contextLength,
        hiddenDim,
        ropeTheta,
        ropeFrequencyScale,
        slidingWindowRopeTheta,
        rmsNormEps,
        slidingWindow,
        slidingWindowPattern,
        finalLogitSoftcap);
  }

  private static float ropeFrequencyScale(GgufMetadata metadata, String arch) {
    float factor =
        getArchFloatKey(metadata, arch, "rope.scaling.factor")
            .or(() -> getArchFloatKey(metadata, arch, "rope.scale_linear"))
            .orElse(1.0f);
    if (!(factor > 0.0f) || !Float.isFinite(factor)) {
      throw new IllegalArgumentException("RoPE scaling factor must be finite and > 0: " + factor);
    }
    return 1.0f / factor;
  }

  /**
   * Looks up an integer metadata key with architecture prefix, falling back to "llama." prefix if
   * the arch-specific key is not found.
   */
  private static java.util.Optional<Integer> getArchKey(
      GgufMetadata metadata, String arch, String key) {
    return metadata.getUint32(arch + "." + key).or(() -> metadata.getUint32("llama." + key));
  }

  /**
   * Looks up a float metadata key with architecture prefix, falling back to "llama." prefix if the
   * arch-specific key is not found.
   */
  private static java.util.Optional<Float> getArchFloatKey(
      GgufMetadata metadata, String arch, String key) {
    return metadata.getFloat32(arch + "." + key).or(() -> metadata.getFloat32("llama." + key));
  }
}
