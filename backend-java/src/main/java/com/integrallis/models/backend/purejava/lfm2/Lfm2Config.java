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
    int shortConvCache,
    boolean causalAttention,
    Pooling pooling) {

  /**
   * Sequence reduction declared by {@code lfm2.pooling_type}, using GGUF's own codes.
   *
   * <p>{@code NONE} is code zero and is what a generative LFM2 carries: it reduces nothing and
   * emits one vector per token. The others match {@code llama_pooling_type} in llama.cpp.
   */
  public enum Pooling {
    NONE(0),
    MEAN(1),
    CLS(2),
    LAST(3),
    RANK(4);

    private final int code;

    Pooling(int code) {
      this.code = code;
    }

    public int code() {
      return code;
    }

    static Pooling fromCode(int code) {
      for (Pooling value : values()) {
        if (value.code == code) {
          return value;
        }
      }
      throw new IllegalArgumentException("unsupported LFM2 pooling type: " + code);
    }
  }

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
    Objects.requireNonNull(pooling, "pooling");
    // A bidirectional LFM2 is an embedding model -- that is the only thing dropping the causal
    // mask is for -- so one that declares no reduction is a file we cannot serve either way: not
    // generatively, because its attention sees the future, and not as an embedder, because it has
    // not said how to reduce the sequence. Refusing at load beats emitting position zero and
    // calling it the embedding.
    if (!causalAttention && pooling == Pooling.NONE) {
      throw new IllegalArgumentException(
          "lfm2.attention.causal is false but lfm2.pooling_type declares no pooling");
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
        requiredInt(metadata, "lfm2.shortconv.l_cache"),
        // Optional, and true when absent: the same default llama.cpp carries in llama_hparams
        // (causal_attn = true, llama-hparams.h:182) and reads generically for every architecture
        // (llama-model.cpp:1069), which is why LFM2.5-Embedding's attention.causal=false applies
        // to lfm2 without any lfm2-specific handling on the reference side.
        metadata.getBool("lfm2.attention.causal").orElse(true),
        Pooling.fromCode(metadata.getUint32("lfm2.pooling_type").orElse(Pooling.NONE.code())));
  }

  /**
   * Whether this file is a whole-sequence encoder rather than a decoder.
   *
   * <p>Dropping the causal mask changes two operators, not one: attention spans the whole sequence,
   * and the short convolution becomes a centred window with symmetric zero padding instead of a
   * shift register over the past. Both are read from llama.cpp's {@code build_shortconv_block} and
   * {@code build_attn}.
   */
  public boolean encodesWholeSequence() {
    return !causalAttention;
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
