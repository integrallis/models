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
    Pooling pooling,
    Experts experts) {

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

  /**
   * The routed feed-forward, or absent on a dense {@code lfm2}.
   *
   * <p>{@code lfm2moe} keeps every structural property of {@code lfm2} -- the same short-conv
   * blocks, the same per-layer attention split, the same tied head -- and replaces the feed-forward
   * on all but the leading blocks. So this is an addition to the dense config rather than a second
   * config, and {@code null} is the dense file.
   *
   * @param count {@code expert_count}
   * @param usedCount {@code expert_used_count}, how many a token routes to
   * @param hiddenDim {@code expert_feed_forward_length}, which is NOT {@code feed_forward_length}
   * @param leadingDenseBlocks {@code leading_dense_block_count}; blocks below this index keep the
   *     dense feed-forward and blocks at or above it are routed
   */
  public record Experts(int count, int usedCount, int hiddenDim, int leadingDenseBlocks) {
    public Experts {
      positive("expert count", count);
      positive("experts used", usedCount);
      positive("expert hidden dim", hiddenDim);
      if (usedCount > count) {
        throw new IllegalArgumentException(
            "expert_used_count must not exceed expert_count: " + usedCount + " > " + count);
      }
      if (leadingDenseBlocks < 0) {
        throw new IllegalArgumentException(
            "leading_dense_block_count must not be negative, was " + leadingDenseBlocks);
      }
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
    // A mixture whose leading dense count covers every block is a dense model wearing the MoE
    // architecture string: the expert tensors would load and never be reached, and the only
    // symptom would be a model that is slower than it should be and right anyway.
    if (experts != null && experts.leadingDenseBlocks() >= numLayers) {
      throw new IllegalArgumentException(
          "leading_dense_block_count leaves no routed block: "
              + experts.leadingDenseBlocks()
              + " >= "
              + numLayers);
    }
  }

  /**
   * Parses the text-only LFM2 GGUF contract, dense ({@code lfm2}) or routed ({@code lfm2moe}).
   *
   * <p>The two share every structural key, so the architecture string selects the metadata prefix
   * and nothing else. {@code lfm2moe}'s addition is the expert block, and upstream treats them the
   * same way: {@code llama_model_lfm2moe::load_arch_hparams} reads the same {@code
   * shortconv.l_cache} and derives the same per-layer split from {@code n_head_kv(il) == 0} as the
   * dense loader does.
   */
  public static Lfm2Config fromMetadata(GgufMetadata metadata) {
    Objects.requireNonNull(metadata, "metadata");
    String architecture = metadata.getString("general.architecture").orElse("");
    if (!"lfm2".equals(architecture) && !"lfm2moe".equals(architecture)) {
      throw new IllegalArgumentException(
          "Expected general.architecture=lfm2 or lfm2moe, found " + architecture);
    }
    String prefix = architecture + ".";
    int numLayers = requiredInt(metadata, prefix + "block_count");
    int embeddingDim = requiredInt(metadata, prefix + "embedding_length");
    int numHeads = requiredInt(metadata, prefix + "attention.head_count");
    List<Integer> kvHeads =
        metadata
            .getInt32Array(prefix + "attention.head_count_kv")
            .orElseThrow(
                () ->
                    new IllegalArgumentException(
                        "LFM2 publishes attention.head_count_kv per layer; a scalar cannot say"
                            + " which layers are convolutional"));
    Experts experts = experts(metadata, prefix, architecture);
    return new Lfm2Config(
        embeddingDim,
        numLayers,
        numHeads,
        kvHeads,
        metadata.getUint32(prefix + "attention.key_length").orElse(embeddingDim / numHeads),
        metadata
            .getUint32(prefix + "vocab_size")
            .or(() -> metadata.getArraySize("tokenizer.ggml.tokens"))
            .orElseThrow(() -> new IllegalArgumentException("Missing LFM2 vocabulary size")),
        requiredInt(metadata, prefix + "context_length"),
        requiredInt(metadata, prefix + "feed_forward_length"),
        requiredFloat(metadata, prefix + "rope.freq_base"),
        requiredFloat(metadata, prefix + "attention.layer_norm_rms_epsilon"),
        requiredInt(metadata, prefix + "shortconv.l_cache"),
        // Optional, and true when absent: the same default llama.cpp carries in llama_hparams
        // (causal_attn = true, llama-hparams.h:182) and reads generically for every architecture
        // (llama-model.cpp:1069), which is why LFM2.5-Embedding's attention.causal=false applies
        // to lfm2 without any lfm2-specific handling on the reference side.
        metadata.getBool(prefix + "attention.causal").orElse(true),
        Pooling.fromCode(metadata.getUint32(prefix + "pooling_type").orElse(Pooling.NONE.code())),
        experts);
  }

  /**
   * Reads the expert block, and refuses a gating function this backend does not implement.
   *
   * <p>The gating check is the point of this method. llama.cpp numbers the choices NONE 0, SOFTMAX
   * 1, SIGMOID 2, SOFTMAX_WEIGHT 3, SQRT_SOFTPLUS 4, and {@code lfm2moe} declares <b>2</b>. A model
   * routed with the wrong gate loads cleanly and produces plausible text, which is the failure this
   * refusal exists to prevent -- #283 says it in as many words: do not work around it with a
   * softmax approximation, because a gating function that is close but not the declared one is the
   * kind of silent divergence the port queue exists to avoid.
   *
   * <p>Unlike the Llama path, the key is <b>required</b> here rather than checked only when
   * present. No {@code lfm2moe} artifact predates it, so there is no shipped file to keep serving.
   */
  private static Experts experts(GgufMetadata metadata, String prefix, String architecture) {
    if (!"lfm2moe".equals(architecture)) {
      return null;
    }
    int gatingFunction = requiredInt(metadata, prefix + "expert_gating_func");
    if (gatingFunction != SIGMOID_GATING) {
      throw new IllegalArgumentException(
          "lfm2moe routes experts with sigmoid gating (expert_gating_func="
              + SIGMOID_GATING
              + "), but this file declares "
              + gatingFunction
              + " (1 is softmax, 3 softmax-over-weights, 4 sqrt-softplus)");
    }
    return new Experts(
        requiredInt(metadata, prefix + "expert_count"),
        requiredInt(metadata, prefix + "expert_used_count"),
        requiredInt(metadata, prefix + "expert_feed_forward_length"),
        // Optional upstream (llama_model_lfm2moe reads it with required=false) and zero when
        // absent, which means every block is routed.
        metadata.getUint32(prefix + "leading_dense_block_count").orElse(0));
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

  /** llama.cpp's {@code LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID}. */
  private static final int SIGMOID_GATING = 2;

  /** Whether this layer's feed-forward is routed rather than dense. */
  public boolean usesExperts(int layer) {
    requireLayer(layer);
    return experts != null && layer >= experts.leadingDenseBlocks();
  }

  /** Whether this file declares a routed feed-forward at all. */
  public boolean isMixtureOfExperts() {
    return experts != null;
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
