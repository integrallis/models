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
import com.integrallis.models.backend.purejava.ops.RotaryTable;
import java.util.Objects;

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
    float finalLogitSoftcap,
    float attnLogitSoftcap,
    float attnTempScale,
    int ropeDimensions,
    int numExperts,
    int numExpertsUsed,
    int expertHiddenDim,
    RopeScaling ropeScaling,
    GraniteScalars graniteScalars) {

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
    // All three expert fields or none: a half-declared mixture of experts is a malformed model, and
    // treating it as dense would silently skip the routed feed-forward and produce wrong output.
    if (!(numExperts == 0 && numExpertsUsed == 0 && expertHiddenDim == 0)) {
      if (numExperts <= 0) throw new IllegalArgumentException("numExperts must be > 0");
      if (numExpertsUsed <= 0) throw new IllegalArgumentException("numExpertsUsed must be > 0");
      if (expertHiddenDim <= 0) throw new IllegalArgumentException("expertHiddenDim must be > 0");
      if (numExpertsUsed > numExperts) {
        throw new IllegalArgumentException(
            "numExpertsUsed must not exceed numExperts: " + numExpertsUsed + " > " + numExperts);
      }
    }
    if (attnTempScale != 0.0f && !Float.isFinite(attnTempScale)) {
      throw new IllegalArgumentException("attnTempScale must be finite: " + attnTempScale);
    }
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
    ropeScaling = Objects.requireNonNull(ropeScaling, "ropeScaling");
    graniteScalars = Objects.requireNonNull(graniteScalars, "graniteScalars");
    // The structured scaling contract is authoritative. Canonicalizing the legacy scalar also
    // avoids reciprocal float round-trip differences and correctly handles GGUF files that retain
    // an unused factor while explicitly declaring scaling type "none".
    ropeFrequencyScale = ropeScaling.frequencyScale();
  }

  /** Compatibility constructor for ordinary or linearly scaled checkpoints. */
  public LlamaConfig(
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
    this(
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
        finalLogitSoftcap,
        // No attention-logit softcap: only Gemma 2 has one, and it is built through fromMetadata.
        0.0f,
        // No attention temperature scaling: only Mistral 3 has it, built through fromMetadata.
        0.0f,
        // Full rotary: the rotary width equals the head width unless fromMetadata says otherwise.
        keyLength,
        // Dense: mixture-of-experts models are built through fromMetadata.
        0,
        0,
        0,
        RopeScaling.linear(ropeFrequencyScale),
        GraniteScalars.none());
  }

  /** GGUF rotary-scaling algorithms implemented by the pure-Java graph. */
  public enum RopeScalingType {
    LINEAR,
    YARN
  }

  /** Complete long-context scaling contract retained from GGUF metadata. */
  public record RopeScaling(
      RopeScalingType type, float factor, int originalContext, float betaFast, float betaSlow) {

    public RopeScaling {
      type = Objects.requireNonNull(type, "type");
      if (!(factor > 0.0f) || !Float.isFinite(factor)) {
        throw new IllegalArgumentException("RoPE scaling factor must be finite and > 0: " + factor);
      }
      if (type == RopeScalingType.YARN && originalContext <= 0) {
        throw new IllegalArgumentException(
            "YaRN originalContext must be positive: " + originalContext);
      }
      if (!(betaFast > 0.0f) || !Float.isFinite(betaFast)) {
        throw new IllegalArgumentException("YaRN betaFast must be finite and > 0: " + betaFast);
      }
      if (!(betaSlow > 0.0f) || !Float.isFinite(betaSlow)) {
        throw new IllegalArgumentException("YaRN betaSlow must be finite and > 0: " + betaSlow);
      }
    }

    static RopeScaling linear(float frequencyScale) {
      if (!(frequencyScale > 0.0f) || !Float.isFinite(frequencyScale)) {
        throw new IllegalArgumentException(
            "RoPE frequency scale must be finite and > 0: " + frequencyScale);
      }
      return new RopeScaling(RopeScalingType.LINEAR, 1.0f / frequencyScale, 0, 32.0f, 1.0f);
    }

    static RopeScaling yarn(float factor, int originalContext, float betaFast, float betaSlow) {
      return new RopeScaling(RopeScalingType.YARN, factor, originalContext, betaFast, betaSlow);
    }

    float frequencyScale() {
      return 1.0f / factor;
    }
  }

  /** Numerical scales that distinguish Granite's decoder graph from a Llama graph. */
  public record GraniteScalars(float embedding, float attention, float residual, float logits) {

    public GraniteScalars {
      requireFinitePositive("Granite embedding scale", embedding);
      requireFinitePositive("Granite attention scale", attention);
      requireFinitePositive("Granite residual scale", residual);
      requireFinitePositive("Granite logit scale", logits);
    }

    static GraniteScalars none() {
      return new GraniteScalars(1.0f, 1.0f, 1.0f, 1.0f);
    }
  }

  /** Query/key dimensions per attention head. */
  public int headDim() {
    return keyLength;
  }

  /** Whether rotary embedding covers only part of each head. */
  public boolean usesPartialRotary() {
    return ropeDimensions < keyLength;
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
    return architecture == DecoderArchitecture.GEMMA
        || architecture == DecoderArchitecture.GEMMA2
        || architecture == DecoderArchitecture.GEMMA3
        || architecture == DecoderArchitecture.GEMMA_EMBEDDING;
  }

  /**
   * Whether this is Gemma 1, which shares the family's scaling and activation but not its extra
   * norms.
   *
   * <p>Gemma 1 carries exactly the Llama tensor set -- verified against {@code codegemma-7b-it},
   * whose blocks hold only {@code attn_{q,k,v,output,norm}} and {@code ffn_{gate,up,down,norm}}.
   * The post-attention and post-feed-forward norms arrived with Gemma 2, so requiring them here
   * would reject every Gemma 1 model for a tensor it was never built with.
   */
  private boolean isGemma1() {
    return architecture == DecoderArchitecture.GEMMA;
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

  /** Whether the feed-forward network is a mixture of experts rather than a single dense one. */
  public boolean usesMixtureOfExperts() {
    return numExperts > 0;
  }

  /** Whether this model scales query magnitudes by a position-dependent attention temperature. */
  public boolean usesAttentionTemperatureScaling() {
    return attnTempScale != 0.0f;
  }

  /**
   * The factor the query vector is multiplied by at one position, before attention and after rope.
   *
   * <p>Transcribed from llama.cpp's {@code llm_graph_input_attn_temp::set_input}:
   *
   * <pre>{@code log(floor((pos + offset) / floorScale) + 1) * scale + 1}</pre>
   *
   * <p>The offset is zero for Mistral 3, which sets it explicitly, and the floor is the original
   * pre-extension context length ({@code n_ctx_orig_yarn}). Mistral 3 3B declares a scale of 0.1
   * against a 16384-token original context inside a 262144-token window.
   *
   * <p><b>This is exactly 1.0 for every position below the floor</b>, because {@code floor(pos /
   * floorScale)} is then 0 and {@code log(1)} is 0. That is the intended behaviour -- the tuning
   * exists to temper attention only once a sequence runs past the length the model was trained for
   * -- but it means any test written at ordinary positions passes without exercising anything.
   *
   * @param position the zero-based token position
   * @return the query scale, 1.0 when temperature scaling is absent or the position is below the
   *     floor
   */
  public float attentionTemperatureScale(int position) {
    if (attnTempScale == 0.0f) {
      return 1.0f;
    }
    if (position < 0) {
      throw new IllegalArgumentException("position must be >= 0: " + position);
    }
    // llama.cpp's floor is n_ctx_orig_yarn, which it defaults to n_ctx_train when the file declares
    // no original context length. Requiring the explicit key instead would refuse a Mistral 3 that
    // used linear or no rope scaling, because RopeScaling.linear records originalContext as zero --
    // stricter than the reference, and a rejection rather than a wrong number, but still wrong.
    int floorScale =
        ropeScaling.originalContext() > 0 ? ropeScaling.originalContext() : contextLength;
    double steps = Math.floor((double) position / floorScale);
    return (float) (Math.log(steps + 1.0) * attnTempScale + 1.0);
  }

  /**
   * Whether rotary embeddings use the NeoX half-split layout rather than the interleaved-pair one.
   *
   * <p>Confirmed against the reference implementation's architecture table rather than inferred: it
   * classifies LLAMA as the interleaved-pair layout and GEMMA, GEMMA2, GEMMA3, GEMMA_EMBEDDING,
   * PHI3, QWEN2, QWEN3, QWEN3MOE and HUNYUAN_DENSE as the half-split one. That is exactly the set
   * below, which also independently confirms the Gemma 1, Gemma 2, Phi-3 and Hunyuan entries added
   * here rather than leaving them resting on shape resemblance. QWEN3MOE shares one fall-through
   * with QWEN2 and QWEN3 in that table, and was missing from the set below until the table was
   * re-read -- an omission that would have given every routed Qwen a silently wrong rotary layout.
   * Nothing in a GGUF distinguishes the two layouts -- which one a file needs was fixed when its Q
   * and K rows were written -- so the table is the only place the answer exists, and a wrong choice
   * yields fluent but degraded text, never an error.
   *
   * <p>The algorithm was read and translated, not depended on: this stack is Java plus Rust shims,
   * and llama.cpp and Ollama serve as benchmark arms that our generated tokens are compared
   * against.
   */
  public boolean usesNeoxRope() {
    return architecture == DecoderArchitecture.QWEN2
        || architecture == DecoderArchitecture.QWEN3
        || architecture == DecoderArchitecture.QWEN3MOE
        || architecture == DecoderArchitecture.PHI3
        || architecture == DecoderArchitecture.HUNYUAN_DENSE
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
    if (architecture == DecoderArchitecture.GRANITE) return graniteScalars.embedding();
    return isGemmaFamily() ? (float) Math.sqrt(embeddingDim) : 1.0f;
  }

  /** Scale applied to QK attention scores; Granite persists this rather than deriving it. */
  public float attentionScale() {
    return architecture == DecoderArchitecture.GRANITE
        ? graniteScalars.attention()
        : (float) (1.0 / Math.sqrt(keyLength));
  }

  /** Scale applied to Granite's attention and feed-forward residual updates. */
  public float residualScale() {
    return architecture == DecoderArchitecture.GRANITE ? graniteScalars.residual() : 1.0f;
  }

  /** Divisor applied to Granite logits after the vocabulary projection. */
  public float logitScale() {
    return architecture == DecoderArchitecture.GRANITE ? graniteScalars.logits() : 1.0f;
  }

  /**
   * Whether this graph has Granite-specific arithmetic that generic Llama shortcuts must not use.
   */
  public boolean usesGraniteScaling() {
    return architecture == DecoderArchitecture.GRANITE;
  }

  /** Whether this architecture uses GELU-gated rather than SiLU-gated feed-forward layers. */
  public boolean usesGeluFfn() {
    return isGemmaFamily();
  }

  /** Whether attention output is normalized before its residual addition. */
  public boolean usesPostAttentionNorm() {
    return isGemmaFamily() && !isGemma1();
  }

  /** Whether feed-forward output is normalized before its residual addition. */
  public boolean usesPostFfnNorm() {
    return isGemmaFamily() && !isGemma1();
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

  /** Builds the global RoPE table with the checkpoint's declared scaling algorithm. */
  RotaryTable globalRotaryTable() {
    if (ropeScaling.type() == RopeScalingType.YARN) {
      return RotaryTable.yarn(
          ropeDimensions,
          ropeTheta,
          ropeScaling.factor(),
          ropeScaling.betaFast(),
          ropeScaling.betaSlow(),
          ropeScaling.originalContext(),
          true);
    }
    return new RotaryTable(ropeDimensions, ropeTheta, ropeFrequencyScale);
  }

  /** Builds the unscaled table used by architectures with alternating sliding-window RoPE. */
  RotaryTable slidingWindowRotaryTable() {
    return new RotaryTable(ropeDimensions, slidingWindowRopeTheta, 1.0f);
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
    return !isGemmaFamily() && !usesGraniteScaling();
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
    RopeScaling ropeScaling = ropeScaling(metadata, arch, contextLength, ropeFrequencyScale);
    float slidingWindowRopeTheta =
        getArchFloatKey(metadata, arch, "rope.freq_base_swa").orElse(10_000.0f);
    float rmsNormEps =
        getArchFloatKey(metadata, arch, "attention.layer_norm_rms_epsilon").orElse(1e-5f);
    int slidingWindow = getArchKey(metadata, arch, "attention.sliding_window").orElse(0);
    // Gemma 2 alternates local and global attention every other layer and publishes no
    // sliding_window_pattern key, so the generic default of 6 would make five layers in six local
    // instead of one in two -- wrong attention spans on 26 of 26 layers, with no error. Gemma 3
    // publishes the key or uses the 6 default.
    int defaultSlidingWindowPattern = architecture == DecoderArchitecture.GEMMA2 ? 2 : 6;
    int slidingWindowPattern =
        getArchKey(metadata, arch, "attention.sliding_window_pattern")
            .orElse(defaultSlidingWindowPattern);
    float finalLogitSoftcap =
        getArchFloatKey(metadata, arch, "final_logit_softcapping").orElse(0.0f);
    // Gemma 2 softcaps ATTENTION logits as well as final logits; Gemma 3 dropped the attention one.
    // Zero means no cap, matching the final-logit convention above.
    float attnLogitSoftcap = getArchFloatKey(metadata, arch, "attn_logit_softcapping").orElse(0.0f);
    float attnTempScale =
        getArchFloatKey(metadata, arch, "attention.temperature_scale").orElse(0.0f);

    // Mixture-of-experts feed-forward. Absent means dense, which is every architecture here except
    // qwen3moe: the three keys arrive together or not at all, and the validator enforces that.
    int numExperts = getArchKey(metadata, arch, "expert_count").orElse(0);
    int numExpertsUsed = getArchKey(metadata, arch, "expert_used_count").orElse(0);
    int expertHiddenDim = getArchKey(metadata, arch, "expert_feed_forward_length").orElse(0);

    // Partial rotary: rope.dimension_count below the head dimension means only the leading dims are
    // rotated. RotaryTable rotates the whole head and takes no dimension count, so a model needing
    // partial rotary would load and generate plausible but wrong text. Refused for Phi-3 rather
    // than
    // served incorrectly. Scoped to this architecture on purpose: the field is ignored for every
    // architecture already supported, and newly enforcing it there would reject models that ship
    // today. phi-3.5-mini and phi-3-mini-4k are full rotary (head_dim 96 = rope dim 96) and load;
    // phi-4-mini is 128 vs 96 and is refused until RotaryTable takes a rotary width.
    // The declared rotary width, which may be NARROWER than the head: Phi-4-mini rotates 96 of 128
    // dimensions and leaves the rest untouched. The rotary tables are built at this width, and both
    // rope layouts pair strictly inside it -- NeoX pairs i with i + width/2 and the interleaved
    // form
    // pairs 2i with 2i+1 -- so the dimensions above it are never written. The frequency denominator
    // is this width too, matching ggml, which raises freq_base to -2*i/n_dims with n_dims = n_rot.
    int ropeDimensions = getArchKey(metadata, arch, "rope.dimension_count").orElse(keyLength);
    if (ropeDimensions <= 0 || ropeDimensions > keyLength || (ropeDimensions & 1) != 0) {
      throw new IllegalArgumentException(
          "rope.dimension_count must be positive, even and at most the head dimension "
              + keyLength
              + ", but was "
              + ropeDimensions);
    }

    GraniteScalars graniteScalars =
        architecture == DecoderArchitecture.GRANITE
            ? new GraniteScalars(
                requiredGraniteScale(metadata, arch, "embedding_scale"),
                requiredGraniteScale(metadata, arch, "attention.scale"),
                requiredGraniteScale(metadata, arch, "residual_scale"),
                requiredGraniteScale(metadata, arch, "logit_scale"))
            : GraniteScalars.none();

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
        finalLogitSoftcap,
        attnLogitSoftcap,
        attnTempScale,
        ropeDimensions,
        numExperts,
        numExpertsUsed,
        expertHiddenDim,
        ropeScaling,
        graniteScalars);
  }

  private static float requiredGraniteScale(GgufMetadata metadata, String arch, String suffix) {
    return getArchFloatKey(metadata, arch, suffix)
        .orElseThrow(() -> new IllegalArgumentException("Missing " + arch + "." + suffix));
  }

  private static void requireFinitePositive(String name, float value) {
    if (!(value > 0.0f) || !Float.isFinite(value)) {
      throw new IllegalArgumentException(name + " must be finite and > 0: " + value);
    }
  }

  private static RopeScaling ropeScaling(
      GgufMetadata metadata, String arch, int contextLength, float frequencyScale) {
    String type =
        metadata
            .getString(arch + ".rope.scaling.type")
            .or(() -> metadata.getString("llama.rope.scaling.type"))
            .orElse("linear");
    if ("none".equals(type) || "linear".equals(type)) {
      return RopeScaling.linear("none".equals(type) ? 1.0f : frequencyScale);
    }
    if (!"yarn".equals(type)) {
      throw new IllegalArgumentException("Unsupported RoPE scaling type: " + type);
    }
    float factor = 1.0f / frequencyScale;
    int originalContext =
        getArchKey(metadata, arch, "rope.scaling.original_context_length").orElse(contextLength);
    float betaFast = getArchFloatKey(metadata, arch, "rope.scaling.yarn_beta_fast").orElse(32.0f);
    float betaSlow = getArchFloatKey(metadata, arch, "rope.scaling.yarn_beta_slow").orElse(1.0f);
    return RopeScaling.yarn(factor, originalContext, betaFast, betaSlow);
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
