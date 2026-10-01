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
package com.integrallis.models.backend.purejava.deepseek2;

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.ops.ExpertRouting;
import com.integrallis.models.backend.purejava.ops.RotaryTable;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.vectors.core.VectorUtil;
import java.util.Arrays;
import java.util.Objects;

/**
 * Stateful pure-Java graph for a DeepSeek-V2 family decoder.
 *
 * <p>One token at a time. Expert selection is per token, so a batched path would need a per-row
 * gather through the expert tensors; the routed Qwen3.5 and the Gemma 4 E-series are contained the
 * same way and for the same reason.
 *
 * <p><b>Latent attention.</b> Per token the layer computes a compressed key-value latent and a
 * single rotary key part, normalises the latent, and decompresses it into a no-rope key and a value
 * for every head. The full key is that no-rope part with the <i>same</i> rotary part appended to
 * every head. The cache therefore holds ordinary per-head keys and values -- this is the
 * decompressing form of MLA, not the absorbed one, which is what the published
 * DeepSeek-Coder-V2-Lite GGUF carries tensors for.
 */
public final class Deepseek2ForwardPass {

  private final Deepseek2Config config;
  private final Deepseek2Weights weights;
  private final RotaryTable rotary;
  private final int capacity;

  private final float[][] keyCache;
  private final float[][] valueCache;

  private final float[] state;
  private final float[] normalized;
  private final float[] projected;
  private final float[] query;
  private final float[] compressed;
  private final float[] latent;
  private final float[] ropeKey;
  private final float[] decompressed;
  private final float[] key;
  private final float[] value;
  private final float[] attended;
  private final float[] scores;
  private final float[] ffnGate;
  private final float[] ffnUp;
  private final float[] routerLogits;
  private final int[] selectedExperts;
  private final float[] routingWeights;
  private final float[] expertGate;
  private final float[] expertUp;
  private final float[] contribution;
  private final float[] sharedGate;
  private final float[] sharedUp;
  private final float[] logits;

  private int nextPosition;

  Deepseek2ForwardPass(Deepseek2Config config, Deepseek2Weights weights, int capacity) {
    this.config = Objects.requireNonNull(config, "config");
    this.weights = Objects.requireNonNull(weights, "weights");
    if (capacity <= 0) {
      throw new IllegalArgumentException("capacity must be > 0: " + capacity);
    }
    this.capacity = capacity;
    // YaRN with the rotary magnitude supplied: this family moves the rest of it into the softmax
    // scale. See Deepseek2Config.ropeAttentionFactor and attentionScale.
    this.rotary =
        config.ropeScalingFactor() > 1.0f
            ? RotaryTable.yarnWithAttentionFactor(
                config.ropeDimension(),
                config.ropeTheta(),
                config.ropeScalingFactor(),
                Deepseek2Config.BETA_FAST,
                Deepseek2Config.BETA_SLOW,
                config.ropeOriginalContext(),
                false,
                config.ropeAttentionFactor())
            : new RotaryTable(config.ropeDimension(), config.ropeTheta(), 1.0f);

    int dim = config.embeddingDim();
    this.keyCache = new float[config.numLayers()][];
    this.valueCache = new float[config.numLayers()][];
    for (int layer = 0; layer < config.numLayers(); layer++) {
      keyCache[layer] = new float[capacity * config.cachedKeyDim()];
      valueCache[layer] = new float[capacity * config.cachedValueDim()];
    }
    this.state = new float[dim];
    this.normalized = new float[dim];
    this.projected = new float[dim];
    this.query = new float[config.queryDim()];
    this.compressed = new float[config.compressedKeyValueDim()];
    this.latent = new float[config.kvLoraRank()];
    this.ropeKey = new float[config.ropeDimension()];
    this.decompressed = new float[config.decompressedKeyValueDim()];
    this.key = new float[config.cachedKeyDim()];
    this.value = new float[config.cachedValueDim()];
    this.attended = new float[config.cachedValueDim()];
    this.scores = new float[capacity];
    this.ffnGate = new float[Math.max(config.hiddenDim(), config.sharedExpertHiddenDim())];
    this.ffnUp = new float[Math.max(config.hiddenDim(), config.sharedExpertHiddenDim())];
    this.routerLogits = new float[config.numExperts()];
    this.selectedExperts = new int[config.numExpertsUsed()];
    this.routingWeights = new float[config.numExpertsUsed()];
    this.expertGate = new float[config.expertHiddenDim()];
    this.expertUp = new float[config.expertHiddenDim()];
    this.contribution = new float[dim];
    this.sharedGate = new float[config.sharedExpertHiddenDim()];
    this.sharedUp = new float[config.sharedExpertHiddenDim()];
    this.logits = new float[config.vocabSize()];
  }

  /** Loads a decoder from a parsed GGUF. */
  public static Deepseek2ForwardPass fromGgufFile(
      GgufFile file, Deepseek2Config config, int capacity) {
    return new Deepseek2ForwardPass(config, Deepseek2Weights.fromGgufFile(file, config), capacity);
  }

  public int nextPosition() {
    return nextPosition;
  }

  public void reset() {
    nextPosition = 0;
  }

  /**
   * Moves the sequence back to an earlier position.
   *
   * <p>Exact, and free. The cache is indexed by absolute position and attention reads only
   * positions up to the current one, so the entries above the checkpoint are unreachable until they
   * are overwritten on the way forward again. There is nothing to clear and nothing to replay --
   * unlike the recurrent decoders here, where a rewind means re-running the retained prefix.
   *
   * @param checkpoint the position to return to, between 0 and the current one
   */
  public void rewindTo(int checkpoint) {
    if (checkpoint < 0 || checkpoint > nextPosition) {
      throw new IllegalArgumentException(
          "checkpoint must be between 0 and " + nextPosition + ": " + checkpoint);
    }
    nextPosition = checkpoint;
  }

  /**
   * Advances one token and returns the vocabulary logits.
   *
   * @param token the token id
   * @param position the position, which must be the next one in sequence
   * @return the logits. <b>The graph's own buffer</b>, valid until the next call: clone it to keep
   *     it.
   */
  public float[] forward(int token, int position) {
    if (position != nextPosition) {
      throw new IllegalArgumentException(
          "position must be sequential: expected " + nextPosition + ", got " + position);
    }
    if (position >= capacity) {
      throw new IllegalArgumentException("position exceeds capacity " + capacity + ": " + position);
    }
    int dim = config.embeddingDim();
    weights.embedToken(token, state);

    for (int layer = 0; layer < config.numLayers(); layer++) {
      Deepseek2Weights.LayerWeights weight = weights.layer(layer);
      TensorOps.rmsNorm(normalized, state, weight.attentionNorm(), dim, config.rmsNormEpsilon());
      latentAttention(layer, weight, position);
      for (int index = 0; index < dim; index++) {
        state[index] += projected[index];
      }

      TensorOps.rmsNorm(normalized, state, weight.ffnNorm(), dim, config.rmsNormEpsilon());
      if (weight.moe() != null) {
        routedFeedForward(weight.moe(), dim);
      } else {
        denseFeedForward(weight, dim);
      }
      for (int index = 0; index < dim; index++) {
        state[index] += projected[index];
      }
    }

    TensorOps.rmsNorm(state, state, weights.outputNorm(), dim, config.rmsNormEpsilon());
    project(weights.output(), state, logits);
    nextPosition++;
    return logits;
  }

  /**
   * Multi-head Latent Attention, in its decompressing form.
   *
   * <p>The query splits into a no-rope part and a rotary part per head. The key-value projection
   * produces a latent plus <b>one</b> rotary key part, which every head shares -- that sharing is
   * the "MQA" in {@code attn_kv_a_mqa}, and copying it to each head is what makes the cached keys
   * ordinary per-head keys.
   */
  private void latentAttention(int layer, Deepseek2Weights.LayerWeights weight, int position) {
    int heads = config.numHeads();
    int keyLength = config.keyLength();
    int noRope = config.noRopeDimension();
    int ropeDim = config.ropeDimension();
    int valueLength = config.valueLength();

    project(weight.query(), normalized, query);
    project(weight.keyValueCompress(), normalized, compressed);
    System.arraycopy(compressed, 0, latent, 0, config.kvLoraRank());
    System.arraycopy(compressed, config.kvLoraRank(), ropeKey, 0, ropeDim);

    rotary.prepare(position);
    // NORM rotary, pairing adjacent elements, NOT the split-half NeoX form. The reference maps this
    // family to LLAMA_ROPE_TYPE_NORM (llama-model.cpp, the DEEPSEEK2 case), unlike the Qwen and
    // Gemma
    // families on the Llama path, where LlamaConfig.usesNeoxRope() selects the split-half form.
    // Using
    // NeoX here rotates the wrong pairs at the right angles, which scrambles position and leaves
    // the
    // decoder emitting newlines -- and the toy test could not catch it, because the scalar
    // reference
    // paired the halves the same wrong way.
    // The query's rotary half sits AFTER its no-rope half within each head.
    for (int head = 0; head < heads; head++) {
      rotary.apply(query, head * keyLength + noRope, false);
    }
    rotary.apply(ropeKey, 0, false);

    TensorOps.rmsNorm(
        latent, latent, weight.keyValueLatentNorm(), config.kvLoraRank(), config.rmsNormEpsilon());
    project(weight.keyValueDecompress(), latent, decompressed);

    // attn_kv_b emits, per head, a no-rope key followed by a value.
    int stride = noRope + valueLength;
    for (int head = 0; head < heads; head++) {
      System.arraycopy(decompressed, head * stride, key, head * keyLength, noRope);
      // The same rotary part on every head.
      System.arraycopy(ropeKey, 0, key, head * keyLength + noRope, ropeDim);
      System.arraycopy(
          decompressed, head * stride + noRope, value, head * valueLength, valueLength);
    }
    System.arraycopy(
        key, 0, keyCache[layer], position * config.cachedKeyDim(), config.cachedKeyDim());
    System.arraycopy(
        value, 0, valueCache[layer], position * config.cachedValueDim(), config.cachedValueDim());

    float scale = config.attentionScale();
    Arrays.fill(attended, 0, config.cachedValueDim(), 0.0f);
    for (int head = 0; head < heads; head++) {
      for (int at = 0; at <= position; at++) {
        scores[at] =
            VectorUtil.dotProduct(
                    query,
                    head * keyLength,
                    keyCache[layer],
                    at * config.cachedKeyDim() + head * keyLength,
                    keyLength)
                * scale;
      }
      TensorOps.softmax(scores, 0, position + 1);
      for (int at = 0; at <= position; at++) {
        VectorUtil.addScaledInPlace(
            attended,
            head * valueLength,
            valueCache[layer],
            at * config.cachedValueDim() + head * valueLength,
            valueLength,
            scores[at]);
      }
    }
    project(weight.attentionOutput(), attended, projected);
  }

  private void denseFeedForward(Deepseek2Weights.LayerWeights weight, int dim) {
    project(weight.ffnGate(), normalized, ffnGate);
    project(weight.ffnUp(), normalized, ffnUp);
    VectorUtil.swiGlu(ffnGate, 0, ffnGate, 0, ffnUp, 0, config.hiddenDim());
    project(weight.ffnDown(), ffnGate, projected);
  }

  /**
   * The routed feed-forward: the selected experts plus the fused shared expert.
   *
   * <p>Routing weights are <b>not</b> renormalised -- see {@link
   * ExpertRouting#selectExpertsWithoutRenormalisation}. The shared expert is added unweighted; the
   * two shared experts the metadata declares are already fused into one wide tensor, so this is one
   * SwiGLU and not two.
   */
  private void routedFeedForward(Deepseek2Weights.MoeFeedForward moe, int dim) {
    project(moe.router(), normalized, routerLogits);
    ExpertRouting.selectExpertsWithoutRenormalisation(
        routerLogits,
        config.numExperts(),
        config.numExpertsUsed(),
        selectedExperts,
        routingWeights);
    if (config.scalesExpertWeights()) {
      for (int slot = 0; slot < config.numExpertsUsed(); slot++) {
        routingWeights[slot] *= config.expertWeightsScale();
      }
    }

    Arrays.fill(projected, 0, dim, 0.0f);
    for (int slot = 0; slot < config.numExpertsUsed(); slot++) {
      int expert = selectedExperts[slot];
      project(moe.gate()[expert], normalized, expertGate);
      project(moe.up()[expert], normalized, expertUp);
      VectorUtil.swiGlu(expertGate, 0, expertGate, 0, expertUp, 0, config.expertHiddenDim());
      project(moe.down()[expert], expertGate, contribution);
      float weight = routingWeights[slot];
      for (int index = 0; index < dim; index++) {
        projected[index] += weight * contribution[index];
      }
    }

    project(moe.sharedGate(), normalized, sharedGate);
    project(moe.sharedUp(), normalized, sharedUp);
    VectorUtil.swiGlu(sharedGate, 0, sharedGate, 0, sharedUp, 0, config.sharedExpertHiddenDim());
    project(moe.sharedDown(), sharedGate, contribution);
    for (int index = 0; index < dim; index++) {
      projected[index] += contribution[index];
    }
  }

  private void project(Deepseek2Weights.Matrix matrix, float[] input, float[] output) {
    TensorOps.ggufMatmul(
        output, input, matrix.data(), matrix.type(), matrix.rows(), matrix.columns());
  }
}
