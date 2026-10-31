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

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.ops.ExpertRouting;
import com.integrallis.models.backend.purejava.ops.RotaryTable;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.vectors.core.GgufQ4Kernel;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Objects;

/**
 * The LFM2 decoder: a hybrid of attention layers and gated short convolutions.
 *
 * <p>Every layer has the same outer shape -- norm, mixer, residual, norm, gated feed-forward,
 * residual -- and only the mixer differs. Six of LFM2.5-1.2B's sixteen layers attend; the rest run
 * {@link Lfm2ShortConv}, which carries a recurrent state instead of a key-value cache. That is the
 * reason this is its own decoder rather than an addition to the Llama one: two kinds of per-layer
 * state with different shapes and different update rules.
 *
 * <p>Serial and single-sequence on purpose. A batched prefill would have to advance the convolution
 * state token by token anyway -- the recurrence is inherently sequential -- so the batched form
 * saves only the projections, and getting it right is worth less than getting the recurrence right
 * first.
 *
 * <p>Rope is the NeoX half-split layout, read from llama.cpp's architecture table, and is applied
 * after the per-head query and key norms, matching the reference's order.
 */
public final class Lfm2ForwardPass {

  private final Lfm2Config config;
  private final Lfm2Weights weights;
  private final RotaryTable rotary;
  private final int maxSeqLen;

  private final float[] state;
  private final float[] normalized;
  private final float[] query;
  private final float[] key;
  private final float[] value;
  private final float[] attentionOut;
  private final float[] projected;
  private final float[] scores;
  private final float[] gate;
  private final float[] up;
  private final float[] ffnOut;
  private final float[] convProjected;

  /**
   * Per-layer residual observer, mirroring {@code LlamaForwardPass.LayerObserver}.
   *
   * <p>Added because this decoder had no way to see inside it, and that cost a diagnosis. The
   * {@code lfm2moe} port generated {@code " France is France is"} where llama.cpp generated {@code
   * " Paris. The city of Paris is known for its beautiful landmarks"}, and with the dense path
   * green on a real artifact and the routed path green on a toy one, there was nothing left to
   * narrow it with: a 24-layer model that is wrong somewhere needs a per-layer comparison, and this
   * decoder could not produce one.
   *
   * <p>Null when nobody is watching, so the hot path pays one null check per layer.
   */
  interface LayerObserver {
    void onLayerComplete(int layer, int position, float[] state, int offset, int length);
  }

  private LayerObserver layerObserver;
  private final float[] routerLogits;
  private final int[] selectedExperts;
  private final float[] routingWeights;
  private final float[] expertOut;
  private final float[] convGated;
  private final float[] logits;

  private final byte[] quantizedActivation;
  private final float[] quantizedActivationScales;
  private final int[] quantizedActivationZeroPointCorrections;
  private final short[] quantizedActivationSums;

  /** Key and value history, allocated only for the layers that attend. */
  private final float[][] keyCache;

  private final float[][] valueCache;

  /** The convolution's shift register, allocated only for the layers that do not attend. */
  private final float[][] convState;

  private int nextPosition;

  /** Attaches a per-layer observer; pass null to detach. Diagnostics only. */
  void observeLayers(LayerObserver observer) {
    this.layerObserver = observer;
  }

  /**
   * Loads a decoder from a GGUF, keeping the weight layout package-private.
   *
   * @param file the parsed GGUF
   * @param config the parsed LFM2 contract
   * @param maxSeqLen the sequence limit to allocate for
   * @return a ready decoder
   */
  public static Lfm2ForwardPass fromGgufFile(GgufFile file, Lfm2Config config, int maxSeqLen) {
    return new Lfm2ForwardPass(config, Lfm2Weights.fromGgufFile(file, config), maxSeqLen);
  }

  Lfm2ForwardPass(Lfm2Config config, Lfm2Weights weights, int maxSeqLen) {
    this.config = Objects.requireNonNull(config, "config");
    this.weights = Objects.requireNonNull(weights, "weights");
    if (maxSeqLen <= 0 || maxSeqLen > config.contextLength()) {
      throw new IllegalArgumentException(
          "maxSeqLen must be in 1.." + config.contextLength() + ", was " + maxSeqLen);
    }
    this.maxSeqLen = maxSeqLen;
    this.rotary = new RotaryTable(config.headDim(), config.ropeTheta(), 1.0f);

    int dim = config.embeddingDim();
    this.state = new float[dim];
    this.normalized = new float[dim];
    this.query = new float[config.queryDim()];
    this.attentionOut = new float[config.queryDim()];
    this.projected = new float[dim];
    this.scores = new float[maxSeqLen];
    // The dense feed-forward is feed_forward_length wide and an expert is
    // expert_feed_forward_length wide, and on lfm2moe those differ (7168 against 1792). One buffer
    // sized for the wider of the two serves both branches; sizing it from hiddenDim alone would
    // overflow on a file whose experts are wider than its dense blocks.
    int ffnWidth =
        config.isMixtureOfExperts()
            ? Math.max(config.hiddenDim(), config.experts().hiddenDim())
            : config.hiddenDim();
    this.gate = new float[ffnWidth];
    this.up = new float[ffnWidth];
    // Hidden-width, not model-width: SwiGLU produces one value per hidden unit and only the down
    // projection returns to the model width.
    this.ffnOut = new float[ffnWidth];
    this.routerLogits = config.isMixtureOfExperts() ? new float[config.experts().count()] : null;
    this.selectedExperts =
        config.isMixtureOfExperts() ? new int[config.experts().usedCount()] : null;
    this.routingWeights =
        config.isMixtureOfExperts() ? new float[config.experts().usedCount()] : null;
    this.expertOut = config.isMixtureOfExperts() ? new float[config.embeddingDim()] : null;
    this.convProjected = new float[3 * dim];
    this.convGated = new float[dim];
    this.logits = new float[config.vocabSize()];

    int maxKeyDim = 0;
    for (int layer = 0; layer < config.numLayers(); layer++) {
      if (config.usesAttention(layer)) {
        maxKeyDim = Math.max(maxKeyDim, config.keyDim(layer));
      }
    }
    this.key = new float[maxKeyDim];
    this.value = new float[maxKeyDim];

    int widest = Math.max(Math.max(dim, config.hiddenDim()), config.queryDim());
    this.quantizedActivation = new byte[widest];
    this.quantizedActivationScales = new float[(widest + 31) / 32];
    this.quantizedActivationZeroPointCorrections = new int[(widest + 3) / 4];
    this.quantizedActivationSums = new short[(widest + 15) / 16];

    this.keyCache = new float[config.numLayers()][];
    this.valueCache = new float[config.numLayers()][];
    this.convState = new float[config.numLayers()][];
    for (int layer = 0; layer < config.numLayers(); layer++) {
      if (config.usesAttention(layer)) {
        keyCache[layer] = new float[maxSeqLen * config.keyDim(layer)];
        valueCache[layer] = new float[maxSeqLen * config.keyDim(layer)];
      } else {
        convState[layer] = new float[config.shortConvStateSize()];
      }
    }
  }

  /** Clears both kinds of state, so a new sequence starts from position zero. */
  public void reset() {
    nextPosition = 0;
    for (int layer = 0; layer < config.numLayers(); layer++) {
      if (convState[layer] != null) {
        java.util.Arrays.fill(convState[layer], 0.0f);
      }
    }
  }

  public int nextPosition() {
    return nextPosition;
  }

  /**
   * Advances one token and returns the vocabulary logits.
   *
   * @param token the token id
   * @param position the position, which must be the next one in sequence
   * @return the logits, valid until the next call
   */
  public float[] forward(int token, int position) {
    return forwardInternal(token, position, true);
  }

  /**
   * Advances one token and returns the final-norm activation instead of the vocabulary logits.
   *
   * <p>Same contract as {@code LlamaForwardPass.hiddenState}: the activation the output projection
   * would consume, with that projection skipped. Skipping it is the whole saving, since it is the
   * widest matmul in the pass, and it is what an embedding needs — LFM2.5-Embedding is an {@code
   * lfm2} decoder, so without this its seven published artifacts cannot be embedded at all.
   *
   * <p>The returned array is backend-owned scratch, valid until the next call. Copy it to keep it.
   *
   * @param token token id to advance with
   * @param position sequential position for the token
   * @return the final-norm activation, owned by this pass
   */
  public float[] hiddenState(int token, int position) {
    return forwardInternal(token, position, false);
  }

  private float[] forwardInternal(int token, int position, boolean projectLogits) {
    if (position != nextPosition) {
      throw new IllegalArgumentException(
          "position must be sequential: expected " + nextPosition + ", got " + position);
    }
    if (position >= maxSeqLen) {
      throw new IllegalArgumentException("position exceeds the sequence limit " + maxSeqLen);
    }
    int dim = config.embeddingDim();
    weights.embedToken(token, state);

    for (int layer = 0; layer < config.numLayers(); layer++) {
      Lfm2Weights.LayerWeights layerWeights = weights.layer(layer);
      TensorOps.rmsNorm(
          normalized, state, layerWeights.attentionNorm(), dim, config.rmsNormEpsilon());
      // Both mixers leave their result in `projected`, so the residual is the same either way.
      if (config.usesAttention(layer)) {
        attend(layer, layerWeights, position);
      } else {
        convolve(layerWeights, layer);
      }
      add(state, projected, dim);

      TensorOps.rmsNorm(normalized, state, layerWeights.ffnNorm(), dim, config.rmsNormEpsilon());
      if (config.usesExperts(layer)) {
        route(layerWeights);
      } else {
        project(layerWeights.gateProjection(), normalized, gate);
        project(layerWeights.upProjection(), normalized, up);
        TensorOps.swiGlu(ffnOut, gate, up, config.hiddenDim());
        project(layerWeights.downProjection(), ffnOut, projected);
      }
      add(state, projected, dim);
      if (layerObserver != null) {
        layerObserver.onLayerComplete(layer, position, state, 0, dim);
      }
    }

    nextPosition++;
    // The final norm is the tensor named token_embd_norm; see Lfm2Weights.
    TensorOps.rmsNorm(normalized, state, weights.outputNorm(), dim, config.rmsNormEpsilon());
    if (!projectLogits) {
      return normalized;
    }
    project(
        weights.outputProjection(),
        weights.outputProjectionType(),
        config.vocabSize(),
        dim,
        normalized,
        logits);
    return logits;
  }

  private void attend(int layer, Lfm2Weights.LayerWeights layerWeights, int position) {
    int headDim = config.headDim();
    int numHeads = config.numHeads();
    int numKvHeads = config.numKvHeads(layer);
    int keyDim = config.keyDim(layer);

    project(layerWeights.queryProjection(), normalized, query);
    project(layerWeights.keyProjection(), normalized, key);
    project(layerWeights.valueProjection(), normalized, value);

    // Per-head norms first, then rope: the reference's order.
    for (int head = 0; head < numHeads; head++) {
      int offset = head * headDim;
      TensorOps.rmsNorm(
          query, offset, query, offset, layerWeights.queryNorm(), headDim, config.rmsNormEpsilon());
    }
    for (int head = 0; head < numKvHeads; head++) {
      int offset = head * headDim;
      TensorOps.rmsNorm(
          key, offset, key, offset, layerWeights.keyNorm(), headDim, config.rmsNormEpsilon());
    }
    rotary.prepare(position);
    for (int head = 0; head < numHeads; head++) {
      rotary.apply(query, head * headDim, true);
    }
    for (int head = 0; head < numKvHeads; head++) {
      rotary.apply(key, head * headDim, true);
    }

    System.arraycopy(key, 0, keyCache[layer], position * keyDim, keyDim);
    System.arraycopy(value, 0, valueCache[layer], position * keyDim, keyDim);

    int groupSize = numHeads / numKvHeads;
    float scale = (float) (1.0 / Math.sqrt(headDim));
    float[] keys = keyCache[layer];
    float[] values = valueCache[layer];
    for (int head = 0; head < numHeads; head++) {
      int kvHead = head / groupSize;
      int queryOffset = head * headDim;
      for (int past = 0; past <= position; past++) {
        int keyOffset = past * keyDim + kvHead * headDim;
        float dot = 0.0f;
        for (int index = 0; index < headDim; index++) {
          dot += query[queryOffset + index] * keys[keyOffset + index];
        }
        scores[past] = dot * scale;
      }
      TensorOps.softmax(scores, 0, position + 1);
      int outOffset = head * headDim;
      java.util.Arrays.fill(attentionOut, outOffset, outOffset + headDim, 0.0f);
      for (int past = 0; past <= position; past++) {
        float weight = scores[past];
        int valueOffset = past * keyDim + kvHead * headDim;
        for (int index = 0; index < headDim; index++) {
          attentionOut[outOffset + index] += weight * values[valueOffset + index];
        }
      }
    }
    project(layerWeights.attentionOutputProjection(), attentionOut, projected);
  }

  /**
   * The routed feed-forward: sigmoid gating with a selection-only bias, then a weighted sum.
   *
   * <p>Routing goes through {@link ExpertRouting#selectExpertsWithSigmoidGating} rather than a
   * local copy, because {@code expert_gating_func = 2} is the whole reason #283 was blocked and a
   * second implementation of it is a second thing to get subtly wrong. {@code lfm2moe} publishes no
   * {@code expert_weights_norm} and no {@code expert_weights_scale}, so the weights are
   * renormalised with a unit scale — which is what upstream's defaults give it.
   *
   * <p>Each expert is evaluated in turn and accumulated, rather than gathered into one batched
   * matmul. That is the slower shape and it is deliberate for a first port: the expert matrices are
   * already non-contiguous slices of one tensor, and a gather would add a copy whose correctness is
   * harder to see than the loop's. The kernel seam is unchanged, so the batched form is a later
   * change measurable against this one.
   */
  private void route(Lfm2Weights.LayerWeights layerWeights) {
    Lfm2Config.Experts experts = config.experts();
    int dim = config.embeddingDim();
    int expertWidth = experts.hiddenDim();

    project(layerWeights.router(), normalized, routerLogits);
    ExpertRouting.selectExpertsWithSigmoidGating(
        routerLogits, layerWeights.selectionBias(), selectedExperts, routingWeights, true, 1.0f);

    Arrays.fill(projected, 0, dim, 0.0f);
    for (int slot = 0; slot < selectedExperts.length; slot++) {
      int expert = selectedExperts[slot];
      project(layerWeights.expertGate()[expert], normalized, gate);
      project(layerWeights.expertUp()[expert], normalized, up);
      TensorOps.swiGlu(ffnOut, gate, up, expertWidth);
      project(layerWeights.expertDown()[expert], ffnOut, expertOut);
      float weight = routingWeights[slot];
      for (int i = 0; i < dim; i++) {
        projected[i] += weight * expertOut[i];
      }
    }
  }

  private void convolve(Lfm2Weights.LayerWeights layerWeights, int layer) {
    project(layerWeights.shortConvInProjection(), normalized, convProjected);
    Lfm2ShortConv.applyToken(
        convProjected,
        config.embeddingDim(),
        layerWeights.shortConvKernel(),
        config.shortConvCache(),
        convState[layer],
        convGated);
    project(layerWeights.shortConvOutProjection(), convGated, projected);
  }

  private static void add(float[] target, float[] addend, int size) {
    for (int index = 0; index < size; index++) {
      target[index] += addend[index];
    }
  }

  private void project(Lfm2Weights.Matrix matrix, float[] input, float[] output) {
    project(matrix.data(), matrix.type(), matrix.rows(), matrix.columns(), input, output);
  }

  private void project(
      MemorySegment matrix,
      GgufTensorType type,
      int rows,
      int columns,
      float[] input,
      float[] output) {
    TensorOps.ggufMatmul(
        output,
        input,
        matrix,
        type,
        rows,
        columns,
        quantizedActivation,
        quantizedActivationScales,
        quantizedActivationZeroPointCorrections,
        quantizedActivationSums,
        GgufQ4Kernel.WIDENED);
  }
}
