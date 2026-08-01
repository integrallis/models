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

import com.integrallis.models.backend.purejava.cache.LayeredKvCache;
import com.integrallis.models.backend.purejava.cache.LayeredKvCache.AttentionSpan;
import com.integrallis.models.backend.purejava.cache.LayeredKvCache.AttentionView;
import com.integrallis.models.backend.purejava.gemma4.Gemma4TensorLayout.ExpertWeights;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.ops.RotaryTable;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.vectors.core.GgufQ4Kernel;
import com.integrallis.vectors.core.VectorUtil;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Objects;

/** Allocation-stable autoregressive execution of the text-only Gemma 4 decoder graph. */
final class Gemma4ForwardPass {

  private final Gemma4Config config;
  private final Gemma4Weights weights;
  private final LayeredKvCache cache;
  private final Gemma4ExpertCache experts;
  private final RotaryTable[] rotaryTables;

  private final float[] state;
  private final float[] normalized;
  private final float[] query;
  private final float[] key;
  private final float[] value;
  private final float[] attentionOutput;
  private final float[] attentionScores;
  private final float[] projected;
  private final float[] sharedInput;
  private final float[] sharedGate;
  private final float[] sharedUp;
  private final float[] sharedActivation;
  private final float[] sharedOutput;
  private final float[] routerInput;
  private final float[] routerLogits;
  private final int[] selectedExperts;
  private final float[] routingWeights;
  private final float[] routedInput;
  private final float[] expertGateUp;
  private final float[] expertActivation;
  private final float[] expertOutput;
  private final float[] routedOutput;
  private final float[] combinedOutput;
  private final float[] logits;
  private final byte[] quantizedActivation;
  private final float[] quantizedActivationScales;
  private final int[] quantizedActivationZeroPointCorrections;
  private final short[] quantizedActivationSums;

  private int nextPosition;

  Gemma4ForwardPass(
      Gemma4Config config, Gemma4Weights weights, LayeredKvCache cache, Gemma4ExpertCache experts) {
    this.config = Objects.requireNonNull(config, "config");
    this.weights = Objects.requireNonNull(weights, "weights");
    this.cache = Objects.requireNonNull(cache, "cache");
    this.experts = Objects.requireNonNull(experts, "experts");
    if (cache.numLayers() != config.numLayers()) {
      throw new IllegalArgumentException("KV cache layer count does not match Gemma 4 config");
    }
    for (int layer = 0; layer < config.numLayers(); layer++) {
      if (cache.keyDim(layer) != config.keyDim(layer)
          || cache.valueDim(layer) != config.valueDim(layer)) {
        throw new IllegalArgumentException("KV cache dimensions do not match layer " + layer);
      }
    }

    this.rotaryTables = new RotaryTable[config.numLayers()];
    for (int layer = 0; layer < config.numLayers(); layer++) {
      float[] frequencyFactors =
          config.usesSlidingWindow(layer) ? null : weights.ropeFrequencyFactors();
      rotaryTables[layer] =
          new RotaryTable(
              config.ropeDimension(layer), config.ropeTheta(layer), 1.0f, frequencyFactors);
    }

    int dim = config.embeddingDim();
    int maxQueryDim = 0;
    int maxKeyDim = 0;
    int maxValueDim = 0;
    int maxAttentionOutputDim = 0;
    for (int layer = 0; layer < config.numLayers(); layer++) {
      maxQueryDim = Math.max(maxQueryDim, config.queryDim(layer));
      maxKeyDim = Math.max(maxKeyDim, config.keyDim(layer));
      maxValueDim = Math.max(maxValueDim, config.valueDim(layer));
      maxAttentionOutputDim = Math.max(maxAttentionOutputDim, config.attentionOutputDim(layer));
    }
    int maxProjectionInput =
        Math.max(
            Math.max(dim, maxAttentionOutputDim),
            Math.max(config.sharedHiddenDim(), config.expertHiddenDim()));

    this.state = new float[dim];
    this.normalized = new float[dim];
    this.query = new float[maxQueryDim];
    this.key = new float[maxKeyDim];
    this.value = new float[maxValueDim];
    this.attentionOutput = new float[maxAttentionOutputDim];
    this.attentionScores = new float[cache.maxSeqLen()];
    this.projected = new float[dim];
    this.sharedInput = new float[dim];
    this.sharedGate = new float[config.sharedHiddenDim()];
    this.sharedUp = new float[config.sharedHiddenDim()];
    this.sharedActivation = new float[config.sharedHiddenDim()];
    this.sharedOutput = new float[dim];
    this.routerInput = new float[dim];
    this.routerLogits = new float[config.numExperts()];
    this.selectedExperts = new int[config.numExpertsUsed()];
    this.routingWeights = new float[config.numExpertsUsed()];
    this.routedInput = new float[dim];
    this.expertGateUp = new float[Math.multiplyExact(2, config.expertHiddenDim())];
    this.expertActivation = new float[config.expertHiddenDim()];
    this.expertOutput = new float[dim];
    this.routedOutput = new float[dim];
    this.combinedOutput = new float[dim];
    this.logits = new float[config.vocabSize()];
    this.quantizedActivation = new byte[maxProjectionInput];
    this.quantizedActivationScales = new float[(maxProjectionInput + 31) / 32];
    this.quantizedActivationZeroPointCorrections = new int[(maxProjectionInput + 3) / 4];
    this.quantizedActivationSums = new short[(maxProjectionInput + 15) / 16];
  }

  /** Executes one token and returns stable logits. */
  float[] forward(int token, int position) {
    return forwardTransient(token, position).clone();
  }

  /** Executes one token and returns logits owned until the next forward operation. */
  float[] forwardTransient(int token, int position) {
    return forwardInternal(token, position, true);
  }

  /** Evaluates a prompt while projecting vocabulary logits only for its last token. */
  float[] prefill(int[] tokens, int startPosition) {
    Objects.requireNonNull(tokens, "tokens");
    if (tokens.length == 0) {
      throw new IllegalArgumentException("tokens must not be empty");
    }
    if (startPosition != nextPosition) {
      throw new IllegalArgumentException(
          "position must be sequential: expected " + nextPosition + ", got " + startPosition);
    }
    if (tokens.length > cache.maxSeqLen() - startPosition) {
      throw new IllegalArgumentException(
          "prompt exceeds context length: " + (startPosition + (long) tokens.length));
    }
    for (int index = 0; index < tokens.length - 1; index++) {
      forwardInternal(tokens[index], startPosition + index, false);
    }
    return forwardInternal(tokens[tokens.length - 1], startPosition + tokens.length - 1, true)
        .clone();
  }

  /** Returns the next absolute sequence position. */
  int checkpoint() {
    return nextPosition;
  }

  /** Discards sequence state at and after a prior checkpoint. */
  void rewind(int checkpoint) {
    if (checkpoint < 0 || checkpoint > nextPosition) {
      throw new IllegalArgumentException(
          "checkpoint must be between 0 and " + nextPosition + ": " + checkpoint);
    }
    cache.discardFrom(checkpoint);
    nextPosition = checkpoint;
  }

  /** Clears sequence state while retaining allocated scratch and KV storage. */
  void reset() {
    cache.clear();
    nextPosition = 0;
  }

  private float[] forwardInternal(int token, int position, boolean computeLogits) {
    if (position != nextPosition) {
      throw new IllegalArgumentException(
          "position must be sequential: expected " + nextPosition + ", got " + position);
    }
    if (position >= cache.maxSeqLen()) {
      throw new IllegalArgumentException("position exceeds context length: " + position);
    }

    weights.embedToken(token, state);
    multiply(state, config.embeddingScale());
    try {
      for (int layer = 0; layer < config.numLayers(); layer++) {
        executeLayer(layer, position);
      }
      nextPosition++;
    } catch (RuntimeException | Error failure) {
      cache.discardFrom(position);
      throw failure;
    }

    if (!computeLogits) {
      return null;
    }
    TensorOps.rmsNorm(
        normalized, state, weights.outputNorm(), config.embeddingDim(), config.rmsNormEps());
    project(
        weights.tokenEmbedding(),
        weights.tokenEmbeddingType(),
        config.vocabSize(),
        config.embeddingDim(),
        normalized,
        logits);
    Gemma4Math.softcap(logits, config.finalLogitSoftcap());
    return logits;
  }

  private void executeLayer(int layer, int position) {
    Gemma4Weights.LayerWeights layerWeights = weights.layer(layer);
    int dim = config.embeddingDim();
    int headDim = config.headDim(layer);
    int queryDim = config.queryDim(layer);
    int keyDim = config.keyDim(layer);
    int valueDim = config.valueDim(layer);
    int attentionDim = config.attentionOutputDim(layer);

    TensorOps.rmsNorm(normalized, state, layerWeights.attentionNorm(), dim, config.rmsNormEps());
    project(layerWeights.queryProjection(), normalized, query);
    project(layerWeights.keyProjection(), normalized, key);
    if (layerWeights.valueProjection() == null) {
      System.arraycopy(key, 0, value, 0, valueDim);
    } else {
      project(layerWeights.valueProjection(), normalized, value);
    }

    for (int head = 0; head < config.numHeads(); head++) {
      TensorOps.rmsNorm(
          query,
          head * headDim,
          query,
          head * headDim,
          layerWeights.queryNorm(),
          headDim,
          config.rmsNormEps());
    }
    for (int head = 0; head < config.numKvHeads(layer); head++) {
      int offset = head * headDim;
      TensorOps.rmsNorm(
          key, offset, key, offset, layerWeights.keyNorm(), headDim, config.rmsNormEps());
      Gemma4Math.normalizeWithoutWeight(value, offset, value, offset, headDim, config.rmsNormEps());
    }

    RotaryTable rotary = rotaryTables[layer];
    rotary.prepare(position);
    for (int head = 0; head < config.numHeads(); head++) {
      rotary.apply(query, head * headDim, true);
    }
    for (int head = 0; head < config.numKvHeads(layer); head++) {
      rotary.apply(key, head * headDim, true);
    }

    cache.store(layer, position, key, 0, value, 0);
    computeAttention(layer, position, queryDim, keyDim, valueDim, attentionDim);
    project(layerWeights.attentionOutputProjection(), attentionOutput, projected);
    TensorOps.rmsNorm(
        projected, projected, layerWeights.attentionPostNorm(), dim, config.rmsNormEps());
    add(state, projected, dim);

    executeSharedFfn(layerWeights);
    executeRoutedFfn(layer, layerWeights);
    for (int index = 0; index < dim; index++) {
      combinedOutput[index] = sharedOutput[index] + routedOutput[index];
    }
    TensorOps.rmsNorm(
        combinedOutput,
        combinedOutput,
        layerWeights.combinedFfnPostNorm(),
        dim,
        config.rmsNormEps());
    add(state, combinedOutput, dim);
    multiply(state, layerWeights.layerOutputScale());
  }

  private void executeSharedFfn(Gemma4Weights.LayerWeights layer) {
    int dim = config.embeddingDim();
    int hidden = config.sharedHiddenDim();
    TensorOps.rmsNorm(sharedInput, state, layer.sharedFfnNorm(), dim, config.rmsNormEps());
    project(layer.sharedGateProjection(), sharedInput, sharedGate);
    project(layer.sharedUpProjection(), sharedInput, sharedUp);
    TensorOps.geluGlu(sharedActivation, sharedGate, sharedUp, hidden);
    project(layer.sharedDownProjection(), sharedActivation, sharedOutput);
    TensorOps.rmsNorm(
        sharedOutput, sharedOutput, layer.sharedFfnPostNorm(), dim, config.rmsNormEps());
  }

  private void executeRoutedFfn(int layerIndex, Gemma4Weights.LayerWeights layer) {
    int dim = config.embeddingDim();
    int hidden = config.expertHiddenDim();
    TensorOps.rmsNorm(routedInput, state, layer.routedFfnNorm(), dim, config.rmsNormEps());
    Gemma4Math.normalizeRouterInput(routerInput, state, layer.routerScale(), config.rmsNormEps());
    project(layer.routerProjection(), routerInput, routerLogits);
    Gemma4Math.selectExperts(routerLogits, layer.expertScales(), selectedExperts, routingWeights);
    Arrays.fill(routedOutput, 0.0f);

    for (int rank = 0; rank < selectedExperts.length; rank++) {
      int expert = selectedExperts[rank];
      ExpertWeights source = weights.expertLayout().layer(layerIndex).expert(expert);
      try (Gemma4ExpertCache.Lease lease = experts.acquire(layerIndex, expert)) {
        project(lease.gateUp(), source.gateUp().type(), 2 * hidden, dim, routedInput, expertGateUp);
        TensorOps.geluGlu(expertActivation, 0, expertGateUp, 0, expertGateUp, hidden, hidden);
        project(lease.down(), source.down().type(), dim, hidden, expertActivation, expertOutput);
        addScaled(routedOutput, expertOutput, routingWeights[rank], dim);
      } catch (IOException exception) {
        throw new UncheckedIOException(
            "Failed to load Gemma 4 expert " + expert + " in layer " + layerIndex, exception);
      }
    }
    TensorOps.rmsNorm(
        routedOutput, routedOutput, layer.routedFfnPostNorm(), dim, config.rmsNormEps());
  }

  private void computeAttention(
      int layer, int position, int queryDim, int keyDim, int valueDim, int attentionDim) {
    int headDim = config.headDim(layer);
    int groupSize = config.numHeads() / config.numKvHeads(layer);
    int fromPosition = config.attentionStartPosition(layer, position);
    AttentionView view = cache.attentionView(layer, fromPosition, position + 1);
    float[] keys = cache.keyBuffer(layer);
    float[] values = cache.valueBuffer(layer);
    Arrays.fill(attentionOutput, 0, attentionDim, 0.0f);

    for (int queryHead = 0; queryHead < queryDim / headDim; queryHead++) {
      int kvHead = queryHead / groupSize;
      int scoreCount = 0;
      for (int spanIndex = 0; spanIndex < view.spanCount(); spanIndex++) {
        AttentionSpan span = view.span(spanIndex);
        for (int row = 0; row < span.positionCount(); row++) {
          int keyOffset = span.keyOffset() + row * keyDim + kvHead * headDim;
          attentionScores[scoreCount++] =
              VectorUtil.dotProduct(query, queryHead * headDim, keys, keyOffset, headDim)
                  * config.attentionScale();
        }
      }
      TensorOps.softmax(attentionScores, 0, scoreCount);

      int score = 0;
      int outputOffset = queryHead * headDim;
      for (int spanIndex = 0; spanIndex < view.spanCount(); spanIndex++) {
        AttentionSpan span = view.span(spanIndex);
        for (int row = 0; row < span.positionCount(); row++) {
          int valueOffset = span.valueOffset() + row * valueDim + kvHead * headDim;
          float probability = attentionScores[score++];
          for (int column = 0; column < headDim; column++) {
            attentionOutput[outputOffset + column] += probability * values[valueOffset + column];
          }
        }
      }
    }
  }

  private void project(Gemma4Weights.Matrix matrix, float[] input, float[] output) {
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

  private static void add(float[] target, float[] addition, int length) {
    for (int index = 0; index < length; index++) {
      target[index] += addition[index];
    }
  }

  private static void addScaled(float[] target, float[] addition, float scale, int length) {
    for (int index = 0; index < length; index++) {
      target[index] += addition[index] * scale;
    }
  }

  private static void multiply(float[] values, float scale) {
    for (int index = 0; index < values.length; index++) {
      values[index] *= scale;
    }
  }
}
