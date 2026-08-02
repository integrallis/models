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

import com.integrallis.models.api.LogitBatch;
import com.integrallis.models.backend.purejava.cache.LayeredKvCache;
import com.integrallis.models.backend.purejava.cache.LayeredKvCache.AttentionSpan;
import com.integrallis.models.backend.purejava.cache.LayeredKvCache.AttentionView;
import com.integrallis.models.backend.purejava.gemma4.Gemma4TensorLayout.ExpertWeights;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.ops.RotaryTable;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import com.integrallis.vectors.core.GgufQ4Kernel;
import com.integrallis.vectors.core.VectorUtil;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.lang.foreign.MemorySegment;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Objects;

/** Allocation-stable autoregressive execution of the text-only Gemma 4 decoder graph. */
final class Gemma4ForwardPass {

  private static final int MAX_SESSION_BATCH_SIZE = 32;
  private static final int MAX_PREFILL_BATCH_SIZE = 32;
  private static final int MAX_INDEPENDENT_ROUTE_GROUP_SIZE = 16;

  /** Mutable sequence state sharing this graph's immutable weights and expert cache. */
  static final class Session {
    private final Gemma4ForwardPass owner;
    private final LayeredKvCache cache;
    private int nextPosition;

    private Session(Gemma4ForwardPass owner, LayeredKvCache cache) {
      this.owner = owner;
      this.cache = cache;
    }

    int checkpoint() {
      return nextPosition;
    }
  }

  private record F32Scratch(int rows, int columns, float[] input, float[] output) {}

  private final Gemma4Config config;
  private final Gemma4Weights weights;
  private final Session defaultSession;
  private final Gemma4Experts experts;
  private final GgufBatchedMatrixKernel batchedMatrixKernel;
  private final RotaryTable[] rotaryTables;

  private final float[] state;
  private final float[] normalized;
  private final float[] query;
  private final float[] key;
  private final float[] value;
  private final float[][] attentionOutputs;
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
  private final float[][] expertGateUps;
  private final float[][] expertActivations;
  private final float[][] expertOutputs;
  private final Gemma4Experts.Lease[] expertLeases;
  private final MemorySegment[] groupedExpertWeights;
  private final GgufTensorType[] groupedExpertTypes;
  private final int[] groupedExpertRows;
  private final float[] routedOutput;
  private final float[] combinedOutput;
  private final float[] logits;
  private final byte[] quantizedActivation;
  private final float[] quantizedActivationScales;
  private final int[] quantizedActivationZeroPointCorrections;
  private final short[] quantizedActivationSums;

  private final boolean batchedPrefill;
  private final int prefillBatchCapacity;
  private final float[] batchState;
  private final float[] batchNormalized;
  private final float[] batchQuery;
  private final float[] batchKey;
  private final float[] batchValue;
  private final float[] batchAttentionOutput;
  private final float[] batchProjected;
  private final float[] batchSharedGate;
  private final float[] batchSharedUp;
  private final float[] batchSharedActivation;
  private final float[] batchSharedOutput;
  private final float[] batchRouterInput;
  private final float[] batchRouterLogits;
  private final int[] batchSelectedExperts;
  private final float[] batchRoutingWeights;
  private final float[] batchRoutedOutput;
  private final float[] batchCombinedOutput;
  private final int[] expertRouteTokens;
  private final int[] expertRouteRanks;
  private final int[] prefillExpertOrder;
  private final int[] prefillExpertLastRoute;
  private final int[] prefillExpertRouteStarts;
  private final int[] prefillExpertRouteCounts;
  private final int[] groupedExpertBatchSizes;
  private final float[][] groupedExpertInputs;
  private final float[][] groupedExpertGateUps;
  private final float[][] groupedExpertActivations;
  private final float[][] groupedExpertOutputs;
  private final F32Scratch[] batchF32Scratch;

  private float[] verificationLogits = new float[0];
  private float[] sessionBatchLogits = new float[0];

  Gemma4ForwardPass(
      Gemma4Config config, Gemma4Weights weights, LayeredKvCache cache, Gemma4Experts experts) {
    this(config, weights, cache, experts, GgufBatchedMatrixKernel.none());
  }

  Gemma4ForwardPass(
      Gemma4Config config,
      Gemma4Weights weights,
      LayeredKvCache cache,
      Gemma4Experts experts,
      GgufBatchedMatrixKernel batchedMatrixKernel) {
    this.config = Objects.requireNonNull(config, "config");
    this.weights = Objects.requireNonNull(weights, "weights");
    LayeredKvCache defaultCache = Objects.requireNonNull(cache, "cache");
    this.experts = Objects.requireNonNull(experts, "experts");
    this.batchedMatrixKernel = Objects.requireNonNull(batchedMatrixKernel, "batchedMatrixKernel");
    if (defaultCache.numLayers() != config.numLayers()) {
      throw new IllegalArgumentException("KV cache layer count does not match Gemma 4 config");
    }
    for (int layer = 0; layer < config.numLayers(); layer++) {
      if (defaultCache.keyDim(layer) != config.keyDim(layer)
          || defaultCache.valueDim(layer) != config.valueDim(layer)) {
        throw new IllegalArgumentException("KV cache dimensions do not match layer " + layer);
      }
    }
    this.defaultSession = new Session(this, defaultCache);

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
    for (int layer = 0; layer < config.numLayers(); layer++) {
      maxQueryDim = Math.max(maxQueryDim, config.queryDim(layer));
      maxKeyDim = Math.max(maxKeyDim, config.keyDim(layer));
      maxValueDim = Math.max(maxValueDim, config.valueDim(layer));
    }
    this.attentionOutputs = exactAttentionOutputs(config);
    int maxAttentionOutputDim = maxLength(attentionOutputs);
    int maxProjectionInput =
        Math.max(
            Math.max(dim, maxAttentionOutputDim),
            Math.max(config.sharedHiddenDim(), config.expertHiddenDim()));

    this.state = new float[dim];
    this.normalized = new float[dim];
    this.query = new float[maxQueryDim];
    this.key = new float[maxKeyDim];
    this.value = new float[maxValueDim];
    this.attentionScores = new float[defaultCache.maxSeqLen()];
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
    this.expertGateUps = new float[config.numExpertsUsed()][2 * config.expertHiddenDim()];
    this.expertActivations = new float[config.numExpertsUsed()][config.expertHiddenDim()];
    this.expertOutputs = new float[config.numExpertsUsed()][dim];
    int expertGroupCapacity = Math.max(config.numExpertsUsed(), MAX_INDEPENDENT_ROUTE_GROUP_SIZE);
    this.expertLeases = new Gemma4Experts.Lease[expertGroupCapacity];
    this.groupedExpertWeights = new MemorySegment[expertGroupCapacity];
    this.groupedExpertTypes = new GgufTensorType[expertGroupCapacity];
    this.groupedExpertRows = new int[expertGroupCapacity];
    this.routedOutput = new float[dim];
    this.combinedOutput = new float[dim];
    this.logits = new float[config.vocabSize()];
    this.quantizedActivation = new byte[maxProjectionInput];
    this.quantizedActivationScales = new float[(maxProjectionInput + 31) / 32];
    this.quantizedActivationZeroPointCorrections = new int[(maxProjectionInput + 3) / 4];
    this.quantizedActivationSums = new short[(maxProjectionInput + 15) / 16];

    this.batchedPrefill = supportsBatchedPrefill(config, weights, batchedMatrixKernel);
    this.prefillBatchCapacity = batchedPrefill ? MAX_PREFILL_BATCH_SIZE : 0;
    this.batchState = batchBuffer(prefillBatchCapacity, dim);
    this.batchNormalized = batchBuffer(prefillBatchCapacity, dim);
    this.batchQuery = batchBuffer(prefillBatchCapacity, maxQueryDim);
    this.batchKey = batchBuffer(prefillBatchCapacity, maxKeyDim);
    this.batchValue = batchBuffer(prefillBatchCapacity, maxValueDim);
    this.batchAttentionOutput = batchBuffer(prefillBatchCapacity, maxAttentionOutputDim);
    this.batchProjected = batchBuffer(prefillBatchCapacity, dim);
    this.batchSharedGate = batchBuffer(prefillBatchCapacity, config.sharedHiddenDim());
    this.batchSharedUp = batchBuffer(prefillBatchCapacity, config.sharedHiddenDim());
    this.batchSharedActivation = batchBuffer(prefillBatchCapacity, config.sharedHiddenDim());
    this.batchSharedOutput = batchBuffer(prefillBatchCapacity, dim);
    this.batchRouterInput = batchBuffer(prefillBatchCapacity, dim);
    this.batchRouterLogits = batchBuffer(prefillBatchCapacity, config.numExperts());
    this.batchSelectedExperts =
        new int[Math.multiplyExact(prefillBatchCapacity, config.numExpertsUsed())];
    this.batchRoutingWeights =
        new float[Math.multiplyExact(prefillBatchCapacity, config.numExpertsUsed())];
    this.batchRoutedOutput = batchBuffer(prefillBatchCapacity, dim);
    this.batchCombinedOutput = batchBuffer(prefillBatchCapacity, dim);
    int routeCapacity = Math.multiplyExact(prefillBatchCapacity, config.numExpertsUsed());
    this.expertRouteTokens = new int[routeCapacity];
    this.expertRouteRanks = new int[routeCapacity];
    this.prefillExpertOrder = new int[batchedPrefill ? config.numExperts() : 0];
    this.prefillExpertLastRoute = new int[batchedPrefill ? config.numExperts() : 0];
    this.prefillExpertRouteStarts = new int[batchedPrefill ? config.numExperts() : 0];
    this.prefillExpertRouteCounts = new int[batchedPrefill ? config.numExperts() : 0];
    this.groupedExpertBatchSizes = new int[MAX_INDEPENDENT_ROUTE_GROUP_SIZE];
    this.groupedExpertInputs =
        matrixBuffer(MAX_INDEPENDENT_ROUTE_GROUP_SIZE, prefillBatchCapacity * dim);
    this.groupedExpertGateUps =
        matrixBuffer(
            MAX_INDEPENDENT_ROUTE_GROUP_SIZE, prefillBatchCapacity * 2 * config.expertHiddenDim());
    this.groupedExpertActivations =
        matrixBuffer(
            MAX_INDEPENDENT_ROUTE_GROUP_SIZE, prefillBatchCapacity * config.expertHiddenDim());
    this.groupedExpertOutputs =
        matrixBuffer(MAX_INDEPENDENT_ROUTE_GROUP_SIZE, prefillBatchCapacity * dim);
    this.batchF32Scratch = batchedPrefill ? createF32Scratch(config, weights) : new F32Scratch[0];
  }

  /** Executes one token and returns stable logits. */
  float[] forward(int token, int position) {
    return forward(defaultSession, token, position);
  }

  /** Executes one token and returns logits owned until the next forward operation. */
  float[] forwardTransient(int token, int position) {
    return forwardTransient(defaultSession, token, position);
  }

  /** Evaluates a prompt while projecting vocabulary logits only for its last token. */
  float[] prefill(int[] tokens, int startPosition) {
    return prefill(defaultSession, tokens, startPosition);
  }

  /** Opens independent sequence state backed by the same weights and expert cache. */
  Session openSession() {
    return new Session(this, Gemma4KvCache.create(config, defaultSession.cache.maxSeqLen(), 1));
  }

  /** Returns the largest independent-session batch accepted by this graph. */
  int maxSessionBatchSize() {
    return MAX_SESSION_BATCH_SIZE;
  }

  /** Returns the prompt batch size selected for this exact loaded tensor topology. */
  int prefillBatchSize() {
    return batchedPrefill ? prefillBatchCapacity : 1;
  }

  /** Executes one token for an independent session and returns stable logits. */
  float[] forward(Session session, int token, int position) {
    return forwardTransient(session, token, position).clone();
  }

  /** Executes one token for an independent session using reusable logits storage. */
  float[] forwardTransient(Session session, int token, int position) {
    return forwardInternal(requireSession(session), token, position, true);
  }

  /** Evaluates a prompt without changing the graph's default sequence. */
  float[] prefill(Session session, int[] tokens, int startPosition) {
    Session sequence = requireSession(session);
    Objects.requireNonNull(tokens, "tokens");
    if (tokens.length == 0) {
      throw new IllegalArgumentException("tokens must not be empty");
    }
    if (startPosition != sequence.nextPosition) {
      throw new IllegalArgumentException(
          "position must be sequential: expected "
              + sequence.nextPosition
              + ", got "
              + startPosition);
    }
    if (tokens.length > sequence.cache.maxSeqLen() - startPosition) {
      throw new IllegalArgumentException(
          "prompt exceeds context length: " + (startPosition + (long) tokens.length));
    }
    if (batchedPrefill && tokens.length > 1) {
      return prefillBatched(sequence, tokens, startPosition);
    }
    for (int index = 0; index < tokens.length - 1; index++) {
      forwardInternal(sequence, tokens[index], startPosition + index, false);
    }
    return forwardInternal(
            sequence, tokens[tokens.length - 1], startPosition + tokens.length - 1, true)
        .clone();
  }

  /** Executes one decode token for each independent session and returns stable logits. */
  LogitBatch forwardBatch(Session[] sessions, int[] tokens) {
    return forwardBatchTransient(sessions, tokens).snapshot();
  }

  /** Executes one decode token for each independent session using reusable batch storage. */
  LogitBatch forwardBatchTransient(Session[] sessions, int[] tokens) {
    validateSessionBatch(sessions, tokens);
    int vocabSize = config.vocabSize();
    int resultLength = Math.multiplyExact(sessions.length, vocabSize);
    if (sessionBatchLogits.length < resultLength) {
      sessionBatchLogits = new float[resultLength];
    }
    for (int index = 0; index < sessions.length; index++) {
      Session session = sessions[index];
      float[] row = forwardInternal(session, tokens[index], session.nextPosition, true);
      System.arraycopy(row, 0, sessionBatchLogits, index * vocabSize, vocabSize);
    }
    return new LogitBatch(sessions.length, vocabSize, sessionBatchLogits);
  }

  /** Consumes a speculative continuation and returns stable logits for every token. */
  LogitBatch verify(int[] tokens, int startPosition) {
    return verifyTransient(tokens, startPosition).snapshot();
  }

  /** Consumes a speculative continuation using reusable batch storage. */
  LogitBatch verifyTransient(int[] tokens, int startPosition) {
    Objects.requireNonNull(tokens, "tokens");
    if (tokens.length == 0) {
      throw new IllegalArgumentException("tokens must not be empty");
    }
    if (startPosition != defaultSession.nextPosition) {
      throw new IllegalArgumentException(
          "position must be sequential: expected "
              + defaultSession.nextPosition
              + ", got "
              + startPosition);
    }
    if (tokens.length > defaultSession.cache.maxSeqLen() - startPosition) {
      throw new IllegalArgumentException(
          "tokens exceed context length: " + (startPosition + (long) tokens.length));
    }
    int vocabSize = config.vocabSize();
    int resultLength = Math.multiplyExact(tokens.length, vocabSize);
    if (verificationLogits.length < resultLength) {
      verificationLogits = new float[resultLength];
    }
    for (int index = 0; index < tokens.length; index++) {
      float[] row = forwardInternal(defaultSession, tokens[index], startPosition + index, true);
      System.arraycopy(row, 0, verificationLogits, index * vocabSize, vocabSize);
    }
    return new LogitBatch(tokens.length, vocabSize, verificationLogits);
  }

  /** Returns the next absolute sequence position. */
  int checkpoint() {
    return defaultSession.nextPosition;
  }

  /** Discards sequence state at and after a prior checkpoint. */
  void rewind(int checkpoint) {
    rewind(defaultSession, checkpoint);
  }

  /** Discards independent-session state at and after a prior checkpoint. */
  void rewind(Session session, int checkpoint) {
    Session sequence = requireSession(session);
    if (checkpoint < 0 || checkpoint > sequence.nextPosition) {
      throw new IllegalArgumentException(
          "checkpoint must be between 0 and " + sequence.nextPosition + ": " + checkpoint);
    }
    sequence.cache.discardFrom(checkpoint);
    sequence.nextPosition = checkpoint;
  }

  /** Clears sequence state while retaining allocated scratch and KV storage. */
  void reset() {
    reset(defaultSession);
  }

  /** Clears one independent session without changing any other sequence. */
  void reset(Session session) {
    Session sequence = requireSession(session);
    sequence.cache.clear();
    sequence.nextPosition = 0;
  }

  private float[] forwardInternal(
      Session sequence, int token, int position, boolean computeLogits) {
    if (position != sequence.nextPosition) {
      throw new IllegalArgumentException(
          "position must be sequential: expected " + sequence.nextPosition + ", got " + position);
    }
    if (position >= sequence.cache.maxSeqLen()) {
      throw new IllegalArgumentException("position exceeds context length: " + position);
    }

    weights.embedToken(token, state);
    multiply(state, config.embeddingScale());
    try {
      for (int layer = 0; layer < config.numLayers(); layer++) {
        executeLayer(sequence.cache, layer, position);
      }
      sequence.nextPosition++;
    } catch (RuntimeException | Error failure) {
      sequence.cache.discardFrom(position);
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

  private float[] prefillBatched(Session sequence, int[] tokens, int startPosition) {
    int tokenOffset = 0;
    while (tokenOffset < tokens.length) {
      int batchSize = Math.min(prefillBatchCapacity, tokens.length - tokenOffset);
      if (batchSize == 1) {
        return forwardInternal(
                sequence, tokens[tokenOffset], Math.addExact(startPosition, tokenOffset), true)
            .clone();
      }
      int batchStartPosition = Math.addExact(startPosition, tokenOffset);
      boolean computeLogits = tokenOffset + batchSize == tokens.length;
      try {
        executePrefillBatch(
            sequence.cache, tokens, tokenOffset, batchSize, batchStartPosition, computeLogits);
        sequence.nextPosition += batchSize;
      } catch (RuntimeException | Error failure) {
        sequence.cache.discardFrom(batchStartPosition);
        throw failure;
      }
      tokenOffset += batchSize;
    }
    return logits.clone();
  }

  private void executePrefillBatch(
      LayeredKvCache sequenceCache,
      int[] tokens,
      int tokenOffset,
      int batchSize,
      int startPosition,
      boolean computeLogits) {
    int dim = config.embeddingDim();
    for (int batch = 0; batch < batchSize; batch++) {
      weights.embedToken(tokens[tokenOffset + batch], state);
      multiply(state, config.embeddingScale());
      System.arraycopy(state, 0, batchState, batch * dim, dim);
    }

    for (int layer = 0; layer < config.numLayers(); layer++) {
      executeLayerBatch(sequenceCache, layer, startPosition, batchSize);
    }
    if (!computeLogits) {
      return;
    }
    int finalStateOffset = (batchSize - 1) * dim;
    TensorOps.rmsNorm(
        normalized,
        0,
        batchState,
        finalStateOffset,
        weights.outputNorm(),
        dim,
        config.rmsNormEps());
    project(
        weights.tokenEmbedding(),
        weights.tokenEmbeddingType(),
        config.vocabSize(),
        dim,
        normalized,
        logits);
    Gemma4Math.softcap(logits, config.finalLogitSoftcap());
  }

  private void executeLayerBatch(
      LayeredKvCache sequenceCache, int layer, int startPosition, int batchSize) {
    Gemma4Weights.LayerWeights layerWeights = weights.layer(layer);
    int dim = config.embeddingDim();
    int headDim = config.headDim(layer);
    int queryDim = config.queryDim(layer);
    int keyDim = config.keyDim(layer);
    int valueDim = config.valueDim(layer);
    int attentionOutputDim = config.attentionOutputDim(layer);

    normalizeBatch(batchNormalized, batchState, batchSize, dim, layerWeights.attentionNorm());
    if (layerWeights.valueProjection() == null) {
      projectDualBatched(
          layerWeights.queryProjection(),
          batchQuery,
          layerWeights.keyProjection(),
          batchKey,
          batchNormalized,
          batchSize);
      for (int batch = 0; batch < batchSize; batch++) {
        System.arraycopy(batchKey, batch * keyDim, batchValue, batch * valueDim, valueDim);
      }
    } else {
      projectTripleBatched(
          layerWeights.queryProjection(),
          batchQuery,
          layerWeights.keyProjection(),
          batchKey,
          layerWeights.valueProjection(),
          batchValue,
          batchNormalized,
          batchSize);
    }

    RotaryTable rotary = rotaryTables[layer];
    for (int batch = 0; batch < batchSize; batch++) {
      int queryOffset = batch * queryDim;
      int keyOffset = batch * keyDim;
      int valueOffset = batch * valueDim;
      for (int head = 0; head < config.numHeads(); head++) {
        int offset = queryOffset + head * headDim;
        TensorOps.rmsNorm(
            batchQuery,
            offset,
            batchQuery,
            offset,
            layerWeights.queryNorm(),
            headDim,
            config.rmsNormEps());
      }
      for (int head = 0; head < config.numKvHeads(layer); head++) {
        int keyHeadOffset = keyOffset + head * headDim;
        int valueHeadOffset = valueOffset + head * headDim;
        TensorOps.rmsNorm(
            batchKey,
            keyHeadOffset,
            batchKey,
            keyHeadOffset,
            layerWeights.keyNorm(),
            headDim,
            config.rmsNormEps());
        Gemma4Math.normalizeWithoutWeight(
            batchValue, valueHeadOffset, batchValue, valueHeadOffset, headDim, config.rmsNormEps());
      }

      int position = startPosition + batch;
      rotary.prepare(position);
      for (int head = 0; head < config.numHeads(); head++) {
        rotary.apply(batchQuery, queryOffset + head * headDim, true);
      }
      for (int head = 0; head < config.numKvHeads(layer); head++) {
        rotary.apply(batchKey, keyOffset + head * headDim, true);
      }

      sequenceCache.store(layer, position, batchKey, keyOffset, batchValue, valueOffset);
      computeAttention(
          sequenceCache,
          layer,
          position,
          batchQuery,
          queryOffset,
          queryDim,
          keyDim,
          valueDim,
          batchAttentionOutput,
          batch * attentionOutputDim);
    }

    projectBatched(
        layerWeights.attentionOutputProjection(), batchAttentionOutput, batchSize, batchProjected);
    normalizeBatchInPlace(batchProjected, batchSize, dim, layerWeights.attentionPostNorm());
    addBatch(batchState, batchProjected, batchSize * dim);

    executeSharedFfnBatch(layerWeights, batchSize);
    executeRoutedFfnBatch(layer, layerWeights, batchSize);
    int activeElements = batchSize * dim;
    for (int index = 0; index < activeElements; index++) {
      batchCombinedOutput[index] = batchSharedOutput[index] + batchRoutedOutput[index];
    }
    normalizeBatchInPlace(batchCombinedOutput, batchSize, dim, layerWeights.combinedFfnPostNorm());
    addBatch(batchState, batchCombinedOutput, activeElements);
    scaleBatch(batchState, activeElements, layerWeights.layerOutputScale());
  }

  private void executeSharedFfnBatch(Gemma4Weights.LayerWeights layer, int batchSize) {
    int dim = config.embeddingDim();
    int hidden = config.sharedHiddenDim();
    normalizeBatch(batchNormalized, batchState, batchSize, dim, layer.sharedFfnNorm());
    projectDualBatched(
        layer.sharedGateProjection(),
        batchSharedGate,
        layer.sharedUpProjection(),
        batchSharedUp,
        batchNormalized,
        batchSize);
    for (int batch = 0; batch < batchSize; batch++) {
      int hiddenOffset = batch * hidden;
      TensorOps.geluGlu(
          batchSharedActivation,
          hiddenOffset,
          batchSharedGate,
          hiddenOffset,
          batchSharedUp,
          hiddenOffset,
          hidden);
    }
    projectBatched(
        layer.sharedDownProjection(), batchSharedActivation, batchSize, batchSharedOutput);
    normalizeBatchInPlace(batchSharedOutput, batchSize, dim, layer.sharedFfnPostNorm());
  }

  private void executeRoutedFfnBatch(
      int layerIndex, Gemma4Weights.LayerWeights layer, int batchSize) {
    int dim = config.embeddingDim();
    int topK = config.numExpertsUsed();
    normalizeBatch(batchNormalized, batchState, batchSize, dim, layer.routedFfnNorm());
    for (int batch = 0; batch < batchSize; batch++) {
      int stateOffset = batch * dim;
      Gemma4Math.normalizeRouterInput(
          batchRouterInput,
          stateOffset,
          batchState,
          stateOffset,
          layer.routerScale(),
          dim,
          config.rmsNormEps());
    }
    projectBatched(layer.routerProjection(), batchRouterInput, batchSize, batchRouterLogits);
    for (int batch = 0; batch < batchSize; batch++) {
      System.arraycopy(
          batchRouterLogits, batch * config.numExperts(), routerLogits, 0, config.numExperts());
      Gemma4Math.selectExperts(routerLogits, layer.expertScales(), selectedExperts, routingWeights);
      System.arraycopy(selectedExperts, 0, batchSelectedExperts, batch * topK, topK);
      System.arraycopy(routingWeights, 0, batchRoutingWeights, batch * topK, topK);
    }
    Arrays.fill(batchRoutedOutput, 0, batchSize * dim, 0.0f);

    try {
      int activeExperts =
          orderExpertsByLastRoute(
              batchSelectedExperts,
              batchSize,
              topK,
              config.numExperts(),
              prefillExpertOrder,
              prefillExpertLastRoute);
      gatherExpertRoutes(activeExperts, batchSize, topK);
      executeExpertPrefillGroups(layerIndex, activeExperts, dim);
    } catch (IOException exception) {
      throw new UncheckedIOException(
          "Failed to load Gemma 4 experts in layer " + layerIndex, exception);
    }
    normalizeBatchInPlace(batchRoutedOutput, batchSize, dim, layer.routedFfnPostNorm());
  }

  private void gatherExpertRoutes(int activeExperts, int batchSize, int topK) {
    int routeCount = 0;
    for (int expertIndex = 0; expertIndex < activeExperts; expertIndex++) {
      int expert = prefillExpertOrder[expertIndex];
      int routeStart = routeCount;
      for (int batch = 0; batch < batchSize; batch++) {
        int routeBase = batch * topK;
        for (int rank = 0; rank < topK; rank++) {
          if (batchSelectedExperts[routeBase + rank] == expert) {
            expertRouteTokens[routeCount] = batch;
            expertRouteRanks[routeCount] = rank;
            routeCount++;
          }
        }
      }
      prefillExpertRouteStarts[expertIndex] = routeStart;
      prefillExpertRouteCounts[expertIndex] = routeCount - routeStart;
    }
  }

  private void executeExpertPrefillGroups(int layerIndex, int activeExperts, int dim)
      throws IOException {
    int maximumGroup =
        Math.min(MAX_INDEPENDENT_ROUTE_GROUP_SIZE, experts.concurrentLeasesPerLayer());
    for (int expertStart = 0; expertStart < activeExperts; expertStart += maximumGroup) {
      int groupSize = Math.min(maximumGroup, activeExperts - expertStart);
      executeExpertPrefillGroup(layerIndex, expertStart, groupSize, dim);
    }
  }

  private void executeExpertPrefillGroup(int layerIndex, int expertStart, int groupSize, int dim)
      throws IOException {
    int hidden = config.expertHiddenDim();
    int acquired = 0;
    try {
      for (; acquired < groupSize; acquired++) {
        int expertIndex = expertStart + acquired;
        int expert = prefillExpertOrder[expertIndex];
        int routeStart = prefillExpertRouteStarts[expertIndex];
        int batchSize = prefillExpertRouteCounts[expertIndex];
        ExpertWeights source = weights.expertLayout().layer(layerIndex).expert(expert);
        Gemma4Experts.Lease lease = experts.acquire(layerIndex, expert, batchSize);
        expertLeases[acquired] = lease;
        groupedExpertWeights[acquired] = lease.gateUp();
        groupedExpertTypes[acquired] = source.gateUp().type();
        groupedExpertRows[acquired] = 2 * hidden;
        groupedExpertBatchSizes[acquired] = batchSize;
        for (int batch = 0; batch < batchSize; batch++) {
          int token = expertRouteTokens[routeStart + batch];
          System.arraycopy(
              batchNormalized, token * dim, groupedExpertInputs[acquired], batch * dim, dim);
        }
      }
      projectRaggedIndependentGroup(groupedExpertGateUps, groupedExpertInputs, groupSize, dim);

      for (int groupIndex = 0; groupIndex < groupSize; groupIndex++) {
        int expertIndex = expertStart + groupIndex;
        int expert = prefillExpertOrder[expertIndex];
        int batchSize = groupedExpertBatchSizes[groupIndex];
        for (int batch = 0; batch < batchSize; batch++) {
          int gateOffset = batch * 2 * hidden;
          TensorOps.geluGlu(
              groupedExpertActivations[groupIndex],
              batch * hidden,
              groupedExpertGateUps[groupIndex],
              gateOffset,
              groupedExpertGateUps[groupIndex],
              gateOffset + hidden,
              hidden);
        }
        ExpertWeights source = weights.expertLayout().layer(layerIndex).expert(expert);
        groupedExpertWeights[groupIndex] = expertLeases[groupIndex].down();
        groupedExpertTypes[groupIndex] = source.down().type();
        groupedExpertRows[groupIndex] = dim;
      }
      projectRaggedIndependentGroup(
          groupedExpertOutputs, groupedExpertActivations, groupSize, hidden);

      int topK = config.numExpertsUsed();
      for (int groupIndex = 0; groupIndex < groupSize; groupIndex++) {
        int expertIndex = expertStart + groupIndex;
        int routeStart = prefillExpertRouteStarts[expertIndex];
        int batchSize = groupedExpertBatchSizes[groupIndex];
        for (int batch = 0; batch < batchSize; batch++) {
          int route = routeStart + batch;
          int token = expertRouteTokens[route];
          int rank = expertRouteRanks[route];
          float scale = batchRoutingWeights[token * topK + rank];
          addScaled(
              batchRoutedOutput,
              token * dim,
              groupedExpertOutputs[groupIndex],
              batch * dim,
              scale,
              dim);
        }
      }
    } finally {
      for (int groupIndex = acquired - 1; groupIndex >= 0; groupIndex--) {
        expertLeases[groupIndex].close();
        expertLeases[groupIndex] = null;
      }
    }
  }

  private void projectRaggedIndependentGroup(
      float[][] outputs, float[][] inputs, int groupSize, int columns) {
    if (batchedMatrixKernel.isRaggedIndependentEligible(
        groupedExpertTypes, groupedExpertRows, groupedExpertBatchSizes, groupSize, columns)) {
      batchedMatrixKernel.multiplyRaggedIndependent(
          outputs,
          groupedExpertWeights,
          groupedExpertTypes,
          groupedExpertRows,
          groupedExpertBatchSizes,
          groupSize,
          inputs,
          columns);
      return;
    }
    for (int groupIndex = 0; groupIndex < groupSize; groupIndex++) {
      projectBatched(
          groupedExpertWeights[groupIndex],
          groupedExpertTypes[groupIndex],
          groupedExpertRows[groupIndex],
          columns,
          inputs[groupIndex],
          groupedExpertBatchSizes[groupIndex],
          outputs[groupIndex]);
    }
  }

  private void executeLayer(LayeredKvCache sequenceCache, int layer, int position) {
    Gemma4Weights.LayerWeights layerWeights = weights.layer(layer);
    int dim = config.embeddingDim();
    int headDim = config.headDim(layer);
    int queryDim = config.queryDim(layer);
    int keyDim = config.keyDim(layer);
    int valueDim = config.valueDim(layer);
    float[] attentionOutput = attentionOutputs[layer];

    TensorOps.rmsNorm(normalized, state, layerWeights.attentionNorm(), dim, config.rmsNormEps());
    if (layerWeights.valueProjection() == null) {
      projectDual(
          layerWeights.queryProjection(), query, layerWeights.keyProjection(), key, normalized);
      System.arraycopy(key, 0, value, 0, valueDim);
    } else {
      projectTriple(
          layerWeights.queryProjection(),
          query,
          layerWeights.keyProjection(),
          key,
          layerWeights.valueProjection(),
          value,
          normalized);
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

    sequenceCache.store(layer, position, key, 0, value, 0);
    computeAttention(sequenceCache, layer, position, queryDim, keyDim, valueDim, attentionOutput);
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
    projectDual(
        layer.sharedGateProjection(),
        sharedGate,
        layer.sharedUpProjection(),
        sharedUp,
        sharedInput);
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

    int groupCapacity = Math.min(selectedExperts.length, experts.concurrentLeasesPerLayer());
    try {
      for (int groupStart = 0; groupStart < selectedExperts.length; groupStart += groupCapacity) {
        int groupSize = Math.min(groupCapacity, selectedExperts.length - groupStart);
        executeExpertGroup(layerIndex, groupStart, groupSize, dim, hidden);
      }
    } catch (IOException exception) {
      throw new UncheckedIOException(
          "Failed to load Gemma 4 experts in layer " + layerIndex, exception);
    }
    TensorOps.rmsNorm(
        routedOutput, routedOutput, layer.routedFfnPostNorm(), dim, config.rmsNormEps());
  }

  private void executeExpertGroup(
      int layerIndex, int groupStart, int groupSize, int dim, int hidden) throws IOException {
    int acquired = 0;
    try {
      for (; acquired < groupSize; acquired++) {
        int expert = selectedExperts[groupStart + acquired];
        ExpertWeights source = weights.expertLayout().layer(layerIndex).expert(expert);
        Gemma4Experts.Lease lease = experts.acquire(layerIndex, expert);
        expertLeases[acquired] = lease;
        groupedExpertWeights[acquired] = lease.gateUp();
        groupedExpertTypes[acquired] = source.gateUp().type();
        groupedExpertRows[acquired] = 2 * hidden;
      }

      if (batchedMatrixKernel.isGroupedEligible(
          groupedExpertTypes, groupedExpertRows, groupSize, 1, dim)) {
        batchedMatrixKernel.multiplyGrouped(
            expertGateUps,
            groupedExpertWeights,
            groupedExpertTypes,
            groupedExpertRows,
            groupSize,
            routedInput,
            1,
            dim);
      } else {
        for (int groupIndex = 0; groupIndex < groupSize; groupIndex++) {
          project(
              groupedExpertWeights[groupIndex],
              groupedExpertTypes[groupIndex],
              groupedExpertRows[groupIndex],
              dim,
              routedInput,
              expertGateUps[groupIndex]);
        }
      }

      for (int groupIndex = 0; groupIndex < groupSize; groupIndex++) {
        int expert = selectedExperts[groupStart + groupIndex];
        ExpertWeights source = weights.expertLayout().layer(layerIndex).expert(expert);
        TensorOps.geluGlu(
            expertActivations[groupIndex],
            0,
            expertGateUps[groupIndex],
            0,
            expertGateUps[groupIndex],
            hidden,
            hidden);
        groupedExpertWeights[groupIndex] = expertLeases[groupIndex].down();
        groupedExpertTypes[groupIndex] = source.down().type();
        groupedExpertRows[groupIndex] = dim;
      }

      if (batchedMatrixKernel.isIndependentEligible(
          groupedExpertTypes, groupedExpertRows, groupSize, 1, hidden)) {
        batchedMatrixKernel.multiplyIndependent(
            expertOutputs,
            groupedExpertWeights,
            groupedExpertTypes,
            groupedExpertRows,
            groupSize,
            expertActivations,
            1,
            hidden);
      } else {
        for (int groupIndex = 0; groupIndex < groupSize; groupIndex++) {
          project(
              groupedExpertWeights[groupIndex],
              groupedExpertTypes[groupIndex],
              groupedExpertRows[groupIndex],
              hidden,
              expertActivations[groupIndex],
              expertOutputs[groupIndex]);
        }
      }

      for (int groupIndex = 0; groupIndex < groupSize; groupIndex++) {
        int rank = groupStart + groupIndex;
        addScaled(routedOutput, expertOutputs[groupIndex], routingWeights[rank], dim);
      }
    } finally {
      for (int groupIndex = acquired - 1; groupIndex >= 0; groupIndex--) {
        expertLeases[groupIndex].close();
        expertLeases[groupIndex] = null;
      }
    }
  }

  private void computeAttention(
      LayeredKvCache sequenceCache,
      int layer,
      int position,
      int queryDim,
      int keyDim,
      int valueDim,
      float[] attentionOutput) {
    computeAttention(
        sequenceCache, layer, position, query, 0, queryDim, keyDim, valueDim, attentionOutput, 0);
  }

  private void computeAttention(
      LayeredKvCache sequenceCache,
      int layer,
      int position,
      float[] querySource,
      int queryOffset,
      int queryDim,
      int keyDim,
      int valueDim,
      float[] attentionOutput,
      int attentionOutputOffset) {
    int headDim = config.headDim(layer);
    int groupSize = config.numHeads() / config.numKvHeads(layer);
    int fromPosition = config.attentionStartPosition(layer, position);
    AttentionView view = sequenceCache.attentionView(layer, fromPosition, position + 1);
    float[] keys = sequenceCache.keyBuffer(layer);
    float[] values = sequenceCache.valueBuffer(layer);
    Arrays.fill(
        attentionOutput,
        attentionOutputOffset,
        attentionOutputOffset + config.attentionOutputDim(layer),
        0.0f);

    for (int queryHead = 0; queryHead < queryDim / headDim; queryHead++) {
      int kvHead = queryHead / groupSize;
      int scoreCount = 0;
      for (int spanIndex = 0; spanIndex < view.spanCount(); spanIndex++) {
        AttentionSpan span = view.span(spanIndex);
        for (int row = 0; row < span.positionCount(); row++) {
          int keyOffset = span.keyOffset() + row * keyDim + kvHead * headDim;
          attentionScores[scoreCount++] =
              VectorUtil.dotProduct(
                      querySource, queryOffset + queryHead * headDim, keys, keyOffset, headDim)
                  * config.attentionScale();
        }
      }
      TensorOps.softmax(attentionScores, 0, scoreCount);

      int score = 0;
      int outputOffset = attentionOutputOffset + queryHead * headDim;
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

  private void projectDual(
      Gemma4Weights.Matrix first,
      float[] firstOutput,
      Gemma4Weights.Matrix second,
      float[] secondOutput,
      float[] input) {
    int columns = first.columns();
    if (columns != second.columns()) {
      throw new IllegalArgumentException("grouped projections must have equal input dimensions");
    }
    if (batchedMatrixKernel.isDualEligible(
        first.type(), first.rows(), second.type(), second.rows(), 1, columns)) {
      batchedMatrixKernel.multiplyDual(
          firstOutput,
          first.data(),
          first.type(),
          first.rows(),
          secondOutput,
          second.data(),
          second.type(),
          second.rows(),
          input,
          1,
          columns);
      return;
    }
    TensorOps.ggufDualMatmul(
        firstOutput,
        first.data(),
        first.type(),
        first.rows(),
        secondOutput,
        second.data(),
        second.type(),
        second.rows(),
        input,
        columns,
        quantizedActivation,
        quantizedActivationScales,
        quantizedActivationZeroPointCorrections,
        quantizedActivationSums,
        GgufQ4Kernel.WIDENED);
  }

  private void projectTriple(
      Gemma4Weights.Matrix first,
      float[] firstOutput,
      Gemma4Weights.Matrix second,
      float[] secondOutput,
      Gemma4Weights.Matrix third,
      float[] thirdOutput,
      float[] input) {
    int columns = first.columns();
    if (columns != second.columns() || columns != third.columns()) {
      throw new IllegalArgumentException("grouped projections must have equal input dimensions");
    }
    if (batchedMatrixKernel.isTripleEligible(
        first.type(),
        first.rows(),
        second.type(),
        second.rows(),
        third.type(),
        third.rows(),
        1,
        columns)) {
      batchedMatrixKernel.multiplyTriple(
          firstOutput,
          first.data(),
          first.type(),
          first.rows(),
          secondOutput,
          second.data(),
          second.type(),
          second.rows(),
          thirdOutput,
          third.data(),
          third.type(),
          third.rows(),
          input,
          1,
          columns);
      return;
    }
    TensorOps.ggufTripleMatmul(
        firstOutput,
        first.data(),
        first.type(),
        first.rows(),
        secondOutput,
        second.data(),
        second.type(),
        second.rows(),
        thirdOutput,
        third.data(),
        third.type(),
        third.rows(),
        input,
        columns,
        quantizedActivation,
        quantizedActivationScales,
        quantizedActivationZeroPointCorrections,
        quantizedActivationSums,
        GgufQ4Kernel.WIDENED);
  }

  private void project(
      MemorySegment matrix,
      GgufTensorType type,
      int rows,
      int columns,
      float[] input,
      float[] output) {
    if (batchedMatrixKernel.isEligible(type, 1, rows, columns)) {
      batchedMatrixKernel.multiply(output, input, matrix, type, 1, rows, columns);
      return;
    }
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

  private void projectBatched(
      Gemma4Weights.Matrix matrix, float[] input, int batchSize, float[] output) {
    projectBatched(
        matrix.data(), matrix.type(), matrix.rows(), matrix.columns(), input, batchSize, output);
  }

  private void projectBatched(
      MemorySegment matrix,
      GgufTensorType type,
      int rows,
      int columns,
      float[] input,
      int batchSize,
      float[] output) {
    if (batchedMatrixKernel.isEligible(type, batchSize, rows, columns)) {
      batchedMatrixKernel.multiply(output, input, matrix, type, batchSize, rows, columns);
      return;
    }
    if (type == GgufTensorType.F32) {
      F32Scratch scratch = findF32Scratch(rows, columns);
      if (scratch == null) {
        throw new IllegalStateException("missing F32 prefill scratch for " + rows + "x" + columns);
      }
      for (int batch = 0; batch < batchSize; batch++) {
        System.arraycopy(input, batch * columns, scratch.input(), 0, columns);
        TensorOps.ggufMatmul(scratch.output(), scratch.input(), matrix, type, rows, columns);
        System.arraycopy(scratch.output(), 0, output, batch * rows, rows);
      }
      return;
    }
    throw new IllegalStateException(
        "loaded Gemma 4 batched-prefill kernel rejected "
            + type
            + " projection "
            + rows
            + "x"
            + columns);
  }

  private void projectDualBatched(
      Gemma4Weights.Matrix first,
      float[] firstOutput,
      Gemma4Weights.Matrix second,
      float[] secondOutput,
      float[] input,
      int batchSize) {
    int columns = first.columns();
    if (batchedMatrixKernel.isDualEligible(
        first.type(), first.rows(), second.type(), second.rows(), batchSize, columns)) {
      batchedMatrixKernel.multiplyDual(
          firstOutput,
          first.data(),
          first.type(),
          first.rows(),
          secondOutput,
          second.data(),
          second.type(),
          second.rows(),
          input,
          batchSize,
          columns);
      return;
    }
    projectBatched(first, input, batchSize, firstOutput);
    projectBatched(second, input, batchSize, secondOutput);
  }

  private void projectTripleBatched(
      Gemma4Weights.Matrix first,
      float[] firstOutput,
      Gemma4Weights.Matrix second,
      float[] secondOutput,
      Gemma4Weights.Matrix third,
      float[] thirdOutput,
      float[] input,
      int batchSize) {
    int columns = first.columns();
    if (batchedMatrixKernel.isTripleEligible(
        first.type(),
        first.rows(),
        second.type(),
        second.rows(),
        third.type(),
        third.rows(),
        batchSize,
        columns)) {
      batchedMatrixKernel.multiplyTriple(
          firstOutput,
          first.data(),
          first.type(),
          first.rows(),
          secondOutput,
          second.data(),
          second.type(),
          second.rows(),
          thirdOutput,
          third.data(),
          third.type(),
          third.rows(),
          input,
          batchSize,
          columns);
      return;
    }
    projectBatched(first, input, batchSize, firstOutput);
    projectBatched(second, input, batchSize, secondOutput);
    projectBatched(third, input, batchSize, thirdOutput);
  }

  private void normalizeBatch(
      float[] output, float[] input, int batchSize, int width, float[] normalizationWeight) {
    for (int batch = 0; batch < batchSize; batch++) {
      int offset = batch * width;
      TensorOps.rmsNorm(
          output, offset, input, offset, normalizationWeight, width, config.rmsNormEps());
    }
  }

  private void normalizeBatchInPlace(
      float[] values, int batchSize, int width, float[] normalizationWeight) {
    normalizeBatch(values, values, batchSize, width, normalizationWeight);
  }

  private static boolean supportsBatchedPrefill(
      Gemma4Config config, Gemma4Weights weights, GgufBatchedMatrixKernel batchedMatrixKernel) {
    for (int layer = 0; layer < config.numLayers(); layer++) {
      Gemma4Weights.LayerWeights layerWeights = weights.layer(layer);
      if (!isBatchedEligible(layerWeights.queryProjection(), batchedMatrixKernel)
          || !isBatchedEligible(layerWeights.keyProjection(), batchedMatrixKernel)
          || (layerWeights.valueProjection() != null
              && !isBatchedEligible(layerWeights.valueProjection(), batchedMatrixKernel))
          || !isBatchedEligible(layerWeights.attentionOutputProjection(), batchedMatrixKernel)
          || !isBatchedEligible(layerWeights.sharedGateProjection(), batchedMatrixKernel)
          || !isBatchedEligible(layerWeights.sharedUpProjection(), batchedMatrixKernel)
          || !isBatchedEligible(layerWeights.sharedDownProjection(), batchedMatrixKernel)
          || !isBatchedEligible(layerWeights.routerProjection(), batchedMatrixKernel)) {
        return false;
      }
      ExpertWeights expert = weights.expertLayout().layer(layer).expert(0);
      if (!batchedMatrixKernel.isEligible(
              expert.gateUp().type(), 2, 2 * config.expertHiddenDim(), config.embeddingDim())
          || !batchedMatrixKernel.isEligible(
              expert.down().type(), 2, config.embeddingDim(), config.expertHiddenDim())) {
        return false;
      }
    }
    return true;
  }

  private static boolean isBatchedEligible(
      Gemma4Weights.Matrix matrix, GgufBatchedMatrixKernel batchedMatrixKernel) {
    return matrix.type() == GgufTensorType.F32
        || batchedMatrixKernel.isEligible(matrix.type(), 2, matrix.rows(), matrix.columns());
  }

  static int orderExpertsByLastRoute(
      int[] selectedExperts, int batchSize, int topK, int numExperts, int[] expertOrder) {
    return orderExpertsByLastRoute(
        selectedExperts, batchSize, topK, numExperts, expertOrder, new int[numExperts]);
  }

  private static int orderExpertsByLastRoute(
      int[] selectedExperts,
      int batchSize,
      int topK,
      int numExperts,
      int[] expertOrder,
      int[] lastRoute) {
    Arrays.fill(lastRoute, -1);
    for (int batch = 0; batch < batchSize; batch++) {
      int routeBase = batch * topK;
      for (int rank = 0; rank < topK; rank++) {
        lastRoute[selectedExperts[routeBase + rank]] = batch;
      }
    }

    int activeExperts = 0;
    for (int expert = 0; expert < numExperts; expert++) {
      if (lastRoute[expert] >= 0) {
        int insertion = activeExperts;
        while (insertion > 0
            && compareExpertRecency(expertOrder[insertion - 1], expert, lastRoute) > 0) {
          expertOrder[insertion] = expertOrder[insertion - 1];
          insertion--;
        }
        expertOrder[insertion] = expert;
        activeExperts++;
      }
    }
    return activeExperts;
  }

  private static int compareExpertRecency(int first, int second, int[] lastRoute) {
    int recency = Integer.compare(lastRoute[first], lastRoute[second]);
    return recency != 0 ? recency : Integer.compare(first, second);
  }

  private static F32Scratch[] createF32Scratch(Gemma4Config config, Gemma4Weights weights) {
    List<F32Scratch> scratch = new ArrayList<>();
    for (int layer = 0; layer < config.numLayers(); layer++) {
      Gemma4Weights.LayerWeights layerWeights = weights.layer(layer);
      registerF32Scratch(scratch, layerWeights.queryProjection());
      registerF32Scratch(scratch, layerWeights.keyProjection());
      if (layerWeights.valueProjection() != null) {
        registerF32Scratch(scratch, layerWeights.valueProjection());
      }
      registerF32Scratch(scratch, layerWeights.attentionOutputProjection());
      registerF32Scratch(scratch, layerWeights.sharedGateProjection());
      registerF32Scratch(scratch, layerWeights.sharedUpProjection());
      registerF32Scratch(scratch, layerWeights.sharedDownProjection());
      registerF32Scratch(scratch, layerWeights.routerProjection());
    }
    return scratch.toArray(F32Scratch[]::new);
  }

  private static void registerF32Scratch(List<F32Scratch> scratch, Gemma4Weights.Matrix matrix) {
    if (matrix.type() != GgufTensorType.F32) {
      return;
    }
    for (F32Scratch candidate : scratch) {
      if (candidate.rows() == matrix.rows() && candidate.columns() == matrix.columns()) {
        return;
      }
    }
    scratch.add(
        new F32Scratch(
            matrix.rows(),
            matrix.columns(),
            new float[matrix.columns()],
            new float[matrix.rows()]));
  }

  private F32Scratch findF32Scratch(int rows, int columns) {
    for (F32Scratch candidate : batchF32Scratch) {
      if (candidate.rows() == rows && candidate.columns() == columns) {
        return candidate;
      }
    }
    return null;
  }

  private void validateSessionBatch(Session[] sessions, int[] tokens) {
    Objects.requireNonNull(sessions, "sessions");
    Objects.requireNonNull(tokens, "tokens");
    if (sessions.length == 0) {
      throw new IllegalArgumentException("sessions must not be empty");
    }
    if (tokens.length != sessions.length) {
      throw new IllegalArgumentException(
          "tokens.length must equal sessions.length: " + tokens.length + " != " + sessions.length);
    }
    if (sessions.length > maxSessionBatchSize()) {
      throw new IllegalArgumentException(
          "session batch exceeds capacity " + maxSessionBatchSize() + ": " + sessions.length);
    }
    for (int index = 0; index < sessions.length; index++) {
      Session session = requireSession(sessions[index]);
      if (session.nextPosition >= session.cache.maxSeqLen()) {
        throw new IllegalArgumentException(
            "session " + index + " has reached context length " + session.cache.maxSeqLen());
      }
      for (int prior = 0; prior < index; prior++) {
        if (sessions[prior] == session) {
          throw new IllegalArgumentException("sessions must be distinct");
        }
      }
    }
  }

  private Session requireSession(Session session) {
    Objects.requireNonNull(session, "session");
    if (session.owner != this) {
      throw new IllegalArgumentException("session belongs to a different forward pass");
    }
    return session;
  }

  private static float[][] exactAttentionOutputs(Gemma4Config config) {
    float[][] outputs = new float[config.numLayers()][];
    for (int layer = 0; layer < outputs.length; layer++) {
      int length = config.attentionOutputDim(layer);
      for (int prior = 0; prior < layer; prior++) {
        if (outputs[prior].length == length) {
          outputs[layer] = outputs[prior];
          break;
        }
      }
      if (outputs[layer] == null) {
        outputs[layer] = new float[length];
      }
    }
    return outputs;
  }

  private static int maxLength(float[][] values) {
    int maximum = 0;
    for (float[] value : values) {
      maximum = Math.max(maximum, value.length);
    }
    return maximum;
  }

  private static float[] batchBuffer(int batchSize, int width) {
    return new float[Math.multiplyExact(batchSize, width)];
  }

  private static float[][] matrixBuffer(int rows, int columns) {
    float[][] result = new float[rows][];
    for (int row = 0; row < rows; row++) {
      result[row] = new float[columns];
    }
    return result;
  }

  private static void addBatch(float[] target, float[] addition, int length) {
    for (int index = 0; index < length; index++) {
      target[index] += addition[index];
    }
  }

  private static void scaleBatch(float[] values, int length, float scale) {
    for (int index = 0; index < length; index++) {
      values[index] *= scale;
    }
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

  private static void addScaled(
      float[] target,
      int targetOffset,
      float[] addition,
      int additionOffset,
      float scale,
      int length) {
    for (int index = 0; index < length; index++) {
      target[targetOffset + index] += addition[additionOffset + index] * scale;
    }
  }

  private static void multiply(float[] values, float scale) {
    for (int index = 0; index < values.length; index++) {
      values[index] *= scale;
    }
  }
}
