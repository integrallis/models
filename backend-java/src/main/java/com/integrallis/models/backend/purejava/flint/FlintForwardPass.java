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
package com.integrallis.models.backend.purejava.flint;

import com.integrallis.models.backend.purejava.cache.LayeredKvCache;
import com.integrallis.models.backend.purejava.cache.LayeredKvCache.AttentionSpan;
import com.integrallis.models.backend.purejava.cache.LayeredKvCache.AttentionView;
import com.integrallis.models.backend.purejava.cache.LayeredKvCache.LayerSpec;
import com.integrallis.models.backend.purejava.flint.FlintWeights.LayerWeights;
import com.integrallis.models.backend.purejava.flint.FlintWeights.Matrix;
import com.integrallis.models.backend.purejava.ops.PinnedReduction;
import com.integrallis.models.backend.purejava.ops.RotaryTable;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import java.util.Arrays;
import java.util.Objects;

/** Flint decoder with mapped GGUF experts and independent, bounded sliding-window sessions. */
public final class FlintForwardPass {

  /** Mutable sequence state; sessions share immutable mapped weights and are otherwise isolated. */
  public static final class Session {
    private final FlintForwardPass owner;
    private final LayeredKvCache cache;
    private final RotaryTable rope;
    private final float[] gate;
    private final float[] up;
    private final float[] contribution;
    private final float[] x;
    private final float[] xNorm;
    private final float[] query;
    private final float[] key;
    private final float[] value;
    private final float[] attentionOutput;
    private final float[] attentionScores;
    private final float[] projectedAttention;
    private final float[] routerLogits;
    private final int[] selectedExperts;
    private final float[] routingWeights;
    private final float[] moeOutput;
    private final float[] logits;
    private final int[] tokenHistory;
    private int nextPosition;

    private Session(FlintForwardPass owner) {
      this.owner = owner;
      FlintConfig config = owner.config;
      cache = new LayeredKvCache(owner.maxSequenceLength, owner.cacheSpecs.clone());
      rope = new RotaryTable(config.headDim(), config.ropeTheta(), 1.0f);
      int ffnWidth =
          Math.max(
              config.denseHiddenSize(),
              Math.max(config.expertHiddenSize(), config.sharedHiddenSize()));
      gate = new float[ffnWidth];
      up = new float[ffnWidth];
      contribution = new float[config.hiddenSize()];
      x = new float[config.hiddenSize()];
      xNorm = new float[config.hiddenSize()];
      query = new float[config.queryDimension()];
      key = new float[config.keyValueDimension()];
      value = new float[config.keyValueDimension()];
      attentionOutput = new float[config.queryDimension()];
      attentionScores = new float[owner.maxSequenceLength];
      projectedAttention = new float[config.hiddenSize()];
      routerLogits = new float[config.numExperts()];
      selectedExperts = new int[config.expertsPerToken()];
      routingWeights = new float[config.expertsPerToken()];
      moeOutput = new float[config.hiddenSize()];
      logits = new float[config.vocabSize()];
      tokenHistory = new int[owner.maxSequenceLength];
    }

    /** Returns the next absolute sequence position accepted by this session. */
    public int checkpoint() {
      return nextPosition;
    }
  }

  private final FlintConfig config;
  private final FlintWeights weights;
  private final LayerWeights[] layers;
  private final LayerSpec[] cacheSpecs;
  private final float[] outputNorm;
  private final int maxSequenceLength;

  FlintForwardPass(FlintConfig config, FlintWeights weights, int maxSequenceLength) {
    this.config = Objects.requireNonNull(config, "config");
    this.weights = Objects.requireNonNull(weights, "weights");
    if (maxSequenceLength <= 0 || maxSequenceLength > config.maxPosition()) {
      throw new IllegalArgumentException(
          "maxSequenceLength must be between 1 and "
              + config.maxPosition()
              + ": "
              + maxSequenceLength);
    }
    this.maxSequenceLength = maxSequenceLength;
    outputNorm = weights.outputNorm();
    layers = new LayerWeights[config.numLayers()];
    cacheSpecs = new LayerSpec[config.numLayers()];
    for (int layer = 0; layer < layers.length; layer++) {
      layers[layer] = weights.layer(layer);
      cacheSpecs[layer] =
          config.usesSlidingAttention(layer)
              ? LayerSpec.ring(
                  config.keyValueDimension(), config.keyValueDimension(), config.slidingWindow())
              : LayerSpec.linear(config.keyValueDimension(), config.keyValueDimension());
    }
  }

  /** Loads the validated graph; all expert matrices stay in the mapped GGUF. */
  public static FlintForwardPass loadGguf(
      FlintConfig config,
      com.integrallis.models.backend.purejava.gguf.GgufFile file,
      int maxSequenceLength) {
    return new FlintForwardPass(config, FlintWeights.fromGgufFile(file, config), maxSequenceLength);
  }

  /** Returns the validated checkpoint execution configuration. */
  public FlintConfig config() {
    return config;
  }

  /** Opens an independent sequence over the same mapped model weights. */
  public Session openSession() {
    return new Session(this);
  }

  /** Runs one sequential token and returns stable vocabulary logits. */
  public float[] forward(Session session, int token, int position) {
    return forwardTransient(session, token, position).clone();
  }

  /** Runs one sequential token using session-owned logits storage. */
  public float[] forwardTransient(Session session, int token, int position) {
    execute(session, token, position, true);
    return session.logits;
  }

  /** Advances a prompt token without paying for a vocabulary projection that will be discarded. */
  public void advance(Session session, int token, int position) {
    execute(session, token, position, false);
  }

  private void execute(Session session, int token, int position, boolean projectLogits) {
    requireSession(session);
    if (position != session.nextPosition) {
      throw new IllegalArgumentException(
          "position must be sequential: expected " + session.nextPosition + ", got " + position);
    }
    if (position >= maxSequenceLength) {
      throw new IllegalArgumentException("position exceeds the configured context: " + position);
    }
    embedToken(token, session.x);
    session.rope.prepare(position);

    for (int layer = 0; layer < layers.length; layer++) {
      executeLayer(session, layer, position);
    }
    if (projectLogits) {
      TensorOps.rmsNorm(
          session.xNorm, session.x, outputNorm, config.hiddenSize(), config.rmsNormEps());
      multiply(weights.output(), session.xNorm, session.logits);
    }
    session.tokenHistory[position] = token;
    session.nextPosition++;
  }

  /** Discards sequence state at and after {@code checkpoint}. */
  public void rewind(Session session, int checkpoint) {
    requireSession(session);
    if (checkpoint < 0 || checkpoint > session.nextPosition) {
      throw new IllegalArgumentException(
          "checkpoint must be between 0 and " + session.nextPosition + ": " + checkpoint);
    }
    if (checkpoint == session.nextPosition) {
      return;
    }
    if (!retainsPrefixWindow(session, checkpoint)) {
      int[] replay = Arrays.copyOf(session.tokenHistory, checkpoint);
      session.cache.clear();
      session.nextPosition = 0;
      for (int position = 0; position < replay.length; position++) {
        advance(session, replay[position], position);
      }
      return;
    }
    session.cache.discardFrom(checkpoint);
    session.nextPosition = checkpoint;
  }

  private boolean retainsPrefixWindow(Session session, int checkpoint) {
    for (int layer = 0; layer < layers.length; layer++) {
      int first = config.attentionStartPosition(layer, checkpoint);
      for (int position = first; position < checkpoint; position++) {
        if (!session.cache.contains(layer, position)) {
          return false;
        }
      }
    }
    return true;
  }

  /** Clears retained KV state and returns a session to position zero. */
  public void reset(Session session) {
    requireSession(session);
    session.cache.clear();
    session.nextPosition = 0;
  }

  private void executeLayer(Session session, int layer, int position) {
    LayerWeights parameters = layers[layer];
    TensorOps.rmsNorm(
        session.xNorm,
        session.x,
        parameters.attentionNorm(),
        config.hiddenSize(),
        config.rmsNormEps());
    multiply(parameters.query(), session.xNorm, session.query);
    multiply(parameters.key(), session.xNorm, session.key);
    multiply(parameters.value(), session.xNorm, session.value);
    for (int head = 0; head < config.numHeads(); head++) {
      int offset = head * config.headDim();
      TensorOps.rmsNorm(
          session.query,
          offset,
          session.query,
          offset,
          parameters.queryNorm(),
          config.headDim(),
          config.rmsNormEps());
      if (config.usesRope(layer)) session.rope.apply(session.query, offset, true);
    }
    for (int head = 0; head < config.numKvHeads(); head++) {
      int offset = head * config.headDim();
      TensorOps.rmsNorm(
          session.key,
          offset,
          session.key,
          offset,
          parameters.keyNorm(),
          config.headDim(),
          config.rmsNormEps());
      if (config.usesRope(layer)) session.rope.apply(session.key, offset, true);
    }
    session.cache.store(layer, position, session.key, session.value);
    computeAttention(session, parameters, layer, position);
    multiply(parameters.attentionOutput(), session.attentionOutput, session.projectedAttention);
    addResidual(session.x, session.projectedAttention);

    TensorOps.rmsNorm(
        session.xNorm, session.x, parameters.ffnNorm(), config.hiddenSize(), config.rmsNormEps());
    if (parameters.moe() == null) {
      feedForward(session, parameters.ffnGate(), parameters.ffnUp(), parameters.ffnDown());
      addResidual(session.x, session.contribution);
    } else {
      var moe = parameters.moe();
      multiply(moe.router(), session.xNorm, session.routerLogits);
      FlintRouting.select(
          session.routerLogits,
          moe.selectionBias(),
          session.selectedExperts,
          session.routingWeights,
          config.normalizeExpertWeights(),
          config.expertWeightsScale());
      Arrays.fill(session.moeOutput, 0f);
      for (int slot = 0; slot < config.expertsPerToken(); slot++) {
        int expert = session.selectedExperts[slot];
        feedForward(session, moe.gate()[expert], moe.up()[expert], moe.down()[expert]);
        for (int i = 0; i < config.hiddenSize(); i++) {
          session.moeOutput[i] += session.routingWeights[slot] * session.contribution[i];
        }
      }
      feedForward(session, moe.sharedGate(), moe.sharedUp(), moe.sharedDown());
      addResidual(session.moeOutput, session.contribution);
      addResidual(session.x, session.moeOutput);
    }
  }

  private static void feedForward(Session session, Matrix gate, Matrix up, Matrix down) {
    multiply(gate, session.xNorm, session.gate);
    multiply(up, session.xNorm, session.up);
    for (int i = 0; i < gate.rows(); i++) {
      float value = session.gate[i];
      session.gate[i] = (value / (1f + (float) Math.exp(-value))) * session.up[i];
    }
    multiply(down, session.gate, session.contribution);
  }

  private static void multiply(Matrix matrix, float[] input, float[] output) {
    TensorOps.ggufMatmul(
        output, input, matrix.data(), matrix.type(), matrix.rows(), matrix.columns());
  }

  private void computeAttention(Session session, LayerWeights parameters, int layer, int position) {
    Arrays.fill(session.attentionOutput, 0.0f);
    int firstPosition = config.attentionStartPosition(layer, position);
    AttentionView view = session.cache.attentionView(layer, firstPosition, position + 1);
    float[] cachedKeys = session.cache.keyBuffer(layer);
    float[] cachedValues = session.cache.valueBuffer(layer);
    int headsPerKv = config.numHeads() / config.numKvHeads();
    float scale = (float) (1.0 / Math.sqrt(config.headDim()));

    for (int head = 0; head < config.numHeads(); head++) {
      int kvHead = head / headsPerKv;
      int queryOffset = head * config.headDim();
      int score = 0;
      for (int spanIndex = 0; spanIndex < view.spanCount(); spanIndex++) {
        AttentionSpan span = view.span(spanIndex);
        int keyOffset = span.keyOffset() + kvHead * config.headDim();
        for (int row = 0; row < span.positionCount(); row++) {
          session.attentionScores[score++] =
              scale
                  * PinnedReduction.dot(
                      session.query,
                      queryOffset,
                      cachedKeys,
                      keyOffset + row * config.keyValueDimension(),
                      config.headDim());
        }
      }
      TensorOps.softmaxWithZeroValueSink(
          session.attentionScores, 0, view.positionCount(), parameters.sinks()[head]);

      int probability = 0;
      int outputOffset = head * config.headDim();
      for (int spanIndex = 0; spanIndex < view.spanCount(); spanIndex++) {
        AttentionSpan span = view.span(spanIndex);
        int valueOffset = span.valueOffset() + kvHead * config.headDim();
        for (int row = 0; row < span.positionCount(); row++) {
          float weight = session.attentionScores[probability++];
          int rowOffset = valueOffset + row * config.keyValueDimension();
          for (int dimension = 0; dimension < config.headDim(); dimension++) {
            session.attentionOutput[outputOffset + dimension] +=
                weight * cachedValues[rowOffset + dimension];
          }
        }
      }
    }
  }

  private void embedToken(int token, float[] destination) {
    if (token < 0 || token >= config.vocabSize()) {
      throw new IllegalArgumentException("token is outside the vocabulary: " + token);
    }
    weights.embedToken(token, destination);
  }

  private void requireSession(Session session) {
    Objects.requireNonNull(session, "session");
    if (session.owner != this) {
      throw new IllegalArgumentException("session belongs to a different Flint forward pass");
    }
  }

  private static void addResidual(float[] residual, float[] update) {
    for (int index = 0; index < residual.length; index++) {
      residual[index] += update[index];
    }
  }
}
