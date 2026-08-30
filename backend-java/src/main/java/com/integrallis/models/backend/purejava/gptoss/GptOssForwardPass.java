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
package com.integrallis.models.backend.purejava.gptoss;

import com.integrallis.models.backend.purejava.cache.LayeredKvCache;
import com.integrallis.models.backend.purejava.cache.LayeredKvCache.AttentionSpan;
import com.integrallis.models.backend.purejava.cache.LayeredKvCache.AttentionView;
import com.integrallis.models.backend.purejava.cache.LayeredKvCache.LayerSpec;
import com.integrallis.models.backend.purejava.ops.RotaryTable;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.vectors.core.BFloat16Matrix;
import com.integrallis.vectors.core.VectorUtil;
import java.util.Arrays;
import java.util.Objects;

/** Complete single-token Java decoder pass over mapped GPT-OSS Safetensors weights. */
final class GptOssForwardPass {

  /** Mutable sequence state; sessions share immutable mapped weights and are otherwise isolated. */
  static final class Session {
    private final GptOssForwardPass owner;
    private final LayeredKvCache cache;
    private final RotaryTable rope;
    private final GptOssMxfp4Moe[] moes;
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
    private int nextPosition;

    private Session(GptOssForwardPass owner) {
      this.owner = owner;
      GptOssHuggingFaceConfig config = owner.config;
      cache = new LayeredKvCache(owner.maxSequenceLength, owner.cacheSpecs.clone());
      rope =
          RotaryTable.yarn(
              config.headDim(),
              config.ropeTheta(),
              config.ropeYarnFactor(),
              config.ropeBetaFast(),
              config.ropeBetaSlow(),
              config.ropeOriginalContext(),
              false);
      moes = new GptOssMxfp4Moe[owner.layers.length];
      for (int layer = 0; layer < moes.length; layer++) {
        moes[layer] =
            new GptOssMxfp4Moe(
                owner.layers[layer].experts(), config.hiddenActAlpha(), config.swigluLimit());
      }
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
    }

    /** Returns the next absolute sequence position accepted by this session. */
    int checkpoint() {
      return nextPosition;
    }
  }

  private record LayerParameters(
      float[] attentionNorm,
      BFloat16Matrix query,
      float[] queryBias,
      BFloat16Matrix key,
      float[] keyBias,
      BFloat16Matrix value,
      float[] valueBias,
      BFloat16Matrix output,
      float[] outputBias,
      float[] sinks,
      float[] postAttentionNorm,
      BFloat16Matrix router,
      float[] routerBias,
      GptOssMxfp4ExpertWeights experts) {

    private static LayerParameters snapshot(GptOssWeights.Layer layer) {
      return new LayerParameters(
          layer.attentionNorm(),
          layer.query(),
          layer.queryBias(),
          layer.key(),
          layer.keyBias(),
          layer.value(),
          layer.valueBias(),
          layer.output(),
          layer.outputBias(),
          layer.sinks(),
          layer.postAttentionNorm(),
          layer.router(),
          layer.routerBias(),
          layer.experts());
    }
  }

  private final GptOssHuggingFaceConfig config;
  private final GptOssWeights weights;
  private final LayerParameters[] layers;
  private final LayerSpec[] cacheSpecs;
  private final float[] outputNorm;
  private final int maxSequenceLength;

  GptOssForwardPass(GptOssHuggingFaceConfig config, GptOssWeights weights, int maxSequenceLength) {
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
    layers = new LayerParameters[config.numLayers()];
    cacheSpecs = new LayerSpec[config.numLayers()];
    for (int layer = 0; layer < layers.length; layer++) {
      layers[layer] = LayerParameters.snapshot(weights.layer(layer));
      cacheSpecs[layer] =
          config.usesSlidingAttention(layer)
              ? LayerSpec.ring(
                  config.keyValueDimension(), config.keyValueDimension(), config.slidingWindow())
              : LayerSpec.linear(config.keyValueDimension(), config.keyValueDimension());
    }
  }

  /** Opens an independent sequence over the same mapped model weights. */
  Session openSession() {
    return new Session(this);
  }

  /** Runs one sequential token and returns stable vocabulary logits. */
  float[] forward(Session session, int token, int position) {
    return forwardTransient(session, token, position).clone();
  }

  /** Runs one sequential token using session-owned logits storage. */
  float[] forwardTransient(Session session, int token, int position) {
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
    TensorOps.rmsNorm(
        session.xNorm, session.x, outputNorm, config.hiddenSize(), config.rmsNormEps());
    weights.output().multiply(session.xNorm, session.logits);
    session.nextPosition++;
    return session.logits;
  }

  /** Clears retained KV state and returns a session to position zero. */
  void reset(Session session) {
    requireSession(session);
    session.cache.clear();
    session.nextPosition = 0;
  }

  private void executeLayer(Session session, int layer, int position) {
    LayerParameters parameters = layers[layer];
    TensorOps.rmsNorm(
        session.xNorm,
        session.x,
        parameters.attentionNorm(),
        config.hiddenSize(),
        config.rmsNormEps());
    parameters.query().multiply(session.xNorm, session.query);
    parameters.key().multiply(session.xNorm, session.key);
    parameters.value().multiply(session.xNorm, session.value);
    addBias(session.query, parameters.queryBias());
    addBias(session.key, parameters.keyBias());
    addBias(session.value, parameters.valueBias());

    for (int head = 0; head < config.numHeads(); head++) {
      session.rope.apply(session.query, head * config.headDim(), true);
    }
    for (int head = 0; head < config.numKvHeads(); head++) {
      session.rope.apply(session.key, head * config.headDim(), true);
    }
    session.cache.store(layer, position, session.key, session.value);
    computeAttention(session, parameters, layer, position);

    parameters.output().multiply(session.attentionOutput, session.projectedAttention);
    addBias(session.projectedAttention, parameters.outputBias());
    addResidual(session.x, session.projectedAttention);

    TensorOps.rmsNorm(
        session.xNorm,
        session.x,
        parameters.postAttentionNorm(),
        config.hiddenSize(),
        config.rmsNormEps());
    parameters.router().multiply(session.xNorm, session.routerLogits);
    addBias(session.routerLogits, parameters.routerBias());
    GptOssMath.selectExperts(session.routerLogits, session.selectedExperts, session.routingWeights);
    session.moes[layer].forwardQ8(
        session.xNorm, session.selectedExperts, session.routingWeights, session.moeOutput);
    addResidual(session.x, session.moeOutput);
  }

  private void computeAttention(
      Session session, LayerParameters parameters, int layer, int position) {
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
                  * VectorUtil.dotProduct(
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
    BFloat16Matrix embedding = weights.tokenEmbedding();
    for (int index = 0; index < destination.length; index++) {
      destination[index] = embedding.value(token, index);
    }
  }

  private void requireSession(Session session) {
    Objects.requireNonNull(session, "session");
    if (session.owner != this) {
      throw new IllegalArgumentException("session belongs to a different GPT-OSS forward pass");
    }
  }

  private static void addBias(float[] values, float[] bias) {
    for (int index = 0; index < values.length; index++) {
      values[index] += bias[index];
    }
  }

  private static void addResidual(float[] residual, float[] update) {
    for (int index = 0; index < residual.length; index++) {
      residual[index] += update[index];
    }
  }
}
