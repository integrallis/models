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
package com.integrallis.models.backend.purejava.mobilemoe;

import com.integrallis.models.backend.purejava.cache.LayeredKvCache;
import com.integrallis.models.backend.purejava.cache.LayeredKvCache.AttentionSpan;
import com.integrallis.models.backend.purejava.cache.LayeredKvCache.AttentionView;
import com.integrallis.models.backend.purejava.cache.LayeredKvCache.LayerSpec;
import com.integrallis.models.backend.purejava.ops.RotaryTable;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.models.backend.purejava.tensor.TensorSource;
import com.integrallis.vectors.core.VectorUtil;
import java.util.Arrays;
import java.util.Objects;

/** Complete single-token Java decoder pass over mapped MobileMoE QAT Safetensors weights. */
public final class MobileMoeForwardPass {

  /** Mutable sequence state; sessions share immutable mapped weights and are otherwise isolated. */
  public static final class Session {
    private final MobileMoeForwardPass owner;
    private final LayeredKvCache cache;
    private final RotaryTable rope;
    private final float[] hidden;
    private final float[] normalized;
    private final float[] query;
    private final float[] key;
    private final float[] value;
    private final float[] attention;
    private final float[] attentionScores;
    private final float[] projectedAttention;
    private final float[] routerLogits;
    private final int[] selectedExperts;
    private final float[] routeWeights;
    private final float[] routedOutput;
    private final float[] gateUp;
    private final float[] activated;
    private final float[] expertOutput;
    private final float[] sharedGate;
    private final float[] sharedUp;
    private final float[] sharedActivated;
    private final float[] sharedOutput;
    private final float[] logits;
    private final int[] tokenHistory;
    private int nextPosition;

    private Session(MobileMoeForwardPass owner) {
      this.owner = owner;
      MobileMoeHuggingFaceConfig config = owner.config;
      LayerSpec[] specs = new LayerSpec[config.numLayers()];
      Arrays.fill(specs, LayerSpec.linear(config.keyValueDimension(), config.keyValueDimension()));
      cache = new LayeredKvCache(owner.maxSequenceLength, specs);
      MobileMoeHuggingFaceConfig.RopeScaling scaling = config.ropeScaling();
      rope =
          RotaryTable.llama3(
              config.headDimension(),
              config.ropeTheta(),
              scaling.factor(),
              scaling.originalContext(),
              scaling.lowFrequency(),
              scaling.highFrequency());
      hidden = new float[config.hiddenSize()];
      normalized = new float[config.hiddenSize()];
      query = new float[config.queryDimension()];
      key = new float[config.keyValueDimension()];
      value = new float[config.keyValueDimension()];
      attention = new float[config.queryDimension()];
      attentionScores = new float[owner.maxSequenceLength];
      projectedAttention = new float[config.hiddenSize()];
      routerLogits = new float[config.numExperts()];
      selectedExperts = new int[config.expertsPerToken()];
      routeWeights = new float[config.expertsPerToken()];
      routedOutput = new float[config.hiddenSize()];
      gateUp = new float[Math.multiplyExact(2, config.intermediateSize())];
      activated = new float[config.intermediateSize()];
      expertOutput = new float[config.hiddenSize()];
      sharedGate = new float[config.sharedIntermediateSize()];
      sharedUp = new float[config.sharedIntermediateSize()];
      sharedActivated = new float[config.sharedIntermediateSize()];
      sharedOutput = new float[config.hiddenSize()];
      logits = new float[config.vocabSize()];
      tokenHistory = new int[owner.maxSequenceLength];
    }

    /** Returns the next absolute sequence position accepted by this session. */
    public int checkpoint() {
      return nextPosition;
    }
  }

  private final MobileMoeHuggingFaceConfig config;
  private final MobileMoeWeights weights;
  private final int maxSequenceLength;

  private MobileMoeForwardPass(
      MobileMoeHuggingFaceConfig config, MobileMoeWeights weights, int maxSequenceLength) {
    this.config = Objects.requireNonNull(config, "config");
    this.weights = Objects.requireNonNull(weights, "weights");
    if (maxSequenceLength <= 0 || maxSequenceLength > config.contextLength()) {
      throw new IllegalArgumentException(
          "maxSequenceLength must be between 1 and "
              + config.contextLength()
              + ": "
              + maxSequenceLength);
    }
    this.maxSequenceLength = maxSequenceLength;
  }

  /** Loads the complete MobileMoE QAT graph from a mapped Safetensors source. */
  public static MobileMoeForwardPass load(
      MobileMoeHuggingFaceConfig config, TensorSource source, int maxSequenceLength) {
    Objects.requireNonNull(config, "config");
    return new MobileMoeForwardPass(
        config, MobileMoeWeights.load(source, config), maxSequenceLength);
  }

  /** Returns the validated checkpoint execution configuration. */
  public MobileMoeHuggingFaceConfig config() {
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

  /** Advances a prompt token without computing a discarded vocabulary projection. */
  public void advance(Session session, int token, int position) {
    execute(session, token, position, false);
  }

  /** Discards sequence state at and after {@code checkpoint}. */
  public void rewind(Session session, int checkpoint) {
    requireSession(session);
    if (checkpoint < 0 || checkpoint > session.nextPosition) {
      throw new IllegalArgumentException(
          "checkpoint must be between 0 and " + session.nextPosition + ": " + checkpoint);
    }
    session.cache.discardFrom(checkpoint);
    session.nextPosition = checkpoint;
  }

  /** Clears retained KV state and returns a session to position zero. */
  public void reset(Session session) {
    requireSession(session);
    session.cache.clear();
    session.nextPosition = 0;
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
    embedToken(token, session.hidden);
    session.rope.prepare(position);
    for (int layer = 0; layer < config.numLayers(); layer++) {
      executeLayer(session, layer, position);
    }
    if (projectLogits) {
      TensorOps.rmsNorm(
          session.normalized,
          session.hidden,
          weights.outputNorm(),
          config.hiddenSize(),
          config.rmsNormEpsilon());
      weights.embedding().multiply(session.normalized, session.logits);
    }
    session.tokenHistory[position] = token;
    session.nextPosition++;
  }

  private void executeLayer(Session session, int layerIndex, int position) {
    MobileMoeWeights.Layer layer = weights.layer(layerIndex);
    TensorOps.rmsNorm(
        session.normalized,
        session.hidden,
        layer.attentionNorm(),
        config.hiddenSize(),
        config.rmsNormEpsilon());
    layer.query().multiply(session.normalized, session.query);
    layer.key().multiply(session.normalized, session.key);
    layer.value().multiply(session.normalized, session.value);
    if (config.usesRope(layerIndex)) {
      for (int head = 0; head < config.numHeads(); head++) {
        session.rope.apply(session.query, head * config.headDimension(), false);
      }
      for (int head = 0; head < config.numKvHeads(); head++) {
        session.rope.apply(session.key, head * config.headDimension(), false);
      }
      MobileMoeMath.normalizeHeads(session.query, config.headDimension(), config.rmsNormEpsilon());
      MobileMoeMath.normalizeHeads(session.key, config.headDimension(), config.rmsNormEpsilon());
    }
    session.cache.store(layerIndex, position, session.key, session.value);
    computeAttention(session, layerIndex, position);
    layer.output().multiply(session.attention, session.projectedAttention);
    add(session.hidden, session.projectedAttention);

    TensorOps.rmsNorm(
        session.normalized,
        session.hidden,
        layer.postAttentionNorm(),
        config.hiddenSize(),
        config.rmsNormEpsilon());
    executeFeedForward(session, layer);
    add(session.hidden, session.routedOutput);
    add(session.hidden, session.sharedOutput);
  }

  private void computeAttention(Session session, int layer, int position) {
    Arrays.fill(session.attention, 0.0f);
    AttentionView view = session.cache.attentionView(layer, 0, position + 1);
    float[] cachedKeys = session.cache.keyBuffer(layer);
    float[] cachedValues = session.cache.valueBuffer(layer);
    int headsPerKeyValue = config.numHeads() / config.numKvHeads();
    float scale = (float) (1.0 / Math.sqrt(config.headDimension()));
    for (int head = 0; head < config.numHeads(); head++) {
      int keyValueHead = head / headsPerKeyValue;
      int queryOffset = head * config.headDimension();
      int score = 0;
      for (int spanIndex = 0; spanIndex < view.spanCount(); spanIndex++) {
        AttentionSpan span = view.span(spanIndex);
        int keyOffset = span.keyOffset() + keyValueHead * config.headDimension();
        for (int row = 0; row < span.positionCount(); row++) {
          session.attentionScores[score++] =
              scale
                  * VectorUtil.dotProduct(
                      session.query,
                      queryOffset,
                      cachedKeys,
                      keyOffset + row * config.keyValueDimension(),
                      config.headDimension());
        }
      }
      TensorOps.softmax(session.attentionScores, 0, view.positionCount());
      int probability = 0;
      int outputOffset = head * config.headDimension();
      for (int spanIndex = 0; spanIndex < view.spanCount(); spanIndex++) {
        AttentionSpan span = view.span(spanIndex);
        int valueOffset = span.valueOffset() + keyValueHead * config.headDimension();
        for (int row = 0; row < span.positionCount(); row++) {
          float weight = session.attentionScores[probability++];
          int rowOffset = valueOffset + row * config.keyValueDimension();
          for (int dimension = 0; dimension < config.headDimension(); dimension++) {
            session.attention[outputOffset + dimension] +=
                weight * cachedValues[rowOffset + dimension];
          }
        }
      }
    }
  }

  private void executeFeedForward(Session session, MobileMoeWeights.Layer layer) {
    layer.router().multiply(session.normalized, session.routerLogits);
    MobileMoeRouting.select(
        session.routerLogits,
        layer.expertBias(),
        config.expertsPerToken(),
        config.routeScale(),
        session.selectedExperts,
        session.routeWeights);
    Arrays.fill(session.routedOutput, 0.0f);
    for (int route = 0; route < session.selectedExperts.length; route++) {
      MobileMoeExperts.Expert expert = layer.experts().expert(session.selectedExperts[route]);
      expert.gateUp().multiply(session.normalized, session.gateUp);
      swiglu(session.gateUp, session.activated);
      expert.down().multiply(session.activated, session.expertOutput);
      addScaled(session.routedOutput, session.expertOutput, session.routeWeights[route]);
    }

    MobileMoeWeights.SharedExpert shared = layer.sharedExpert();
    shared.gate().multiply(session.normalized, session.sharedGate);
    shared.up().multiply(session.normalized, session.sharedUp);
    for (int index = 0; index < session.sharedActivated.length; index++) {
      session.sharedActivated[index] = silu(session.sharedGate[index]) * session.sharedUp[index];
    }
    shared.down().multiply(session.sharedActivated, session.sharedOutput);
  }

  private void embedToken(int token, float[] destination) {
    if (token < 0 || token >= config.vocabSize()) {
      throw new IllegalArgumentException("token is outside the vocabulary: " + token);
    }
    MobileMoePackedInt4Matrix embedding = weights.embedding();
    for (int index = 0; index < destination.length; index++) {
      destination[index] = embedding.value(token, index);
    }
  }

  private void requireSession(Session session) {
    Objects.requireNonNull(session, "session");
    if (session.owner != this) {
      throw new IllegalArgumentException("session belongs to a different MobileMoE forward pass");
    }
  }

  private static void swiglu(float[] gateUp, float[] output) {
    int width = output.length;
    for (int index = 0; index < width; index++) {
      output[index] = silu(gateUp[index]) * gateUp[width + index];
    }
  }

  private static float silu(float value) {
    return value * MobileMoeRouting.sigmoid(value);
  }

  private static void add(float[] destination, float[] update) {
    for (int index = 0; index < destination.length; index++) {
      destination[index] += update[index];
    }
  }

  private static void addScaled(float[] destination, float[] update, float scale) {
    for (int index = 0; index < destination.length; index++) {
      destination[index] += scale * update[index];
    }
  }
}
