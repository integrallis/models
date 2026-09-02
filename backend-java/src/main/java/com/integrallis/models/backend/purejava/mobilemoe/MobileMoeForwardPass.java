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

  private static final int DEFAULT_PREFILL_BATCH_SIZE = 64;

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

  private static final class BatchState {
    private final int capacity;
    private final float[] hidden;
    private final float[] normalized;
    private final float[] query;
    private final float[] key;
    private final float[] value;
    private final float[] attention;
    private final float[] projectedAttention;
    private final float[] routerLogits;
    private final int[] selectedExperts;
    private final float[] routeWeights;
    private final float[] routedOutput;
    private final float[] sharedGate;
    private final float[] sharedUp;
    private final float[] sharedActivated;
    private final float[] sharedOutput;
    private final float[] expertInput;
    private final float[] expertGateUp;
    private final float[] expertActivated;
    private final float[] expertOutput;
    private final int[] expertTokens;
    private final int[] expertRoutes;

    private BatchState(MobileMoeHuggingFaceConfig config, int capacity) {
      this.capacity = capacity;
      hidden = batchBuffer(capacity, config.hiddenSize());
      normalized = batchBuffer(capacity, config.hiddenSize());
      query = batchBuffer(capacity, config.queryDimension());
      key = batchBuffer(capacity, config.keyValueDimension());
      value = batchBuffer(capacity, config.keyValueDimension());
      attention = batchBuffer(capacity, config.queryDimension());
      projectedAttention = batchBuffer(capacity, config.hiddenSize());
      routerLogits = batchBuffer(capacity, config.numExperts());
      selectedExperts = new int[Math.multiplyExact(capacity, config.expertsPerToken())];
      routeWeights = batchBuffer(capacity, config.expertsPerToken());
      routedOutput = batchBuffer(capacity, config.hiddenSize());
      sharedGate = batchBuffer(capacity, config.sharedIntermediateSize());
      sharedUp = batchBuffer(capacity, config.sharedIntermediateSize());
      sharedActivated = batchBuffer(capacity, config.sharedIntermediateSize());
      sharedOutput = batchBuffer(capacity, config.hiddenSize());
      expertInput = batchBuffer(capacity, config.hiddenSize());
      expertGateUp = batchBuffer(capacity, Math.multiplyExact(2, config.intermediateSize()));
      expertActivated = batchBuffer(capacity, config.intermediateSize());
      expertOutput = batchBuffer(capacity, config.hiddenSize());
      expertTokens = new int[capacity];
      expertRoutes = new int[capacity];
    }

    private static float[] batchBuffer(int capacity, int width) {
      return new float[Math.multiplyExact(capacity, width)];
    }
  }

  private final MobileMoeHuggingFaceConfig config;
  private final MobileMoeWeights weights;
  private final int maxSequenceLength;
  private final int prefillBatchSize;

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
    int requestedBatchSize =
        Integer.getInteger("models.mobilemoe.prefillBatchSize", DEFAULT_PREFILL_BATCH_SIZE);
    if (requestedBatchSize <= 0) {
      throw new IllegalArgumentException(
          "models.mobilemoe.prefillBatchSize must be positive: " + requestedBatchSize);
    }
    prefillBatchSize = Math.min(requestedBatchSize, maxSequenceLength);
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

  /**
   * Prefills consecutive prompt tokens in weight-reusing batches and returns final-token logits.
   */
  public float[] prefill(Session session, int[] tokens, int startPosition) {
    requireSession(session);
    Objects.requireNonNull(tokens, "tokens");
    if (tokens.length == 0) {
      throw new IllegalArgumentException("tokens must not be empty");
    }
    if (startPosition != session.nextPosition) {
      throw new IllegalArgumentException(
          "position must be sequential: expected "
              + session.nextPosition
              + ", got "
              + startPosition);
    }
    if ((long) startPosition + tokens.length > maxSequenceLength) {
      throw new IllegalArgumentException("prefill exceeds the configured context");
    }

    BatchState batch = new BatchState(config, Math.min(prefillBatchSize, tokens.length));
    for (int tokenOffset = 0; tokenOffset < tokens.length; ) {
      int batchSize = Math.min(batch.capacity, tokens.length - tokenOffset);
      int batchStart = startPosition + tokenOffset;
      executeBatch(session, batch, tokens, tokenOffset, batchStart, batchSize);
      for (int index = 0; index < batchSize; index++) {
        session.tokenHistory[batchStart + index] = tokens[tokenOffset + index];
      }
      session.nextPosition += batchSize;
      tokenOffset += batchSize;
    }
    return session.logits.clone();
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

  private void executeBatch(
      Session session,
      BatchState batch,
      int[] tokens,
      int tokenOffset,
      int startPosition,
      int batchSize) {
    int hiddenSize = config.hiddenSize();
    for (int index = 0; index < batchSize; index++) {
      embedToken(tokens[tokenOffset + index], batch.hidden, index * hiddenSize);
    }
    session.rope.prepareBatch(startPosition, batchSize);
    for (int layer = 0; layer < config.numLayers(); layer++) {
      executeLayerBatch(session, batch, layer, startPosition, batchSize);
    }
    int finalOffset = (batchSize - 1) * hiddenSize;
    TensorOps.rmsNorm(
        session.normalized,
        0,
        batch.hidden,
        finalOffset,
        weights.outputNorm(),
        hiddenSize,
        config.rmsNormEpsilon());
    weights.embedding().multiply(session.normalized, session.logits);
  }

  private void executeLayerBatch(
      Session session, BatchState batch, int layerIndex, int startPosition, int batchSize) {
    MobileMoeWeights.Layer layer = weights.layer(layerIndex);
    int hiddenSize = config.hiddenSize();
    int queryDimension = config.queryDimension();
    int keyValueDimension = config.keyValueDimension();
    float[] attentionNorm = layer.attentionNorm();
    for (int index = 0; index < batchSize; index++) {
      int hiddenOffset = index * hiddenSize;
      TensorOps.rmsNorm(
          batch.normalized,
          hiddenOffset,
          batch.hidden,
          hiddenOffset,
          attentionNorm,
          hiddenSize,
          config.rmsNormEpsilon());
    }
    layer.query().multiplyBatch(batch.normalized, batchSize, batch.query);
    layer.key().multiplyBatch(batch.normalized, batchSize, batch.key);
    layer.value().multiplyBatch(batch.normalized, batchSize, batch.value);
    if (config.usesRope(layerIndex)) {
      for (int index = 0; index < batchSize; index++) {
        int queryOffset = index * queryDimension;
        int keyOffset = index * keyValueDimension;
        for (int head = 0; head < config.numHeads(); head++) {
          session.rope.applyBatch(
              batch.query, queryOffset + head * config.headDimension(), index, false);
        }
        for (int head = 0; head < config.numKvHeads(); head++) {
          session.rope.applyBatch(
              batch.key, keyOffset + head * config.headDimension(), index, false);
        }
        MobileMoeMath.normalizeHeads(
            batch.query,
            queryOffset,
            queryDimension,
            config.headDimension(),
            config.rmsNormEpsilon());
        MobileMoeMath.normalizeHeads(
            batch.key,
            keyOffset,
            keyValueDimension,
            config.headDimension(),
            config.rmsNormEpsilon());
      }
    }
    for (int index = 0; index < batchSize; index++) {
      session.cache.store(
          layerIndex,
          startPosition + index,
          batch.key,
          index * keyValueDimension,
          batch.value,
          index * keyValueDimension);
    }
    for (int index = 0; index < batchSize; index++) {
      computeAttention(
          session,
          layerIndex,
          startPosition + index,
          batch.query,
          index * queryDimension,
          batch.attention,
          index * queryDimension);
    }
    layer.output().multiplyBatch(batch.attention, batchSize, batch.projectedAttention);
    addBatch(batch.hidden, batch.projectedAttention, batchSize, hiddenSize);

    float[] postAttentionNorm = layer.postAttentionNorm();
    for (int index = 0; index < batchSize; index++) {
      int hiddenOffset = index * hiddenSize;
      TensorOps.rmsNorm(
          batch.normalized,
          hiddenOffset,
          batch.hidden,
          hiddenOffset,
          postAttentionNorm,
          hiddenSize,
          config.rmsNormEpsilon());
    }
    executeFeedForwardBatch(session, batch, layer, batchSize);
    addBatch(batch.hidden, batch.routedOutput, batchSize, hiddenSize);
    addBatch(batch.hidden, batch.sharedOutput, batchSize, hiddenSize);
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
    computeAttention(session, layer, position, session.query, 0, session.attention, 0);
  }

  private void computeAttention(
      Session session,
      int layer,
      int position,
      float[] query,
      int queryBaseOffset,
      float[] attention,
      int attentionBaseOffset) {
    Arrays.fill(
        attention, attentionBaseOffset, attentionBaseOffset + config.queryDimension(), 0.0f);
    AttentionView view = session.cache.attentionView(layer, 0, position + 1);
    float[] cachedKeys = session.cache.keyBuffer(layer);
    float[] cachedValues = session.cache.valueBuffer(layer);
    int headsPerKeyValue = config.numHeads() / config.numKvHeads();
    float scale = (float) (1.0 / Math.sqrt(config.headDimension()));
    for (int head = 0; head < config.numHeads(); head++) {
      int keyValueHead = head / headsPerKeyValue;
      int queryOffset = queryBaseOffset + head * config.headDimension();
      int score = 0;
      for (int spanIndex = 0; spanIndex < view.spanCount(); spanIndex++) {
        AttentionSpan span = view.span(spanIndex);
        int keyOffset = span.keyOffset() + keyValueHead * config.headDimension();
        for (int row = 0; row < span.positionCount(); row++) {
          session.attentionScores[score++] =
              scale
                  * VectorUtil.dotProduct(
                      query,
                      queryOffset,
                      cachedKeys,
                      keyOffset + row * config.keyValueDimension(),
                      config.headDimension());
        }
      }
      TensorOps.softmax(session.attentionScores, 0, view.positionCount());
      int probability = 0;
      int outputOffset = attentionBaseOffset + head * config.headDimension();
      for (int spanIndex = 0; spanIndex < view.spanCount(); spanIndex++) {
        AttentionSpan span = view.span(spanIndex);
        int valueOffset = span.valueOffset() + keyValueHead * config.headDimension();
        for (int row = 0; row < span.positionCount(); row++) {
          float weight = session.attentionScores[probability++];
          int rowOffset = valueOffset + row * config.keyValueDimension();
          for (int dimension = 0; dimension < config.headDimension(); dimension++) {
            attention[outputOffset + dimension] += weight * cachedValues[rowOffset + dimension];
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

  private void executeFeedForwardBatch(
      Session session, BatchState batch, MobileMoeWeights.Layer layer, int batchSize) {
    int hiddenSize = config.hiddenSize();
    int expertHidden = config.intermediateSize();
    int gateUpWidth = Math.multiplyExact(2, expertHidden);
    int topK = config.expertsPerToken();
    layer.router().multiplyBatch(batch.normalized, batchSize, batch.routerLogits);
    float[] expertBias = layer.expertBias();
    for (int token = 0; token < batchSize; token++) {
      System.arraycopy(
          batch.routerLogits,
          token * config.numExperts(),
          session.routerLogits,
          0,
          config.numExperts());
      MobileMoeRouting.select(
          session.routerLogits,
          expertBias,
          topK,
          config.routeScale(),
          session.selectedExperts,
          session.routeWeights);
      System.arraycopy(session.selectedExperts, 0, batch.selectedExperts, token * topK, topK);
      System.arraycopy(session.routeWeights, 0, batch.routeWeights, token * topK, topK);
    }

    Arrays.fill(batch.routedOutput, 0, batchSize * hiddenSize, 0.0f);
    for (int expertIndex = 0; expertIndex < config.numExperts(); expertIndex++) {
      int assignmentCount = 0;
      for (int token = 0; token < batchSize; token++) {
        for (int route = 0; route < topK; route++) {
          if (batch.selectedExperts[token * topK + route] == expertIndex) {
            batch.expertTokens[assignmentCount] = token;
            batch.expertRoutes[assignmentCount] = route;
            System.arraycopy(
                batch.normalized,
                token * hiddenSize,
                batch.expertInput,
                assignmentCount * hiddenSize,
                hiddenSize);
            assignmentCount++;
          }
        }
      }
      if (assignmentCount == 0) {
        continue;
      }
      MobileMoeExperts.Expert expert = layer.experts().expert(expertIndex);
      expert.gateUp().multiplyBatch(batch.expertInput, assignmentCount, batch.expertGateUp);
      swigluBatch(
          batch.expertGateUp, batch.expertActivated, assignmentCount, expertHidden, gateUpWidth);
      expert.down().multiplyBatch(batch.expertActivated, assignmentCount, batch.expertOutput);
      for (int assignment = 0; assignment < assignmentCount; assignment++) {
        int token = batch.expertTokens[assignment];
        int route = batch.expertRoutes[assignment];
        addScaled(
            batch.routedOutput,
            token * hiddenSize,
            batch.expertOutput,
            assignment * hiddenSize,
            hiddenSize,
            batch.routeWeights[token * topK + route]);
      }
    }

    MobileMoeWeights.SharedExpert shared = layer.sharedExpert();
    shared.gate().multiplyBatch(batch.normalized, batchSize, batch.sharedGate);
    shared.up().multiplyBatch(batch.normalized, batchSize, batch.sharedUp);
    int sharedWidth = config.sharedIntermediateSize();
    for (int token = 0; token < batchSize; token++) {
      int offset = token * sharedWidth;
      for (int index = 0; index < sharedWidth; index++) {
        batch.sharedActivated[offset + index] =
            silu(batch.sharedGate[offset + index]) * batch.sharedUp[offset + index];
      }
    }
    shared.down().multiplyBatch(batch.sharedActivated, batchSize, batch.sharedOutput);
  }

  private void embedToken(int token, float[] destination) {
    embedToken(token, destination, 0);
  }

  private void embedToken(int token, float[] destination, int destinationOffset) {
    if (token < 0 || token >= config.vocabSize()) {
      throw new IllegalArgumentException("token is outside the vocabulary: " + token);
    }
    MobileMoePackedInt4Matrix embedding = weights.embedding();
    Objects.checkFromIndexSize(destinationOffset, config.hiddenSize(), destination.length);
    for (int index = 0; index < config.hiddenSize(); index++) {
      destination[destinationOffset + index] = embedding.value(token, index);
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

  private static void swigluBatch(
      float[] gateUp, float[] output, int batchSize, int width, int gateUpWidth) {
    for (int batch = 0; batch < batchSize; batch++) {
      int gateOffset = batch * gateUpWidth;
      int outputOffset = batch * width;
      for (int index = 0; index < width; index++) {
        output[outputOffset + index] =
            silu(gateUp[gateOffset + index]) * gateUp[gateOffset + width + index];
      }
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

  private static void addBatch(float[] destination, float[] update, int batchSize, int rowWidth) {
    int entries = Math.multiplyExact(batchSize, rowWidth);
    for (int index = 0; index < entries; index++) {
      destination[index] += update[index];
    }
  }

  private static void addScaled(float[] destination, float[] update, float scale) {
    for (int index = 0; index < destination.length; index++) {
      destination[index] += scale * update[index];
    }
  }

  private static void addScaled(
      float[] destination,
      int destinationOffset,
      float[] update,
      int updateOffset,
      int length,
      float scale) {
    for (int index = 0; index < length; index++) {
      destination[destinationOffset + index] += scale * update[updateOffset + index];
    }
  }
}
