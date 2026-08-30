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
package com.integrallis.models.backend.purejava.qwen35;

import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.ops.RotaryTable;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.models.backend.purejava.plan.ModelTopology;
import com.integrallis.models.backend.purejava.plan.PureJavaExecutionPlan;
import com.integrallis.models.backend.purejava.plan.PureJavaPlanConfiguration;
import com.integrallis.models.backend.purejava.qwen35.Qwen35Weights.FullAttention;
import com.integrallis.models.backend.purejava.qwen35.Qwen35Weights.GatedDeltaNet;
import com.integrallis.models.backend.purejava.qwen35.Qwen35Weights.Layer;
import com.integrallis.models.backend.purejava.qwen35.Qwen35Weights.Matrix;
import com.integrallis.vectors.core.GgufQ4Kernel;
import com.integrallis.vectors.core.GgufQ6BatchedKernel;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** Stateful scalar/Vector-API compatibility graph for dense Qwen3.5. */
public final class Qwen35ForwardPass {

  /** Independent Qwen3.5 sequence state. */
  public static final class Session {
    private final Qwen35ForwardPass owner;
    private final int capacity;
    private final int[] tokenHistory;
    private SessionState state;
    private int checkpoint;

    private Session(Qwen35ForwardPass owner, int capacity) {
      if (capacity <= 0 || capacity > owner.config.contextLength()) {
        throw new IllegalArgumentException(
            "capacity must be between 1 and " + owner.config.contextLength() + ": " + capacity);
      }
      this.owner = owner;
      this.capacity = capacity;
      this.tokenHistory = new int[capacity];
      this.state = new SessionState(owner.config, owner.weights, capacity, owner.prefillBatchSize);
    }

    public int checkpoint() {
      return checkpoint;
    }
  }

  private final Qwen35Config config;
  private final Qwen35Weights weights;
  private final RotaryTable rotary;
  private final int prefillBatchSize;
  private final GgufQ4Kernel q4Kernel;
  private final GgufQ6BatchedKernel q6BatchedKernel;

  private Qwen35ForwardPass(
      Qwen35Config config,
      Qwen35Weights weights,
      int prefillBatchSize,
      GgufQ4Kernel q4Kernel,
      GgufQ6BatchedKernel q6BatchedKernel) {
    this.config = config;
    this.weights = weights;
    this.rotary = new RotaryTable(config.ropeDimension(), config.ropeTheta(), 1.0f);
    if (prefillBatchSize < 1) {
      throw new IllegalArgumentException("prefillBatchSize must be >= 1: " + prefillBatchSize);
    }
    this.prefillBatchSize = Math.min(prefillBatchSize, config.contextLength());
    this.q4Kernel = Objects.requireNonNull(q4Kernel, "q4Kernel");
    this.q6BatchedKernel = Objects.requireNonNull(q6BatchedKernel, "q6BatchedKernel");
  }

  public static Qwen35ForwardPass fromGgufFile(GgufFile file) {
    return fromGgufFile(file, PureJavaPlanConfiguration.DEFAULT_PREFILL_BATCH_SIZE);
  }

  static Qwen35ForwardPass fromGgufFile(GgufFile file, int prefillBatchSize) {
    Objects.requireNonNull(file, "file");
    Qwen35Config config = Qwen35Config.fromMetadata(file.metadata());
    return new Qwen35ForwardPass(
        config,
        Qwen35Weights.fromGgufFile(file, config),
        prefillBatchSize,
        GgufQ4Kernel.WIDENED,
        GgufQ6BatchedKernel.ONE_QUERY_BLOCK);
  }

  /** Returns a graph sharing the same mapped weights with the requested prefill batch size. */
  public Qwen35ForwardPass withPrefillBatchSize(int batchSize) {
    return new Qwen35ForwardPass(config, weights, batchSize, q4Kernel, q6BatchedKernel);
  }

  /** Returns a graph sharing the same weights and honoring the selected execution plan. */
  public Qwen35ForwardPass withExecutionPlan(PureJavaExecutionPlan plan) {
    Objects.requireNonNull(plan, "plan");
    return new Qwen35ForwardPass(
        config, weights, plan.prefillBatchSize(), plan.q4Kernel(), plan.q6BatchedKernel());
  }

  /** Returns the projection topology loaded from this graph's actual GGUF tensors. */
  public ModelTopology topology() {
    List<ModelTopology.LayerTopology> layers = new ArrayList<>(config.numLayers());
    for (int layerIndex = 0; layerIndex < config.numLayers(); layerIndex++) {
      Layer layer = weights.layer(layerIndex);
      if (layer.fullAttention() != null) {
        FullAttention attention = layer.fullAttention();
        layers.add(
            new ModelTopology.LayerTopology(
                attention.queryGate().type(),
                attention.key().type(),
                attention.value().type(),
                attention.output().type(),
                layer.ffnGate().type(),
                layer.ffnUp().type(),
                layer.ffnDown().type()));
      } else {
        GatedDeltaNet gdn = layer.gatedDeltaNet();
        layers.add(
            new ModelTopology.LayerTopology(
                gdn.queryKeyValue().type(),
                gdn.queryKeyValue().type(),
                gdn.queryKeyValue().type(),
                gdn.output().type(),
                layer.ffnGate().type(),
                layer.ffnUp().type(),
                layer.ffnDown().type(),
                List.of(gdn.outputGate().type(), gdn.beta().type(), gdn.alpha().type())));
      }
    }
    return new ModelTopology(
        "qwen35",
        config.attentionQueryDim(),
        config.attentionKeyDim(),
        config.attentionKeyDim(),
        layers,
        weights.hasThreadShareableProjectionWeights());
  }

  /** Runs one token from empty full-attention, convolution, and recurrent state. */
  float[] forwardFresh(int token) {
    return forward(new int[] {token});
  }

  /** Runs a non-empty prompt from empty state and returns logits after its final token. */
  float[] forward(int[] tokens) {
    Objects.requireNonNull(tokens, "tokens");
    if (tokens.length == 0) {
      throw new IllegalArgumentException("tokens must not be empty");
    }
    if (tokens.length > config.contextLength()) {
      throw new IllegalArgumentException(
          "tokens exceed context length " + config.contextLength() + ": " + tokens.length);
    }

    Session session = openSession(tokens.length);
    return prefill(session, tokens, 0);
  }

  /** Opens independent sequence state with a bounded KV cache. */
  public Session openSession(int capacity) {
    return new Session(this, capacity);
  }

  /** Advances one sequence position and returns newly allocated vocabulary logits. */
  public float[] forward(Session session, int token, int position) {
    Session checked = requireSession(session);
    requireSequentialPosition(checked, position);
    return advance(checked, token, true);
  }

  /** Advances a non-empty token sequence and returns logits after its final position. */
  public float[] prefill(Session session, int[] tokens, int startPosition) {
    Objects.requireNonNull(tokens, "tokens");
    if (tokens.length == 0) {
      throw new IllegalArgumentException("tokens must not be empty");
    }
    Session checked = requireSession(session);
    requireSequentialPosition(checked, startPosition);
    if (tokens.length > checked.capacity - checked.checkpoint) {
      throw new IllegalArgumentException(
          "tokens exceed remaining session capacity: "
              + tokens.length
              + " > "
              + (checked.capacity - checked.checkpoint));
    }
    int tokenOffset = 0;
    while (tokenOffset < tokens.length) {
      int batchSize = Math.min(prefillBatchSize, tokens.length - tokenOffset);
      boolean finalBatch = tokenOffset + batchSize == tokens.length;
      if (batchSize == 1) {
        float[] logits = advance(checked, tokens[tokenOffset], finalBatch);
        tokenOffset++;
        if (finalBatch) {
          return logits;
        }
        continue;
      }
      executePrefillBatch(checked, tokens, tokenOffset, batchSize);
      tokenOffset += batchSize;
      if (finalBatch) {
        return projectFinalBatchLogits(
            checked.state.batchScratch, batchSize, checked.state.scratch);
      }
    }
    throw new AssertionError("non-empty prefill produced no logits");
  }

  /** Reconstructs sequence state at an earlier checkpoint from its retained token prefix. */
  public void rewind(Session session, int checkpoint) {
    Session checked = requireSession(session);
    if (checkpoint < 0 || checkpoint > checked.checkpoint) {
      throw new IllegalArgumentException(
          "checkpoint must be between 0 and " + checked.checkpoint + ": " + checkpoint);
    }
    checked.state = new SessionState(config, weights, checked.capacity, prefillBatchSize);
    checked.checkpoint = 0;
    for (int index = 0; index < checkpoint; index++) {
      advance(checked, checked.tokenHistory[index], false);
    }
  }

  /** Clears all sequence state. */
  public void reset(Session session) {
    Session checked = requireSession(session);
    checked.state = new SessionState(config, weights, checked.capacity, prefillBatchSize);
    checked.checkpoint = 0;
  }

  public Qwen35Config config() {
    return config;
  }

  int prefillBatchSize() {
    return prefillBatchSize;
  }

  private float[] advance(Session session, int token, boolean projectLogits) {
    if (session.checkpoint >= session.capacity) {
      throw new IllegalStateException("session context capacity exhausted: " + session.capacity);
    }
    int position = session.checkpoint;
    float[] hidden = forwardToken(token, position, session.state);
    session.tokenHistory[position] = token;
    session.checkpoint++;
    return projectLogits ? projectLogits(hidden, session.state.scratch) : null;
  }

  private float[] projectLogits(float[] hidden, ProjectionScratch scratch) {
    float[] normalized = new float[config.embeddingDim()];
    TensorOps.rmsNorm(
        normalized, hidden, weights.outputNorm(), config.embeddingDim(), config.rmsNormEpsilon());
    float[] logits = new float[config.vocabSize()];
    project(logits, normalized, weights.output(), scratch);
    return logits;
  }

  private float[] projectFinalBatchLogits(
      BatchScratch batch, int batchSize, ProjectionScratch scratch) {
    int dimension = config.embeddingDim();
    int offset = (batchSize - 1) * dimension;
    float[] normalized = new float[dimension];
    TensorOps.rmsNorm(
        normalized,
        0,
        batch.state,
        offset,
        weights.outputNorm(),
        dimension,
        config.rmsNormEpsilon());
    float[] logits = new float[config.vocabSize()];
    project(logits, normalized, weights.output(), scratch);
    return logits;
  }

  private Session requireSession(Session session) {
    Objects.requireNonNull(session, "session");
    if (session.owner != this) {
      throw new IllegalArgumentException("session belongs to a different Qwen3.5 graph");
    }
    return session;
  }

  private static void requireSequentialPosition(Session session, int position) {
    if (position != session.checkpoint) {
      throw new IllegalArgumentException(
          "position must be sequential: expected " + session.checkpoint + ", got " + position);
    }
  }

  private float[] forwardToken(int token, int position, SessionState session) {
    int embeddingDimension = config.embeddingDim();
    float[] state = new float[embeddingDimension];
    float[] normalized = new float[embeddingDimension];
    float[] projected = new float[embeddingDimension];
    float[] ffnGate = new float[config.hiddenDim()];
    float[] ffnUp = new float[config.hiddenDim()];
    float[] ffn = new float[config.hiddenDim()];
    weights.embedToken(token, state);

    for (int layerIndex = 0; layerIndex < config.numLayers(); layerIndex++) {
      Layer layer = weights.layer(layerIndex);
      TensorOps.rmsNorm(
          normalized, state, layer.attentionNorm(), embeddingDimension, config.rmsNormEpsilon());
      if (config.usesFullAttention(layerIndex)) {
        fullAttention(
            projected,
            normalized,
            layer.fullAttention(),
            position,
            session.keys[layerIndex],
            session.values[layerIndex],
            session.scratch);
      } else {
        gatedDeltaNet(projected, normalized, layer.gatedDeltaNet(), layerIndex, session);
      }
      add(state, projected);

      TensorOps.rmsNorm(
          normalized,
          state,
          layer.postAttentionNorm(),
          embeddingDimension,
          config.rmsNormEpsilon());
      dualProject(ffnGate, layer.ffnGate(), ffnUp, layer.ffnUp(), normalized, session.scratch);
      TensorOps.swiGlu(ffn, ffnGate, ffnUp, config.hiddenDim());
      project(projected, ffn, layer.ffnDown(), session.scratch);
      add(state, projected);
    }

    return state;
  }

  private void executePrefillBatch(Session session, int[] tokens, int tokenOffset, int batchSize) {
    SessionState sessionState = session.state;
    BatchScratch batch = sessionState.batchScratch;
    int dimension = config.embeddingDim();
    int startPosition = session.checkpoint;
    for (int index = 0; index < batchSize; index++) {
      weights.embedToken(tokens[tokenOffset + index], batch.rowState);
      System.arraycopy(batch.rowState, 0, batch.state, index * dimension, dimension);
    }

    for (int layerIndex = 0; layerIndex < config.numLayers(); layerIndex++) {
      Layer layer = weights.layer(layerIndex);
      normalizeBatch(batch.normalized, batch.state, batchSize, dimension, layer.attentionNorm());
      if (config.usesFullAttention(layerIndex)) {
        fullAttentionBatched(
            batch.projected,
            batch.normalized,
            layer.fullAttention(),
            layerIndex,
            startPosition,
            batchSize,
            sessionState,
            batch);
      } else {
        gatedDeltaNetBatched(
            batch.projected,
            batch.normalized,
            layer.gatedDeltaNet(),
            layerIndex,
            batchSize,
            sessionState,
            batch);
      }
      addBatch(batch.state, batch.projected, batchSize * dimension);

      normalizeBatch(
          batch.normalized, batch.state, batchSize, dimension, layer.postAttentionNorm());
      dualProjectBatched(
          batch.ffnGate,
          layer.ffnGate(),
          batch.ffnUp,
          layer.ffnUp(),
          batch.normalized,
          batchSize,
          sessionState.scratch,
          batch.projectionScratch);
      for (int index = 0; index < batchSize; index++) {
        int offset = index * config.hiddenDim();
        TensorOps.swiGlu(
            batch.ffn, offset, batch.ffnGate, offset, batch.ffnUp, offset, config.hiddenDim());
      }
      projectBatched(
          batch.projected,
          batch.ffn,
          layer.ffnDown(),
          batchSize,
          sessionState.scratch,
          batch.projectionScratch);
      addBatch(batch.state, batch.projected, batchSize * dimension);
    }

    for (int index = 0; index < batchSize; index++) {
      session.tokenHistory[startPosition + index] = tokens[tokenOffset + index];
    }
    session.checkpoint += batchSize;
  }

  private void fullAttentionBatched(
      float[] output,
      float[] input,
      FullAttention weights,
      int layer,
      int startPosition,
      int batchSize,
      SessionState session,
      BatchScratch batch) {
    int headDimension = config.attentionHeadDim();
    int queryHeads = config.numHeads();
    int kvHeads = config.numKvHeads();
    int queryDimension = config.attentionQueryDim();
    int keyDimension = config.attentionKeyDim();
    tripleProjectBatched(
        batch.queryGate,
        weights.queryGate(),
        batch.key,
        weights.key(),
        batch.value,
        weights.value(),
        input,
        batchSize,
        session.scratch,
        batch.projectionScratch);

    rotary.prepareBatch(startPosition, batchSize);
    for (int token = 0; token < batchSize; token++) {
      int queryGateBase = token * 2 * queryDimension;
      int queryBase = token * queryDimension;
      int keyBase = token * keyDimension;
      for (int head = 0; head < queryHeads; head++) {
        int source = queryGateBase + head * 2 * headDimension;
        int destination = queryBase + head * headDimension;
        System.arraycopy(batch.queryGate, source, batch.query, destination, headDimension);
        TensorOps.rmsNorm(
            batch.query,
            destination,
            batch.query,
            destination,
            weights.queryNorm(),
            headDimension,
            config.rmsNormEpsilon());
        rotary.applyBatch(batch.query, destination, token, true);
      }
      for (int head = 0; head < kvHeads; head++) {
        int offset = keyBase + head * headDimension;
        TensorOps.rmsNorm(
            batch.key,
            offset,
            batch.key,
            offset,
            weights.keyNorm(),
            headDimension,
            config.rmsNormEpsilon());
        rotary.applyBatch(batch.key, offset, token, true);
      }
      int position = startPosition + token;
      System.arraycopy(
          batch.key, keyBase, session.keys[layer], position * keyDimension, keyDimension);
      System.arraycopy(
          batch.value, keyBase, session.values[layer], position * keyDimension, keyDimension);
    }

    int queriesPerKvHead = queryHeads / kvHeads;
    float scale = (float) (1.0 / Math.sqrt(headDimension));
    for (int token = 0; token < batchSize; token++) {
      int position = startPosition + token;
      int queryBase = token * queryDimension;
      int queryGateBase = token * 2 * queryDimension;
      for (int head = 0; head < queryHeads; head++) {
        int kvHead = head / queriesPerKvHead;
        int queryOffset = queryBase + head * headDimension;
        int gate = queryGateBase + head * 2 * headDimension + headDimension;
        for (int prior = 0; prior <= position; prior++) {
          int keyOffset = prior * keyDimension + kvHead * headDimension;
          float score = 0.0f;
          for (int column = 0; column < headDimension; column++) {
            score += batch.query[queryOffset + column] * session.keys[layer][keyOffset + column];
          }
          batch.attentionScores[prior] = score * scale;
        }
        TensorOps.softmax(batch.attentionScores, 0, position + 1);
        for (int column = 0; column < headDimension; column++) {
          float sum = 0.0f;
          for (int prior = 0; prior <= position; prior++) {
            int valueOffset = prior * keyDimension + kvHead * headDimension;
            sum += batch.attentionScores[prior] * session.values[layer][valueOffset + column];
          }
          batch.attended[queryOffset + column] = sum * sigmoid(batch.queryGate[gate + column]);
        }
      }
    }
    projectBatched(
        output,
        batch.attended,
        weights.output(),
        batchSize,
        session.scratch,
        batch.projectionScratch);
  }

  private void gatedDeltaNetBatched(
      float[] output,
      float[] input,
      GatedDeltaNet weights,
      int layer,
      int batchSize,
      SessionState session,
      BatchScratch batch) {
    int convDimension = config.gdnConvDim();
    int keyDimension = config.gdnKeyDim();
    int valueDimension = config.gdnValueDim();
    int valueHeads = config.gdnValueHeads();
    int headDimension = config.gdnHeadDim();
    projectBatched(
        batch.mixed,
        input,
        weights.queryKeyValue(),
        batchSize,
        session.scratch,
        batch.projectionScratch);
    projectBatched(
        batch.outputGate,
        input,
        weights.outputGate(),
        batchSize,
        session.scratch,
        batch.projectionScratch);
    dualProjectBatched(
        batch.beta,
        weights.beta(),
        batch.alpha,
        weights.alpha(),
        input,
        batchSize,
        session.scratch,
        batch.projectionScratch);

    int kernel = config.gdnConvKernel();
    int historyLength = kernel - 1;
    float[] history = session.convolutionHistory[layer];
    for (int token = 0; token < batchSize; token++) {
      int mixedBase = token * convDimension;
      for (int channel = 0; channel < convDimension; channel++) {
        int historyOffset = channel * historyLength;
        int kernelOffset = channel * kernel;
        float convolved =
            batch.mixed[mixedBase + channel] * weights.convolution()[kernelOffset + historyLength];
        for (int tap = 0; tap < historyLength; tap++) {
          convolved += history[historyOffset + tap] * weights.convolution()[kernelOffset + tap];
        }
        if (historyLength > 1) {
          System.arraycopy(history, historyOffset + 1, history, historyOffset, historyLength - 1);
        }
        history[historyOffset + historyLength - 1] = batch.mixed[mixedBase + channel];
        batch.mixed[mixedBase + channel] = silu(convolved);
      }
      System.arraycopy(batch.mixed, mixedBase, batch.gdnQuery, token * keyDimension, keyDimension);
      System.arraycopy(
          batch.mixed, mixedBase + keyDimension, batch.gdnKey, token * keyDimension, keyDimension);
      System.arraycopy(
          batch.mixed,
          mixedBase + 2 * keyDimension,
          batch.gdnValue,
          token * valueDimension,
          valueDimension);
      int gateBase = token * valueHeads;
      for (int head = 0; head < valueHeads; head++) {
        batch.beta[gateBase + head] = sigmoid(batch.beta[gateBase + head]);
        batch.logDecay[gateBase + head] =
            weights.decay()[head]
                * softplus(batch.alpha[gateBase + head] + weights.timeStepBias()[head]);
      }
    }

    GatedDeltaNetRecurrence.forwardPrefixInPlace(
        batch.gdnQuery,
        batch.gdnKey,
        batch.gdnValue,
        batch.logDecay,
        batch.beta,
        session.recurrentState[layer],
        batch.recurrent,
        batch.normalizedQuery,
        batch.normalizedKey,
        batch.memory,
        batch.delta,
        batchSize,
        config.gdnKeyHeads(),
        valueHeads,
        headDimension,
        headDimension);
    for (int token = 0; token < batchSize; token++) {
      int valueBase = token * valueDimension;
      for (int head = 0; head < valueHeads; head++) {
        int offset = valueBase + head * headDimension;
        TensorOps.rmsNorm(
            batch.recurrent,
            offset,
            batch.recurrent,
            offset,
            weights.outputNorm(),
            headDimension,
            config.rmsNormEpsilon());
        for (int column = 0; column < headDimension; column++) {
          batch.recurrent[offset + column] *= silu(batch.outputGate[offset + column]);
        }
      }
    }
    projectBatched(
        output,
        batch.recurrent,
        weights.output(),
        batchSize,
        session.scratch,
        batch.projectionScratch);
  }

  private void fullAttention(
      float[] output,
      float[] input,
      FullAttention weights,
      int position,
      float[] keyCache,
      float[] valueCache,
      ProjectionScratch scratch) {
    int headDimension = config.attentionHeadDim();
    int queryHeads = config.numHeads();
    int kvHeads = config.numKvHeads();
    int queryDimension = config.attentionQueryDim();
    float[] queryGate = new float[2 * queryDimension];
    float[] query = new float[queryDimension];
    float[] key = new float[config.attentionKeyDim()];
    float[] value = new float[config.attentionKeyDim()];
    float[] attended = new float[queryDimension];
    tripleProject(
        queryGate, weights.queryGate(), key, weights.key(), value, weights.value(), input, scratch);

    for (int head = 0; head < queryHeads; head++) {
      int sourceOffset = head * 2 * headDimension;
      int queryOffset = head * headDimension;
      System.arraycopy(queryGate, sourceOffset, query, queryOffset, headDimension);
      TensorOps.rmsNorm(
          query,
          queryOffset,
          query,
          queryOffset,
          weights.queryNorm(),
          headDimension,
          config.rmsNormEpsilon());
    }
    for (int head = 0; head < kvHeads; head++) {
      int keyOffset = head * headDimension;
      TensorOps.rmsNorm(
          key,
          keyOffset,
          key,
          keyOffset,
          weights.keyNorm(),
          headDimension,
          config.rmsNormEpsilon());
    }

    rotary.prepare(position);
    for (int head = 0; head < queryHeads; head++) {
      rotary.apply(query, head * headDimension, true);
    }
    for (int head = 0; head < kvHeads; head++) {
      rotary.apply(key, head * headDimension, true);
    }
    System.arraycopy(key, 0, keyCache, position * key.length, key.length);
    System.arraycopy(value, 0, valueCache, position * value.length, value.length);

    int queriesPerKvHead = queryHeads / kvHeads;
    float scale = (float) (1.0 / Math.sqrt(headDimension));
    float[] scores = new float[position + 1];
    for (int head = 0; head < queryHeads; head++) {
      int kvHead = head / queriesPerKvHead;
      int queryOffset = head * headDimension;
      int destination = head * headDimension;
      int gate = head * 2 * headDimension + headDimension;
      for (int prior = 0; prior <= position; prior++) {
        int keyOffset = prior * key.length + kvHead * headDimension;
        float score = 0.0f;
        for (int column = 0; column < headDimension; column++) {
          score += query[queryOffset + column] * keyCache[keyOffset + column];
        }
        scores[prior] = score * scale;
      }
      TensorOps.softmax(scores, 0, scores.length);
      for (int column = 0; column < headDimension; column++) {
        float sum = 0.0f;
        for (int prior = 0; prior <= position; prior++) {
          int valueOffset = prior * value.length + kvHead * headDimension;
          sum += scores[prior] * valueCache[valueOffset + column];
        }
        attended[destination + column] = sum * sigmoid(queryGate[gate + column]);
      }
    }
    project(output, attended, weights.output(), scratch);
  }

  private void gatedDeltaNet(
      float[] output, float[] input, GatedDeltaNet weights, int layer, SessionState session) {
    int convDimension = config.gdnConvDim();
    int keyDimension = config.gdnKeyDim();
    int valueDimension = config.gdnValueDim();
    int heads = config.gdnValueHeads();
    int headDimension = config.gdnHeadDim();
    GatedDeltaNetScratch workspace = session.gatedDeltaNetScratch;
    float[] mixed = workspace.mixed;
    float[] outputGate = workspace.outputGate;
    float[] beta = workspace.beta;
    float[] alpha = workspace.alpha;
    project(mixed, input, weights.queryKeyValue(), session.scratch);
    project(outputGate, input, weights.outputGate(), session.scratch);
    dualProject(beta, weights.beta(), alpha, weights.alpha(), input, session.scratch);

    int kernel = config.gdnConvKernel();
    int historyLength = kernel - 1;
    float[] history = session.convolutionHistory[layer];
    for (int channel = 0; channel < convDimension; channel++) {
      int historyOffset = channel * historyLength;
      int kernelOffset = channel * kernel;
      float convolved = mixed[channel] * weights.convolution()[kernelOffset + historyLength];
      for (int tap = 0; tap < historyLength; tap++) {
        convolved += history[historyOffset + tap] * weights.convolution()[kernelOffset + tap];
      }
      if (historyLength > 1) {
        System.arraycopy(history, historyOffset + 1, history, historyOffset, historyLength - 1);
      }
      history[historyOffset + historyLength - 1] = mixed[channel];
      mixed[channel] = silu(convolved);
    }
    float[] query = workspace.query;
    float[] key = workspace.key;
    float[] value = workspace.value;
    System.arraycopy(mixed, 0, query, 0, keyDimension);
    System.arraycopy(mixed, keyDimension, key, 0, keyDimension);
    System.arraycopy(mixed, 2 * keyDimension, value, 0, valueDimension);
    float[] logDecay = workspace.logDecay;
    for (int head = 0; head < heads; head++) {
      beta[head] = sigmoid(beta[head]);
      logDecay[head] = weights.decay()[head] * softplus(alpha[head] + weights.timeStepBias()[head]);
    }

    float[] recurrent = workspace.recurrent;
    GatedDeltaNetRecurrence.forwardInPlace(
        query,
        key,
        value,
        logDecay,
        beta,
        session.recurrentState[layer],
        recurrent,
        workspace.normalizedQuery,
        workspace.normalizedKey,
        workspace.memory,
        workspace.delta,
        1,
        config.gdnKeyHeads(),
        heads,
        headDimension,
        headDimension);
    for (int head = 0; head < heads; head++) {
      int offset = head * headDimension;
      TensorOps.rmsNorm(
          recurrent,
          offset,
          recurrent,
          offset,
          weights.outputNorm(),
          headDimension,
          config.rmsNormEpsilon());
      for (int column = 0; column < headDimension; column++) {
        recurrent[offset + column] *= silu(outputGate[offset + column]);
      }
    }
    project(output, recurrent, weights.output(), session.scratch);
  }

  private static final class SessionState {
    private final float[][] keys;
    private final float[][] values;
    private final float[][] convolutionHistory;
    private final float[][] recurrentState;
    private final ProjectionScratch scratch;
    private final GatedDeltaNetScratch gatedDeltaNetScratch;
    private final BatchScratch batchScratch;

    private SessionState(
        Qwen35Config config, Qwen35Weights weights, int capacity, int prefillBatchSize) {
      keys = new float[config.numLayers()][];
      values = new float[config.numLayers()][];
      convolutionHistory = new float[config.numLayers()][];
      recurrentState = new float[config.numLayers()][];
      scratch = new ProjectionScratch(config);
      gatedDeltaNetScratch = new GatedDeltaNetScratch(config);
      batchScratch = new BatchScratch(config, weights, capacity, prefillBatchSize);
      for (int layer = 0; layer < config.numLayers(); layer++) {
        if (config.usesFullAttention(layer)) {
          keys[layer] = new float[Math.multiplyExact(capacity, config.attentionKeyDim())];
          values[layer] = new float[Math.multiplyExact(capacity, config.attentionKeyDim())];
        } else {
          convolutionHistory[layer] =
              new float[Math.multiplyExact(config.gdnConvDim(), config.gdnConvKernel() - 1)];
          recurrentState[layer] =
              new float
                  [Math.multiplyExact(
                      Math.multiplyExact(config.gdnValueHeads(), config.gdnHeadDim()),
                      config.gdnHeadDim())];
        }
      }
    }
  }

  private static final class BatchScratch {
    private final float[] rowState;
    private final float[] state;
    private final float[] normalized;
    private final float[] projected;
    private final float[] ffnGate;
    private final float[] ffnUp;
    private final float[] ffn;
    private final float[] queryGate;
    private final float[] query;
    private final float[] key;
    private final float[] value;
    private final float[] attended;
    private final float[] attentionScores;
    private final float[] mixed;
    private final float[] outputGate;
    private final float[] beta;
    private final float[] alpha;
    private final float[] logDecay;
    private final float[] gdnQuery;
    private final float[] gdnKey;
    private final float[] gdnValue;
    private final float[] recurrent;
    private final float[] normalizedQuery;
    private final float[] normalizedKey;
    private final float[] memory;
    private final float[] delta;
    private final BatchProjectionScratch projectionScratch;

    private BatchScratch(Qwen35Config config, Qwen35Weights weights, int capacity, int batchSize) {
      rowState = new float[config.embeddingDim()];
      state = batchBuffer(batchSize, config.embeddingDim());
      normalized = batchBuffer(batchSize, config.embeddingDim());
      projected = batchBuffer(batchSize, config.embeddingDim());
      ffnGate = batchBuffer(batchSize, config.hiddenDim());
      ffnUp = batchBuffer(batchSize, config.hiddenDim());
      ffn = batchBuffer(batchSize, config.hiddenDim());
      queryGate = batchBuffer(batchSize, 2 * config.attentionQueryDim());
      query = batchBuffer(batchSize, config.attentionQueryDim());
      key = batchBuffer(batchSize, config.attentionKeyDim());
      value = batchBuffer(batchSize, config.attentionKeyDim());
      attended = batchBuffer(batchSize, config.attentionQueryDim());
      attentionScores = new float[capacity];
      mixed = batchBuffer(batchSize, config.gdnConvDim());
      outputGate = batchBuffer(batchSize, config.gdnValueDim());
      beta = batchBuffer(batchSize, config.gdnValueHeads());
      alpha = batchBuffer(batchSize, config.gdnValueHeads());
      logDecay = batchBuffer(batchSize, config.gdnValueHeads());
      gdnQuery = batchBuffer(batchSize, config.gdnKeyDim());
      gdnKey = batchBuffer(batchSize, config.gdnKeyDim());
      gdnValue = batchBuffer(batchSize, config.gdnValueDim());
      recurrent = batchBuffer(batchSize, config.gdnValueDim());
      normalizedQuery = new float[config.gdnHeadDim()];
      normalizedKey = new float[config.gdnHeadDim()];
      memory = new float[config.gdnHeadDim()];
      delta = new float[config.gdnHeadDim()];
      projectionScratch = new BatchProjectionScratch(config, weights, batchSize);
    }
  }

  private static final class BatchProjectionScratch {
    private final byte[] quantizedActivations;
    private final float[] quantizedActivationScales;
    private final int[] quantizedActivationZeroPointCorrections;
    private final short[] quantizedActivationSums;
    private final float[] q4LaneScratch;

    private BatchProjectionScratch(Qwen35Config config, Qwen35Weights weights, int batchSize) {
      int maximumInput =
          Math.max(config.hiddenDim(), Math.max(config.attentionQueryDim(), config.gdnValueDim()));
      int maximumRows =
          Math.max(
              Math.max(2 * config.attentionQueryDim(), config.gdnConvDim()),
              Math.max(config.hiddenDim(), config.embeddingDim()));
      quantizedActivations = new byte[Math.multiplyExact(batchSize, maximumInput)];
      quantizedActivationScales =
          new float[Math.multiplyExact(batchSize, (maximumInput + 31) / 32)];
      quantizedActivationZeroPointCorrections =
          new int[Math.multiplyExact(batchSize, (maximumInput + 3) / 4)];
      quantizedActivationSums = new short[Math.multiplyExact(batchSize, (maximumInput + 15) / 16)];
      q4LaneScratch =
          weights.usesMatrixType(GgufTensorType.Q4_0)
              ? new float[Math.multiplyExact(Math.multiplyExact(batchSize, maximumRows), 8)]
              : new float[0];
    }
  }

  private static final class GatedDeltaNetScratch {
    private final float[] mixed;
    private final float[] outputGate;
    private final float[] beta;
    private final float[] alpha;
    private final float[] query;
    private final float[] key;
    private final float[] value;
    private final float[] logDecay;
    private final float[] recurrent;
    private final float[] normalizedQuery;
    private final float[] normalizedKey;
    private final float[] memory;
    private final float[] delta;

    private GatedDeltaNetScratch(Qwen35Config config) {
      mixed = new float[config.gdnConvDim()];
      outputGate = new float[config.gdnValueDim()];
      beta = new float[config.gdnValueHeads()];
      alpha = new float[config.gdnValueHeads()];
      query = new float[config.gdnKeyDim()];
      key = new float[config.gdnKeyDim()];
      value = new float[config.gdnValueDim()];
      logDecay = new float[config.gdnValueHeads()];
      recurrent = new float[config.gdnValueDim()];
      normalizedQuery = new float[config.gdnHeadDim()];
      normalizedKey = new float[config.gdnHeadDim()];
      memory = new float[config.gdnHeadDim()];
      delta = new float[config.gdnHeadDim()];
    }
  }

  private static final class ProjectionScratch {
    private final byte[] quantizedActivation;
    private final float[] quantizedActivationScales;
    private final int[] quantizedActivationZeroPointCorrections;
    private final short[] quantizedActivationSums;

    private ProjectionScratch(Qwen35Config config) {
      int maximumProjectionInput =
          Math.max(config.hiddenDim(), Math.max(config.attentionQueryDim(), config.gdnValueDim()));
      quantizedActivation = new byte[maximumProjectionInput];
      quantizedActivationScales = new float[(maximumProjectionInput + 31) / 32];
      quantizedActivationZeroPointCorrections = new int[(maximumProjectionInput + 3) / 4];
      quantizedActivationSums = new short[(maximumProjectionInput + 15) / 16];
    }
  }

  private void project(float[] output, float[] input, Matrix weights, ProjectionScratch scratch) {
    TensorOps.ggufMatmul(
        output,
        input,
        weights.data(),
        weights.type(),
        weights.rows(),
        weights.columns(),
        scratch.quantizedActivation,
        scratch.quantizedActivationScales,
        scratch.quantizedActivationZeroPointCorrections,
        scratch.quantizedActivationSums,
        q4Kernel);
  }

  private void dualProject(
      float[] firstOutput,
      Matrix firstWeights,
      float[] secondOutput,
      Matrix secondWeights,
      float[] input,
      ProjectionScratch scratch) {
    if (firstWeights.columns() != secondWeights.columns()) {
      throw new IllegalArgumentException("grouped projections must share their input width");
    }
    TensorOps.ggufDualMatmul(
        firstOutput,
        firstWeights.data(),
        firstWeights.type(),
        firstWeights.rows(),
        secondOutput,
        secondWeights.data(),
        secondWeights.type(),
        secondWeights.rows(),
        input,
        firstWeights.columns(),
        scratch.quantizedActivation,
        scratch.quantizedActivationScales,
        scratch.quantizedActivationZeroPointCorrections,
        scratch.quantizedActivationSums,
        q4Kernel);
  }

  private void tripleProject(
      float[] firstOutput,
      Matrix firstWeights,
      float[] secondOutput,
      Matrix secondWeights,
      float[] thirdOutput,
      Matrix thirdWeights,
      float[] input,
      ProjectionScratch scratch) {
    if (firstWeights.columns() != secondWeights.columns()
        || firstWeights.columns() != thirdWeights.columns()) {
      throw new IllegalArgumentException("grouped projections must share their input width");
    }
    TensorOps.ggufTripleMatmul(
        firstOutput,
        firstWeights.data(),
        firstWeights.type(),
        firstWeights.rows(),
        secondOutput,
        secondWeights.data(),
        secondWeights.type(),
        secondWeights.rows(),
        thirdOutput,
        thirdWeights.data(),
        thirdWeights.type(),
        thirdWeights.rows(),
        input,
        firstWeights.columns(),
        scratch.quantizedActivation,
        scratch.quantizedActivationScales,
        scratch.quantizedActivationZeroPointCorrections,
        scratch.quantizedActivationSums,
        q4Kernel);
  }

  private void projectBatched(
      float[] output,
      float[] input,
      Matrix weights,
      int batchSize,
      ProjectionScratch scalarScratch,
      BatchProjectionScratch batchScratch) {
    if (!TensorOps.supportsBatchedMatmul(weights.type())) {
      float[] rowInput = new float[weights.columns()];
      float[] rowOutput = new float[weights.rows()];
      for (int batch = 0; batch < batchSize; batch++) {
        System.arraycopy(input, batch * weights.columns(), rowInput, 0, weights.columns());
        project(rowOutput, rowInput, weights, scalarScratch);
        System.arraycopy(rowOutput, 0, output, batch * weights.rows(), weights.rows());
      }
      return;
    }
    TensorOps.ggufBatchedMatmul(
        output,
        input,
        weights.data(),
        weights.type(),
        batchSize,
        weights.rows(),
        weights.columns(),
        batchScratch.quantizedActivations,
        batchScratch.quantizedActivationScales,
        batchScratch.quantizedActivationZeroPointCorrections,
        batchScratch.quantizedActivationSums,
        batchScratch.q4LaneScratch,
        q4Kernel,
        q6BatchedKernel);
  }

  private void dualProjectBatched(
      float[] firstOutput,
      Matrix firstWeights,
      float[] secondOutput,
      Matrix secondWeights,
      float[] input,
      int batchSize,
      ProjectionScratch scalarScratch,
      BatchProjectionScratch batchScratch) {
    if (firstWeights.columns() != secondWeights.columns()) {
      throw new IllegalArgumentException("grouped projections must share their input width");
    }
    if (!TensorOps.supportsBatchedMatmul(firstWeights.type())
        || !TensorOps.supportsBatchedMatmul(secondWeights.type())) {
      projectBatched(firstOutput, input, firstWeights, batchSize, scalarScratch, batchScratch);
      projectBatched(secondOutput, input, secondWeights, batchSize, scalarScratch, batchScratch);
      return;
    }
    TensorOps.ggufDualBatchedMatmul(
        firstOutput,
        firstWeights.data(),
        firstWeights.type(),
        firstWeights.rows(),
        secondOutput,
        secondWeights.data(),
        secondWeights.type(),
        secondWeights.rows(),
        input,
        batchSize,
        firstWeights.columns(),
        batchScratch.quantizedActivations,
        batchScratch.quantizedActivationScales,
        batchScratch.quantizedActivationZeroPointCorrections,
        batchScratch.quantizedActivationSums,
        batchScratch.q4LaneScratch,
        q4Kernel,
        q6BatchedKernel);
  }

  private void tripleProjectBatched(
      float[] firstOutput,
      Matrix firstWeights,
      float[] secondOutput,
      Matrix secondWeights,
      float[] thirdOutput,
      Matrix thirdWeights,
      float[] input,
      int batchSize,
      ProjectionScratch scalarScratch,
      BatchProjectionScratch batchScratch) {
    if (firstWeights.columns() != secondWeights.columns()
        || firstWeights.columns() != thirdWeights.columns()) {
      throw new IllegalArgumentException("grouped projections must share their input width");
    }
    if (!TensorOps.supportsBatchedMatmul(firstWeights.type())
        || !TensorOps.supportsBatchedMatmul(secondWeights.type())
        || !TensorOps.supportsBatchedMatmul(thirdWeights.type())) {
      projectBatched(firstOutput, input, firstWeights, batchSize, scalarScratch, batchScratch);
      projectBatched(secondOutput, input, secondWeights, batchSize, scalarScratch, batchScratch);
      projectBatched(thirdOutput, input, thirdWeights, batchSize, scalarScratch, batchScratch);
      return;
    }
    TensorOps.ggufTripleBatchedMatmul(
        firstOutput,
        firstWeights.data(),
        firstWeights.type(),
        firstWeights.rows(),
        secondOutput,
        secondWeights.data(),
        secondWeights.type(),
        secondWeights.rows(),
        thirdOutput,
        thirdWeights.data(),
        thirdWeights.type(),
        thirdWeights.rows(),
        input,
        batchSize,
        firstWeights.columns(),
        batchScratch.quantizedActivations,
        batchScratch.quantizedActivationScales,
        batchScratch.quantizedActivationZeroPointCorrections,
        batchScratch.quantizedActivationSums,
        batchScratch.q4LaneScratch,
        q4Kernel,
        q6BatchedKernel,
        false);
  }

  private void normalizeBatch(
      float[] output, float[] input, int batchSize, int width, float[] normalizationWeight) {
    for (int batch = 0; batch < batchSize; batch++) {
      int offset = batch * width;
      TensorOps.rmsNorm(
          output, offset, input, offset, normalizationWeight, width, config.rmsNormEpsilon());
    }
  }

  private static float[] batchBuffer(int batchSize, int width) {
    return new float[Math.multiplyExact(batchSize, width)];
  }

  private static void addBatch(float[] destination, float[] source, int activeElements) {
    for (int index = 0; index < activeElements; index++) {
      destination[index] += source[index];
    }
  }

  private static void add(float[] destination, float[] source) {
    for (int index = 0; index < destination.length; index++) {
      destination[index] += source[index];
    }
  }

  private static float sigmoid(float value) {
    if (value >= 0.0f) {
      float exponential = (float) Math.exp(-value);
      return 1.0f / (1.0f + exponential);
    }
    float exponential = (float) Math.exp(value);
    return exponential / (1.0f + exponential);
  }

  private static float silu(float value) {
    return value * sigmoid(value);
  }

  private static float softplus(float value) {
    if (value > 20.0f) {
      return value;
    }
    if (value < -20.0f) {
      return (float) Math.exp(value);
    }
    return (float) Math.log1p(Math.exp(value));
  }
}
