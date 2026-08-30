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
import com.integrallis.models.backend.purejava.ops.RotaryTable;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.models.backend.purejava.qwen35.Qwen35Weights.FullAttention;
import com.integrallis.models.backend.purejava.qwen35.Qwen35Weights.GatedDeltaNet;
import com.integrallis.models.backend.purejava.qwen35.Qwen35Weights.Layer;
import com.integrallis.models.backend.purejava.qwen35.Qwen35Weights.Matrix;
import java.util.Arrays;
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
      this.state = new SessionState(owner.config, capacity);
    }

    public int checkpoint() {
      return checkpoint;
    }
  }

  private final Qwen35Config config;
  private final Qwen35Weights weights;
  private final RotaryTable rotary;

  private Qwen35ForwardPass(Qwen35Config config, Qwen35Weights weights) {
    this.config = config;
    this.weights = weights;
    this.rotary = new RotaryTable(config.ropeDimension(), config.ropeTheta(), 1.0f);
  }

  public static Qwen35ForwardPass fromGgufFile(GgufFile file) {
    Objects.requireNonNull(file, "file");
    Qwen35Config config = Qwen35Config.fromMetadata(file.metadata());
    return new Qwen35ForwardPass(config, Qwen35Weights.fromGgufFile(file, config));
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
    for (int index = 0; index < tokens.length - 1; index++) {
      advance(checked, tokens[index], false);
    }
    return advance(checked, tokens[tokens.length - 1], true);
  }

  /** Reconstructs sequence state at an earlier checkpoint from its retained token prefix. */
  public void rewind(Session session, int checkpoint) {
    Session checked = requireSession(session);
    if (checkpoint < 0 || checkpoint > checked.checkpoint) {
      throw new IllegalArgumentException(
          "checkpoint must be between 0 and " + checked.checkpoint + ": " + checkpoint);
    }
    checked.state = new SessionState(config, checked.capacity);
    checked.checkpoint = 0;
    for (int index = 0; index < checkpoint; index++) {
      advance(checked, checked.tokenHistory[index], false);
    }
  }

  /** Clears all sequence state. */
  public void reset(Session session) {
    Session checked = requireSession(session);
    checked.state = new SessionState(config, checked.capacity);
    checked.checkpoint = 0;
  }

  public Qwen35Config config() {
    return config;
  }

  private float[] advance(Session session, int token, boolean projectLogits) {
    if (session.checkpoint >= session.capacity) {
      throw new IllegalStateException("session context capacity exhausted: " + session.capacity);
    }
    int position = session.checkpoint;
    float[] hidden = forwardToken(token, position, session.state);
    session.tokenHistory[position] = token;
    session.checkpoint++;
    return projectLogits ? projectLogits(hidden) : null;
  }

  private float[] projectLogits(float[] hidden) {
    float[] normalized = new float[config.embeddingDim()];
    TensorOps.rmsNorm(
        normalized, hidden, weights.outputNorm(), config.embeddingDim(), config.rmsNormEpsilon());
    float[] logits = new float[config.vocabSize()];
    project(logits, normalized, weights.output());
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
            session.values[layerIndex]);
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
      project(ffnGate, normalized, layer.ffnGate());
      project(ffnUp, normalized, layer.ffnUp());
      TensorOps.swiGlu(ffn, ffnGate, ffnUp, config.hiddenDim());
      project(projected, ffn, layer.ffnDown());
      add(state, projected);
    }

    return state;
  }

  private void fullAttention(
      float[] output,
      float[] input,
      FullAttention weights,
      int position,
      float[] keyCache,
      float[] valueCache) {
    int headDimension = config.attentionHeadDim();
    int queryHeads = config.numHeads();
    int kvHeads = config.numKvHeads();
    int queryDimension = config.attentionQueryDim();
    float[] queryGate = new float[2 * queryDimension];
    float[] query = new float[queryDimension];
    float[] key = new float[config.attentionKeyDim()];
    float[] value = new float[config.attentionKeyDim()];
    float[] attended = new float[queryDimension];
    project(queryGate, input, weights.queryGate());
    project(key, input, weights.key());
    project(value, input, weights.value());

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
    project(output, attended, weights.output());
  }

  private void gatedDeltaNet(
      float[] output, float[] input, GatedDeltaNet weights, int layer, SessionState session) {
    int convDimension = config.gdnConvDim();
    int keyDimension = config.gdnKeyDim();
    int valueDimension = config.gdnValueDim();
    int heads = config.gdnValueHeads();
    int headDimension = config.gdnHeadDim();
    float[] mixed = new float[convDimension];
    float[] outputGate = new float[valueDimension];
    float[] beta = new float[heads];
    float[] alpha = new float[heads];
    project(mixed, input, weights.queryKeyValue());
    project(outputGate, input, weights.outputGate());
    project(beta, input, weights.beta());
    project(alpha, input, weights.alpha());

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
    float[] query = Arrays.copyOfRange(mixed, 0, keyDimension);
    float[] key = Arrays.copyOfRange(mixed, keyDimension, 2 * keyDimension);
    float[] value = Arrays.copyOfRange(mixed, 2 * keyDimension, convDimension);
    float[] logDecay = new float[heads];
    for (int head = 0; head < heads; head++) {
      beta[head] = sigmoid(beta[head]);
      logDecay[head] = weights.decay()[head] * softplus(alpha[head] + weights.timeStepBias()[head]);
    }

    GatedDeltaNetRecurrence.Result recurrentResult =
        GatedDeltaNetRecurrence.forward(
            query,
            key,
            value,
            logDecay,
            beta,
            session.recurrentState[layer],
            1,
            heads,
            headDimension,
            headDimension);
    session.recurrentState[layer] = recurrentResult.finalState();
    float[] recurrent = recurrentResult.output();
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
    project(output, recurrent, weights.output());
  }

  private static final class SessionState {
    private final float[][] keys;
    private final float[][] values;
    private final float[][] convolutionHistory;
    private final float[][] recurrentState;

    private SessionState(Qwen35Config config, int capacity) {
      keys = new float[config.numLayers()][];
      values = new float[config.numLayers()][];
      convolutionHistory = new float[config.numLayers()][];
      recurrentState = new float[config.numLayers()][];
      for (int layer = 0; layer < config.numLayers(); layer++) {
        if (config.usesFullAttention(layer)) {
          keys[layer] = new float[Math.multiplyExact(capacity, config.attentionKeyDim())];
          values[layer] = new float[Math.multiplyExact(capacity, config.attentionKeyDim())];
        } else {
          convolutionHistory[layer] =
              new float[Math.multiplyExact(config.gdnConvDim(), config.gdnConvKernel() - 1)];
        }
      }
    }
  }

  private static void project(float[] output, float[] input, Matrix weights) {
    TensorOps.ggufMatmul(
        output, input, weights.data(), weights.type(), weights.rows(), weights.columns());
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
