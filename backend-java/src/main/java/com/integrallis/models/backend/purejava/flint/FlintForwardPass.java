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
import com.integrallis.models.backend.purejava.cache.LayeredKvCache.AttentionView;
import com.integrallis.models.backend.purejava.cache.LayeredKvCache.LayerSpec;
import com.integrallis.models.backend.purejava.flint.FlintWeights.LayerWeights;
import com.integrallis.models.backend.purejava.flint.FlintWeights.Matrix;
import com.integrallis.models.backend.purejava.flint.FlintWeights.MoeFeedForward;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.ops.RotaryTable;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import com.integrallis.vectors.core.GgufQ4Kernel;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Objects;

/** Flint decoder with mapped GGUF experts and independent, bounded sliding-window sessions. */
public final class FlintForwardPass {

  /** Mutable sequence state; sessions share immutable mapped weights and are otherwise isolated. */
  public static final class Session {
    private final FlintForwardPass owner;
    private final LayeredKvCache cache;
    private final RotaryTable rope;
    private final byte[] quantizedActivation;
    private final float[] quantizedActivationScales;
    private final int[] quantizedActivationCorrections;
    private final short[] quantizedActivationSums;
    private final float[] gate;
    private final float[] up;
    private final float[] contribution;
    private final float[] x;
    private final float[] xNorm;
    private final float[] query;
    private final float[] key;
    private final float[] value;
    private final float[] attentionOutput;
    private final float[][] attentionScores;
    private final float[] projectedAttention;
    private final float[] routerLogits;
    private final int[] selectedExperts;
    private final float[] routingWeights;
    private final float[] moeOutput;
    private final MemorySegment[] expertGateUpWeights;
    private final GgufTensorType[] expertGateUpTypes;
    private final int[] expertGateUpRows;
    private final float[][] expertGateUps;
    private final MemorySegment[] expertDownWeights;
    private final GgufTensorType[] expertDownTypes;
    private final int[] expertDownRows;
    private final float[][] expertActivations;
    private final float[][] expertContributions;
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
      int projectionWidth =
          Math.max(config.queryDimension(), Math.max(config.hiddenSize(), ffnWidth));
      quantizedActivation = new byte[projectionWidth];
      quantizedActivationScales = new float[(projectionWidth + 31) / 32];
      quantizedActivationCorrections = new int[(projectionWidth + 3) / 4];
      quantizedActivationSums = new short[(projectionWidth + 15) / 16];
      gate = new float[ffnWidth];
      up = new float[ffnWidth];
      contribution = new float[config.hiddenSize()];
      x = new float[config.hiddenSize()];
      xNorm = new float[config.hiddenSize()];
      query = new float[config.queryDimension()];
      key = new float[config.keyValueDimension()];
      value = new float[config.keyValueDimension()];
      attentionOutput = new float[config.queryDimension()];
      attentionScores = new float[config.numHeads()][owner.maxSequenceLength];
      projectedAttention = new float[config.hiddenSize()];
      routerLogits = new float[config.numExperts()];
      selectedExperts = new int[config.expertsPerToken()];
      routingWeights = new float[config.expertsPerToken()];
      moeOutput = new float[config.hiddenSize()];
      int experts = config.expertsPerToken();
      expertGateUpWeights = new MemorySegment[2 * experts];
      expertGateUpTypes = new GgufTensorType[2 * experts];
      expertGateUpRows = new int[2 * experts];
      expertGateUps = new float[2 * experts][config.expertHiddenSize()];
      expertDownWeights = new MemorySegment[experts];
      expertDownTypes = new GgufTensorType[experts];
      expertDownRows = new int[experts];
      expertActivations = new float[experts][];
      for (int slot = 0; slot < experts; slot++) expertActivations[slot] = expertGateUps[2 * slot];
      expertContributions = new float[experts][config.hiddenSize()];
      logits = new float[config.vocabSize()];
      tokenHistory = new int[owner.maxSequenceLength];
    }

    /** Returns the next absolute sequence position accepted by this session. */
    public int checkpoint() {
      return nextPosition;
    }
  }

  private final GgufBatchedMatrixKernel matrixKernel;
  private final FlintConfig config;
  private final FlintWeights weights;
  private final LayerWeights[] layers;
  private final LayerSpec[] cacheSpecs;
  private final float[] outputNorm;
  private final FlintRouterProjection[] routerProjections;
  private final long routerProjectionBytes;
  private static final long ROUTER_CACHE_BUDGET = 128L << 20;
  private final int maxSequenceLength;

  FlintForwardPass(
      FlintConfig config,
      FlintWeights weights,
      int maxSequenceLength,
      GgufBatchedMatrixKernel matrixKernel) {
    this.matrixKernel = Objects.requireNonNull(matrixKernel, "matrixKernel");
    // Tiny projection fold differences can change discrete MoE routing decisions.
    this.matrixKernel.requireSequentialQ4KReduction();
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
    routerProjections = new FlintRouterProjection[layers.length];
    long routerBytes = 0;
    cacheSpecs = new LayerSpec[config.numLayers()];
    for (int layer = 0; layer < layers.length; layer++) {
      layers[layer] = weights.layer(layer);
      MoeFeedForward moe = layers[layer].moe();
      if (moe != null) {
        Matrix router = moe.router();
        if (!matrixKernel.isEligible(router.type(), 1, router.rows(), router.columns())) {
          routerProjections[layer] =
              FlintRouterProjection.prepare(router, ROUTER_CACHE_BUDGET - routerBytes);
          if (routerProjections[layer] != null) routerBytes += routerProjections[layer].bytes();
        }
      }
      cacheSpecs[layer] =
          config.usesSlidingAttention(layer)
              ? LayerSpec.ring(
                  config.keyValueDimension(), config.keyValueDimension(), config.slidingWindow())
              : LayerSpec.linear(config.keyValueDimension(), config.keyValueDimension());
    }
    routerProjectionBytes = routerBytes;
  }

  /** Heap bytes retained for immutable F32 router projections, shared by all sessions. */
  public long routerProjectionBytes() {
    return routerProjectionBytes;
  }

  /** Loads the validated graph; all expert matrices stay in the mapped GGUF. */
  public static FlintForwardPass loadGguf(
      FlintConfig config,
      com.integrallis.models.backend.purejava.gguf.GgufFile file,
      int maxSequenceLength) {
    return loadGguf(config, file, maxSequenceLength, GgufBatchedMatrixKernel.none());
  }

  /** Loads mapped weights with a caller-owned optional matrix kernel. */
  public static FlintForwardPass loadGguf(
      FlintConfig config,
      com.integrallis.models.backend.purejava.gguf.GgufFile file,
      int maxSequenceLength,
      GgufBatchedMatrixKernel matrixKernel) {
    return new FlintForwardPass(
        config, FlintWeights.fromGgufFile(file, config), maxSequenceLength, matrixKernel);
  }

  /** Returns the validated checkpoint execution configuration. */
  public FlintConfig config() {
    return config;
  }

  /** Load-time eligibility, not invocation counts or a claim of measured acceleration. */
  public record ProjectionRoutes(
      String implementation, int qkvLayers, int gateUpLayers, int expertLayers) {}

  /** Describes the actual mapped tensor shapes instead of the generic architecture placeholder. */
  public ProjectionRoutes projectionRoutes() {
    int qkv = 0;
    int gateUp = 0;
    int experts = 0;
    for (LayerWeights layer : layers) {
      Matrix q = layer.query(), k = layer.key(), v = layer.value();
      if (matrixKernel.isTripleEligible(
          q.type(), q.rows(), k.type(), k.rows(), v.type(), v.rows(), 1, q.columns())) qkv++;
      MoeFeedForward moe = layer.moe();
      Matrix gate = moe == null ? layer.ffnGate() : moe.sharedGate();
      Matrix up = moe == null ? layer.ffnUp() : moe.sharedUp();
      if (matrixKernel.isDualEligible(
          gate.type(), gate.rows(), up.type(), up.rows(), 1, gate.columns())) gateUp++;
      if (moe != null) {
        int count = config.expertsPerToken();
        GgufTensorType[] gateTypes = new GgufTensorType[2 * count];
        int[] gateRows = new int[2 * count];
        GgufTensorType[] downTypes = new GgufTensorType[count];
        int[] downRows = new int[count];
        // Each family is sliced from one stacked tensor, so every expert has the same type/shape.
        for (int slot = 0; slot < count; slot++) {
          gateTypes[2 * slot] = moe.gate()[0].type();
          gateTypes[2 * slot + 1] = moe.up()[0].type();
          gateRows[2 * slot] = moe.gate()[0].rows();
          gateRows[2 * slot + 1] = moe.up()[0].rows();
          downTypes[slot] = moe.down()[0].type();
          downRows[slot] = moe.down()[0].rows();
        }
        if (matrixKernel.isGroupedEligible(gateTypes, gateRows, 2 * count, 1, config.hiddenSize())
            && matrixKernel.isIndependentEligible(
                downTypes, downRows, count, 1, config.expertHiddenSize())) experts++;
      }
    }
    return new ProjectionRoutes(matrixKernel.implementation(), qkv, gateUp, experts);
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
      multiply(session, weights.output(), session.xNorm, session.logits);
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
    multiplyTriple(
        session,
        parameters.query(),
        session.query,
        parameters.key(),
        session.key,
        parameters.value(),
        session.value,
        session.xNorm);
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
    multiply(
        session, parameters.attentionOutput(), session.attentionOutput, session.projectedAttention);
    addResidual(session.x, session.projectedAttention);

    TensorOps.rmsNorm(
        session.xNorm, session.x, parameters.ffnNorm(), config.hiddenSize(), config.rmsNormEps());
    if (parameters.moe() == null) {
      feedForward(session, parameters.ffnGate(), parameters.ffnUp(), parameters.ffnDown());
      addResidual(session.x, session.contribution);
    } else {
      var moe = parameters.moe();
      if (routerProjections[layer] != null) {
        routerProjections[layer].multiply(session.xNorm, session.routerLogits);
      } else {
        multiply(session, moe.router(), session.xNorm, session.routerLogits);
      }
      FlintRouting.select(
          session.routerLogits,
          moe.selectionBias(),
          session.selectedExperts,
          session.routingWeights,
          config.normalizeExpertWeights(),
          config.expertWeightsScale());
      Arrays.fill(session.moeOutput, 0f);
      boolean grouped = executeExpertGroup(session, moe);
      for (int slot = 0; slot < config.expertsPerToken(); slot++) {
        float[] contribution;
        if (grouped) {
          contribution = session.expertContributions[slot];
        } else {
          int expert = session.selectedExperts[slot];
          feedForward(session, moe.gate()[expert], moe.up()[expert], moe.down()[expert]);
          contribution = session.contribution;
        }
        // Accumulate in the original routing order regardless of kernel scheduling.
        for (int i = 0; i < config.hiddenSize(); i++) {
          session.moeOutput[i] += session.routingWeights[slot] * contribution[i];
        }
      }
      feedForward(session, moe.sharedGate(), moe.sharedUp(), moe.sharedDown());
      addResidual(session.moeOutput, session.contribution);
      addResidual(session.x, session.moeOutput);
    }
  }

  private boolean executeExpertGroup(Session session, MoeFeedForward moe) {
    int count = config.expertsPerToken();
    for (int slot = 0; slot < count; slot++) {
      int expert = session.selectedExperts[slot];
      Matrix gate = moe.gate()[expert];
      Matrix up = moe.up()[expert];
      Matrix down = moe.down()[expert];
      session.expertGateUpWeights[2 * slot] = gate.data();
      session.expertGateUpWeights[2 * slot + 1] = up.data();
      session.expertGateUpTypes[2 * slot] = gate.type();
      session.expertGateUpTypes[2 * slot + 1] = up.type();
      session.expertGateUpRows[2 * slot] = gate.rows();
      session.expertGateUpRows[2 * slot + 1] = up.rows();
      session.expertDownWeights[slot] = down.data();
      session.expertDownTypes[slot] = down.type();
      session.expertDownRows[slot] = down.rows();
    }
    if (!matrixKernel.isGroupedEligible(
            session.expertGateUpTypes, session.expertGateUpRows, 2 * count, 1, config.hiddenSize())
        || !matrixKernel.isIndependentEligible(
            session.expertDownTypes, session.expertDownRows, count, 1, config.expertHiddenSize()))
      return false;
    matrixKernel.multiplyGrouped(
        session.expertGateUps,
        session.expertGateUpWeights,
        session.expertGateUpTypes,
        session.expertGateUpRows,
        2 * count,
        session.xNorm,
        1,
        config.hiddenSize());
    for (int slot = 0; slot < count; slot++) {
      float[] gate = session.expertGateUps[2 * slot];
      float[] up = session.expertGateUps[2 * slot + 1];
      for (int i = 0; i < gate.length; i++) {
        float value = gate[i];
        gate[i] = (value / (1f + (float) Math.exp(-value))) * up[i];
      }
    }
    matrixKernel.multiplyIndependent(
        session.expertContributions,
        session.expertDownWeights,
        session.expertDownTypes,
        session.expertDownRows,
        count,
        session.expertActivations,
        1,
        config.expertHiddenSize());
    return true;
  }

  private void feedForward(Session session, Matrix gate, Matrix up, Matrix down) {
    multiplyDual(session, gate, session.gate, up, session.up, session.xNorm);
    for (int i = 0; i < gate.rows(); i++) {
      float value = session.gate[i];
      session.gate[i] = (value / (1f + (float) Math.exp(-value))) * session.up[i];
    }
    multiply(session, down, session.gate, session.contribution);
  }

  private void multiply(Session session, Matrix matrix, float[] input, float[] output) {
    if (matrixKernel.isEligible(matrix.type(), 1, matrix.rows(), matrix.columns())) {
      matrixKernel.multiply(
          output, input, matrix.data(), matrix.type(), 1, matrix.rows(), matrix.columns());
      return;
    }
    TensorOps.ggufMatmul(
        output,
        input,
        matrix.data(),
        matrix.type(),
        matrix.rows(),
        matrix.columns(),
        session.quantizedActivation,
        session.quantizedActivationScales,
        session.quantizedActivationCorrections,
        session.quantizedActivationSums,
        GgufQ4Kernel.WIDENED);
  }

  private void multiplyDual(
      Session session, Matrix a, float[] ao, Matrix b, float[] bo, float[] input) {
    int cols = a.columns();
    if (matrixKernel.isDualEligible(a.type(), a.rows(), b.type(), b.rows(), 1, cols)) {
      matrixKernel.multiplyDual(
          ao, a.data(), a.type(), a.rows(), bo, b.data(), b.type(), b.rows(), input, 1, cols);
    } else if (matrixKernel.isEligible(a.type(), 1, a.rows(), cols)
        || matrixKernel.isEligible(b.type(), 1, b.rows(), cols)) {
      multiply(session, a, input, ao);
      multiply(session, b, input, bo);
    } else {
      TensorOps.ggufDualMatmul(
          ao,
          a.data(),
          a.type(),
          a.rows(),
          bo,
          b.data(),
          b.type(),
          b.rows(),
          input,
          cols,
          session.quantizedActivation,
          session.quantizedActivationScales,
          session.quantizedActivationCorrections,
          session.quantizedActivationSums,
          GgufQ4Kernel.WIDENED);
    }
  }

  private void multiplyTriple(
      Session session,
      Matrix a,
      float[] ao,
      Matrix b,
      float[] bo,
      Matrix c,
      float[] co,
      float[] input) {
    int cols = a.columns();
    if (matrixKernel.isTripleEligible(
        a.type(), a.rows(), b.type(), b.rows(), c.type(), c.rows(), 1, cols)) {
      matrixKernel.multiplyTriple(
          ao, a.data(), a.type(), a.rows(), bo, b.data(), b.type(), b.rows(), co, c.data(),
          c.type(), c.rows(), input, 1, cols);
    } else if (matrixKernel.isEligible(a.type(), 1, a.rows(), cols)
        || matrixKernel.isEligible(b.type(), 1, b.rows(), cols)
        || matrixKernel.isEligible(c.type(), 1, c.rows(), cols)) {
      multiply(session, a, input, ao);
      multiply(session, b, input, bo);
      multiply(session, c, input, co);
    } else {
      TensorOps.ggufTripleMatmul(
          ao,
          a.data(),
          a.type(),
          a.rows(),
          bo,
          b.data(),
          b.type(),
          b.rows(),
          co,
          c.data(),
          c.type(),
          c.rows(),
          input,
          cols,
          session.quantizedActivation,
          session.quantizedActivationScales,
          session.quantizedActivationCorrections,
          session.quantizedActivationSums,
          GgufQ4Kernel.WIDENED);
    }
  }

  private void computeAttention(Session session, LayerWeights parameters, int layer, int position) {
    int firstPosition = config.attentionStartPosition(layer, position);
    AttentionView view = session.cache.attentionView(layer, firstPosition, position + 1);
    FlintAttention.compute(
        session.query,
        session.cache.keyBuffer(layer),
        session.cache.valueBuffer(layer),
        session.attentionScores,
        session.attentionOutput,
        parameters.sinks(),
        view,
        config.numHeads(),
        config.numKvHeads(),
        config.headDim());
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
