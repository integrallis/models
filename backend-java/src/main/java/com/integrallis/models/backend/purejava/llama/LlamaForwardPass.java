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
package com.integrallis.models.backend.purejava.llama;

import com.integrallis.models.api.LogitBatch;
import com.integrallis.models.backend.purejava.cache.KvCache;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.ops.RotaryTable;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.models.backend.purejava.plan.ExecutionPlanner;
import com.integrallis.models.backend.purejava.plan.ModelTopology;
import com.integrallis.models.backend.purejava.plan.PureJavaExecutionPlan;
import com.integrallis.models.backend.purejava.plan.PureJavaPlanConfiguration;
import com.integrallis.models.backend.purejava.plan.RuntimeFingerprint;
import com.integrallis.models.backend.purejava.spi.BatchedCausalAttentionKernel;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import com.integrallis.vectors.core.GgufQ4Kernel;
import com.integrallis.vectors.core.GgufQ6BatchedKernel;
import com.integrallis.vectors.core.GgufQ8BlockMajorKernel;
import com.integrallis.vectors.core.VectorUtil;
import java.lang.foreign.MemorySegment;
import java.util.Map;
import java.util.Objects;

/**
 * Single-token forward pass for Llama-family models. Implements the full transformer decoder
 * pipeline: embed, (RMSNorm → QKV → RoPE → GQA attention → residual → RMSNorm → SwiGLU FFN →
 * residual) per layer, final RMSNorm → output logits.
 */
public final class LlamaForwardPass {

  @FunctionalInterface
  interface LayerObserver {
    void onLayerComplete(int layer, int position, float[] state, int offset, int length);
  }

  /** Mutable sequence state sharing this forward pass's immutable model and execution plan. */
  public static final class Session {
    private final LlamaForwardPass owner;
    private final KvCache cache;
    private int nextPosition;

    private Session(LlamaForwardPass owner, KvCache cache) {
      this.owner = owner;
      this.cache = cache;
    }

    /** Returns the next sequence position. */
    public int checkpoint() {
      return nextPosition;
    }

    KvCache cache() {
      return cache;
    }
  }

  private final LlamaConfig config;
  private final LlamaWeights weights;
  private final KvCache cache;
  private final RotaryTable globalRopeTable;
  private final RotaryTable slidingWindowRopeTable;
  private final LayerObserver layerObserver;
  private final boolean groupedProjections;
  private final boolean mixedKProjections;
  private final GgufQ4Kernel q4Kernel;
  private final GgufQ6BatchedKernel q6BatchedKernel;
  private final boolean batchedPrefill;
  private final boolean groupedBatchedPrefill;
  private final boolean batchedSessionOutputProjection;
  private final boolean finalLayerPrefillPruning;
  private final boolean finalLayerKvOnlyPrefill;
  private final boolean batchedAttentionScores;
  private final boolean batchedAttentionValues;
  private final boolean stagedQuantizedFfn;
  private final boolean stagedQuantizedLayer;
  private final boolean blockMajorQ8Activations;
  private final GgufQ8BlockMajorKernel q8BlockMajorKernel;
  private final boolean parallelQ8FfnPreparation;
  private final GgufBatchedMatrixKernel batchedMatrixKernel;
  private final BatchedCausalAttentionKernel batchedAttentionKernel;
  private final QuantizedBatchedLayerPlan stagedQuantizedPlan;
  private final int prefillBatchCapacity;

  // Scratch buffers
  private final float[] x;
  private final float[] xNorm;
  private final float[] q;
  private final float[] k;
  private final float[] v;
  private final float[] attnOut;
  private final float[] attentionScores;
  private final float[] attnProjected;
  private final float[] ffnGate;
  private final float[] ffnUp;
  private final float[] ffnOut;
  private final float[] ffnProjected;
  private final float[] logits;
  private final byte[] quantizedActivation;
  private final float[] quantizedActivationScales;
  private final int[] quantizedActivationZeroPointCorrections;
  private final short[] quantizedActivationSums;
  private final float[] batchX;
  private final float[] batchXNorm;
  private final float[] batchQ;
  private final float[] batchK;
  private final float[] batchV;
  private final float[] batchAttentionScores;
  private final float[] batchAttnOut;
  private final float[] batchAttnProjected;
  private final float[] batchFfnGate;
  private final float[] batchFfnUp;
  private final float[] batchFfnOut;
  private final float[] batchFfnProjected;
  private final int[] batchPositions;
  private final byte[] batchQuantizedActivation;
  private final float[] batchQuantizedActivationScales;
  private final int[] batchQuantizedActivationZeroPointCorrections;
  private final short[] batchQuantizedActivationSums;
  private final float[] batchQ4LaneScratch;
  private float[] verificationLogits = new float[0];
  private float[] sessionBatchLogits = new float[0];
  private int nextPosition;

  public LlamaForwardPass(LlamaConfig config, LlamaWeights weights, KvCache cache) {
    this(
        config,
        weights,
        cache,
        null,
        defaultPlan(config, weights),
        GgufBatchedMatrixKernel.none(),
        BatchedCausalAttentionKernel.none());
  }

  public LlamaForwardPass(
      LlamaConfig config,
      LlamaWeights weights,
      KvCache cache,
      PureJavaExecutionPlan executionPlan) {
    this(
        config,
        weights,
        cache,
        null,
        executionPlan,
        GgufBatchedMatrixKernel.none(),
        BatchedCausalAttentionKernel.none());
  }

  public LlamaForwardPass(
      LlamaConfig config,
      LlamaWeights weights,
      KvCache cache,
      PureJavaExecutionPlan executionPlan,
      GgufBatchedMatrixKernel batchedMatrixKernel) {
    this(
        config,
        weights,
        cache,
        null,
        executionPlan,
        batchedMatrixKernel,
        BatchedCausalAttentionKernel.none());
  }

  public LlamaForwardPass(
      LlamaConfig config,
      LlamaWeights weights,
      KvCache cache,
      PureJavaExecutionPlan executionPlan,
      GgufBatchedMatrixKernel batchedMatrixKernel,
      BatchedCausalAttentionKernel batchedAttentionKernel) {
    this(config, weights, cache, null, executionPlan, batchedMatrixKernel, batchedAttentionKernel);
  }

  LlamaForwardPass(
      LlamaConfig config, LlamaWeights weights, KvCache cache, LayerObserver layerObserver) {
    this(
        config,
        weights,
        cache,
        layerObserver,
        defaultPlan(config, weights),
        GgufBatchedMatrixKernel.none(),
        BatchedCausalAttentionKernel.none());
  }

  LlamaForwardPass(
      LlamaConfig config,
      LlamaWeights weights,
      KvCache cache,
      LayerObserver layerObserver,
      PureJavaExecutionPlan executionPlan) {
    this(
        config,
        weights,
        cache,
        layerObserver,
        executionPlan,
        GgufBatchedMatrixKernel.none(),
        BatchedCausalAttentionKernel.none());
  }

  LlamaForwardPass(
      LlamaConfig config,
      LlamaWeights weights,
      KvCache cache,
      LayerObserver layerObserver,
      PureJavaExecutionPlan executionPlan,
      GgufBatchedMatrixKernel batchedMatrixKernel) {
    this(
        config,
        weights,
        cache,
        layerObserver,
        executionPlan,
        batchedMatrixKernel,
        BatchedCausalAttentionKernel.none());
  }

  LlamaForwardPass(
      LlamaConfig config,
      LlamaWeights weights,
      KvCache cache,
      LayerObserver layerObserver,
      PureJavaExecutionPlan executionPlan,
      GgufBatchedMatrixKernel batchedMatrixKernel,
      BatchedCausalAttentionKernel batchedAttentionKernel) {
    this.config = config;
    this.weights = weights;
    this.cache = cache;
    this.layerObserver = layerObserver;
    if (config.usesBidirectionalAttention()) {
      // This pass is causal throughout: it walks one token at a time and reads keys from a cache of
      // positions already visited. Given encoder weights it would run to completion and return
      // activations that are wrong in a way nothing downstream can detect.
      throw new IllegalArgumentException(
          "architecture "
              + config.architecture()
              + " attends bidirectionally and cannot be run through the causal pass; use"
              + " EncoderForwardPass");
    }
    Objects.requireNonNull(executionPlan, "executionPlan");
    ModelTopology actualTopology =
        ModelTopology.from(executionPlan.topology().architecture(), config, weights);
    if (!executionPlan.topology().equals(actualTopology)) {
      throw new IllegalArgumentException("execution plan topology does not match loaded weights");
    }
    GgufBatchedMatrixKernel loadedMatrixKernel =
        Objects.requireNonNull(batchedMatrixKernel, "batchedMatrixKernel");
    if (executionPlan.injectedGroupedProjections()
        && !actualTopology.hasInjectedGroupedProjection(loadedMatrixKernel)) {
      throw new IllegalArgumentException(
          "execution plan grouped projections do not match the injected matrix kernel");
    }
    this.groupedProjections = executionPlan.groupedProjections();
    this.mixedKProjections = executionPlan.mixedKProjections();
    this.q4Kernel = executionPlan.q4Kernel();
    this.q6BatchedKernel = executionPlan.q6BatchedKernel();
    this.finalLayerPrefillPruning = executionPlan.finalLayerPrefillPruning();
    this.finalLayerKvOnlyPrefill = executionPlan.finalLayerKvOnlyPrefill();
    this.batchedAttentionScores = executionPlan.batchedAttentionScores();
    this.batchedAttentionValues = executionPlan.batchedAttentionValues();
    this.stagedQuantizedFfn = executionPlan.stagedQuantizedFfn();
    this.stagedQuantizedLayer = executionPlan.stagedQuantizedLayer();
    this.blockMajorQ8Activations = executionPlan.blockMajorQ8Activations();
    this.q8BlockMajorKernel = executionPlan.q8BlockMajorKernel();
    this.parallelQ8FfnPreparation = executionPlan.parallelQ8FfnPreparation();
    this.batchedMatrixKernel = loadedMatrixKernel;
    this.batchedAttentionKernel =
        Objects.requireNonNull(batchedAttentionKernel, "batchedAttentionKernel");
    if (!config.usesStandardLlamaLayerSemantics()
        && (finalLayerPrefillPruning
            || finalLayerKvOnlyPrefill
            || stagedQuantizedFfn
            || stagedQuantizedLayer)) {
      throw new IllegalArgumentException(
          "execution plan selected Llama-only layer shortcuts for " + config.architecture());
    }
    this.globalRopeTable =
        new RotaryTable(config.keyLength(), config.ropeTheta(), config.ropeFrequencyScale());
    this.slidingWindowRopeTable =
        config.slidingWindow() > 0
            ? new RotaryTable(config.keyLength(), config.slidingWindowRopeTheta(), 1.0f)
            : globalRopeTable;

    int dim = config.embeddingDim();
    int hiddenDim = config.hiddenDim();
    int vocabSize = config.vocabSize();

    this.x = new float[dim];
    this.xNorm = new float[dim];
    this.q = new float[config.queryDim()];
    this.k = new float[config.keyDim()];
    this.v = new float[config.valueDim()];
    this.attnOut = new float[config.attentionOutputDim()];
    this.attentionScores = new float[cache.maxSeqLen()];
    this.attnProjected = new float[dim];
    this.ffnGate = new float[hiddenDim];
    this.ffnUp = new float[hiddenDim];
    this.ffnOut = new float[hiddenDim];
    this.ffnProjected = new float[dim];
    this.logits = new float[vocabSize];
    int maxProjectionInput = Math.max(Math.max(dim, hiddenDim), config.attentionOutputDim());
    this.quantizedActivation = new byte[maxProjectionInput];
    this.quantizedActivationScales = new float[(maxProjectionInput + 31) / 32];
    this.quantizedActivationZeroPointCorrections = new int[(maxProjectionInput + 3) / 4];
    this.quantizedActivationSums = new short[(maxProjectionInput + 15) / 16];

    int requestedBatchSize = executionPlan.prefillBatchSize();
    int capacity = Math.min(requestedBatchSize, cache.maxSeqLen());
    this.batchedPrefill = capacity > 1;
    this.groupedBatchedPrefill =
        batchedPrefill
            && groupedProjections
            && usesGroupedBatchedProjection(config, weights, mixedKProjections);
    this.batchedSessionOutputProjection =
        batchedPrefill && TensorOps.supportsBatchedMatmul(weights.outputType());
    this.prefillBatchCapacity = batchedPrefill ? capacity : 0;
    this.batchX = batchBuffer(prefillBatchCapacity, dim);
    this.batchXNorm = batchBuffer(prefillBatchCapacity, dim);
    this.batchQ = batchBuffer(prefillBatchCapacity, config.queryDim());
    this.batchK = batchBuffer(prefillBatchCapacity, config.keyDim());
    this.batchV = batchBuffer(prefillBatchCapacity, config.valueDim());
    this.batchAttentionScores =
        stagedQuantizedLayer ? batchBuffer(prefillBatchCapacity, cache.maxSeqLen()) : new float[0];
    this.batchAttnOut = batchBuffer(prefillBatchCapacity, config.attentionOutputDim());
    this.batchAttnProjected = batchBuffer(prefillBatchCapacity, dim);
    this.batchFfnGate = batchBuffer(prefillBatchCapacity, hiddenDim);
    this.batchFfnUp = batchBuffer(prefillBatchCapacity, hiddenDim);
    this.batchFfnOut = batchBuffer(prefillBatchCapacity, hiddenDim);
    this.batchFfnProjected = batchBuffer(prefillBatchCapacity, dim);
    this.batchPositions = new int[prefillBatchCapacity];
    this.batchQuantizedActivation =
        new byte[Math.multiplyExact(prefillBatchCapacity, maxProjectionInput)];
    this.batchQuantizedActivationScales =
        new float[Math.multiplyExact(prefillBatchCapacity, (maxProjectionInput + 31) / 32)];
    this.batchQuantizedActivationZeroPointCorrections =
        new int[Math.multiplyExact(prefillBatchCapacity, (maxProjectionInput + 3) / 4)];
    this.batchQuantizedActivationSums =
        new short[Math.multiplyExact(prefillBatchCapacity, (maxProjectionInput + 15) / 16)];
    int maxProjectionOutput =
        Math.max(
            Math.max(dim, hiddenDim),
            Math.max(config.queryDim(), Math.max(config.keyDim(), config.valueDim())));
    int q4LaneRows =
        groupedBatchedPrefill ? maxGroupedQ4ProjectionRows(config, weights) : maxProjectionOutput;
    if (weights.outputType() == GgufTensorType.Q4_0) {
      q4LaneRows = Math.max(q4LaneRows, vocabSize);
    }
    this.batchQ4LaneScratch =
        (weights.outputType() == GgufTensorType.Q4_0
                || usesProjectionType(config, weights, GgufTensorType.Q4_0))
            ? new float[Math.multiplyExact(Math.multiplyExact(prefillBatchCapacity, q4LaneRows), 8)]
            : new float[0];
    this.stagedQuantizedPlan =
        (stagedQuantizedFfn || stagedQuantizedLayer) && batchedPrefill
            ? new QuantizedBatchedLayerPlan(
                prefillBatchCapacity,
                dim,
                config.attentionOutputDim(),
                hiddenDim,
                config.rmsNormEps(),
                usesVectorizedSwiGlu(),
                q4Kernel,
                blockMajorQ8Activations,
                q8BlockMajorKernel,
                parallelQ8FfnPreparation,
                executionPlan.runtime().ggufThreads(),
                batchX,
                batchXNorm,
                batchAttnOut,
                batchAttnProjected,
                batchFfnGate,
                batchFfnUp,
                batchFfnOut,
                batchFfnProjected,
                batchQ4LaneScratch,
                this::prepareBatchedAttention,
                this::computeBatchedAttention)
            : null;
  }

  /** Runs a single forward pass for the given token at the given position. Returns logits. */
  public float[] forward(int token, int position) {
    return forwardTransient(token, position).clone();
  }

  /**
   * Runs a forward pass and returns backend-owned logits valid until the next logits-producing
   * call.
   */
  public float[] forwardTransient(int token, int position) {
    return forwardInternal(token, position, Head.LOGITS);
  }

  /** Processes prompt tokens while computing the vocabulary projection only for the final token. */
  public float[] prefill(int[] tokens, int startPosition) {
    Objects.requireNonNull(tokens, "tokens");
    if (tokens.length == 0) {
      throw new IllegalArgumentException("tokens must not be empty");
    }
    if (startPosition < 0) {
      throw new IllegalArgumentException("startPosition must be >= 0");
    }

    if (batchedPrefill && tokens.length > 1) {
      return prefillBatched(tokens, startPosition);
    }

    int finalIndex = tokens.length - 1;
    for (int index = 0; index < finalIndex; index++) {
      forwardInternal(tokens[index], Math.addExact(startPosition, index), Head.NONE);
    }
    return forwardInternal(
        tokens[finalIndex], Math.addExact(startPosition, finalIndex), Head.LOGITS);
  }

  /**
   * Runs one step and returns the final normalized hidden state rather than logits.
   *
   * <p>This is the activation the vocabulary projection would consume. Omitting that projection is
   * the entire saving — it is the widest matmul in the pass — which makes embedding strictly
   * cheaper per token than generation on the same model.
   *
   * <p>The returned array is backend-owned scratch, valid until the next call. Copy it to keep it.
   */
  public float[] hiddenState(int token, int position) {
    return forwardInternal(token, position, Head.HIDDEN);
  }

  /**
   * Prefills a sequence and returns the hidden state of its final position.
   *
   * <p>Suits last-token pooling, which is what Qwen3-Embedding and most decoder-only embedding
   * models use. Mean pooling is a caller-side loop over {@link #hiddenState(int, int)}: pooling is
   * a property of the embedding model, not of the forward pass.
   *
   * <p>Deliberately walks tokens one at a time rather than using batched prefill, whose fast path
   * is built around producing logits for the final token only.
   */
  public float[] prefillHiddenState(int[] tokens, int startPosition) {
    Objects.requireNonNull(tokens, "tokens");
    if (tokens.length == 0) {
      throw new IllegalArgumentException("tokens must not be empty");
    }
    if (startPosition < 0) {
      throw new IllegalArgumentException("startPosition must be >= 0");
    }
    int finalIndex = tokens.length - 1;
    for (int index = 0; index < finalIndex; index++) {
      forwardInternal(tokens[index], Math.addExact(startPosition, index), Head.NONE);
    }
    return forwardInternal(
        tokens[finalIndex], Math.addExact(startPosition, finalIndex), Head.HIDDEN);
  }

  /** Opens independent sequence state backed by the same loaded weights and execution plan. */
  public Session openSession() {
    return new Session(
        this,
        new KvCache(config.numLayers(), cache.maxSeqLen(), config.keyDim(), config.valueDim()));
  }

  /** Returns the largest independent-session batch supported by this execution plan. */
  public int maxSessionBatchSize() {
    return batchedPrefill ? prefillBatchCapacity : 1;
  }

  /** Runs one independent session step and returns stable logits. */
  public float[] forward(Session session, int token, int position) {
    return forwardTransient(session, token, position).clone();
  }

  /** Runs one independent session step using reusable logits storage. */
  public float[] forwardTransient(Session session, int token, int position) {
    requireSession(session);
    return forwardSessionInternal(session, token, position, Head.LOGITS);
  }

  /** Prefills one independent session without changing the default sequence. */
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
    if (tokens.length > session.cache.maxSeqLen() - startPosition) {
      throw new IllegalArgumentException(
          "prompt exceeds context length: " + (startPosition + (long) tokens.length));
    }

    int finalIndex = tokens.length - 1;
    for (int index = 0; index < finalIndex; index++) {
      forwardSessionInternal(
          session, tokens[index], Math.addExact(startPosition, index), Head.NONE);
    }
    return forwardSessionInternal(
        session, tokens[finalIndex], Math.addExact(startPosition, finalIndex), Head.LOGITS);
  }

  /** Runs one decode token for each independent session and returns stable logits. */
  public LogitBatch forwardBatch(Session[] sessions, int[] tokens) {
    return forwardBatchTransient(sessions, tokens).snapshot();
  }

  /**
   * Runs one decode token for each independent session. Returned storage may be reused by the next
   * logits-producing call.
   */
  public LogitBatch forwardBatchTransient(Session[] sessions, int[] tokens) {
    validateSessionBatch(sessions, tokens);
    if (sessions.length == 1) {
      ensureSessionBatchLogits(config.vocabSize());
      float[] row =
          forwardSessionInternal(sessions[0], tokens[0], sessions[0].nextPosition, Head.LOGITS);
      System.arraycopy(row, 0, sessionBatchLogits, 0, config.vocabSize());
      return new LogitBatch(1, config.vocabSize(), sessionBatchLogits);
    }
    return forwardIndependentSessionBatch(sessions, tokens);
  }

  /** Discards session state at and after {@code checkpoint}. */
  public void rewind(Session session, int checkpoint) {
    requireSession(session);
    if (checkpoint < 0 || checkpoint > session.nextPosition) {
      throw new IllegalArgumentException(
          "checkpoint must be between 0 and " + session.nextPosition + ": " + checkpoint);
    }
    session.cache.discardFrom(checkpoint);
    session.nextPosition = checkpoint;
  }

  /** Clears one independent session without changing any other sequence. */
  public void reset(Session session) {
    requireSession(session);
    session.cache.clear();
    session.nextPosition = 0;
  }

  private float[] prefillBatched(int[] tokens, int startPosition) {
    if (startPosition != nextPosition) {
      throw new IllegalArgumentException(
          "position must be sequential: expected " + nextPosition + ", got " + startPosition);
    }
    if (tokens.length > cache.maxSeqLen() - startPosition) {
      throw new IllegalArgumentException(
          "prompt exceeds context length: " + (startPosition + (long) tokens.length));
    }

    int tokenOffset = 0;
    while (tokenOffset < tokens.length) {
      int batchSize = Math.min(prefillBatchCapacity, tokens.length - tokenOffset);
      boolean finalBatch = tokenOffset + batchSize == tokens.length;
      if (batchSize == 1) {
        return forwardInternal(tokens[tokenOffset], startPosition + tokenOffset, Head.LOGITS);
      }
      prefillBatch(
          tokens, tokenOffset, batchSize, startPosition + tokenOffset, finalBatch, null, 0);
      nextPosition += batchSize;
      tokenOffset += batchSize;
    }
    return logits;
  }

  private void prefillBatch(
      int[] tokens,
      int tokenOffset,
      int batchSize,
      int startPosition,
      boolean computeFinalLogits,
      float[] batchLogits,
      int batchLogitsOffset) {
    int dim = config.embeddingDim();
    int queryDim = config.queryDim();
    int keyDim = config.keyDim();
    int valueDim = config.valueDim();
    int attentionOutputDim = config.attentionOutputDim();
    int hiddenDim = config.hiddenDim();
    prepareRopeBatch(startPosition, batchSize);

    for (int batch = 0; batch < batchSize; batch++) {
      weights.embedToken(tokens[tokenOffset + batch], x);
      scaleActive(x, 0, dim, config.embeddingScale());
      System.arraycopy(x, 0, batchX, batch * dim, dim);
    }

    for (int layer = 0; layer < config.numLayers(); layer++) {
      LlamaWeights.LayerWeights lw = weights.layer(layer);
      for (int batch = 0; batch < batchSize; batch++) {
        int xOffset = batch * dim;
        TensorOps.rmsNorm(
            batchXNorm, xOffset, batchX, xOffset, lw.attentionNorm(), dim, config.rmsNormEps());
      }

      boolean useFinalLayerKvOnlyPrefill =
          finalLayerKvOnlyPrefill
              && batchLogits == null
              && layerObserver == null
              && layer == config.numLayers() - 1;
      if (useFinalLayerKvOnlyPrefill) {
        finishFinalLayerKvOnlyBatch(lw, layer, batchSize, startPosition, computeFinalLogits);
        break;
      }

      if (groupedBatchedPrefill) {
        tripleBatchedMatmulDispatch(
            batchQ,
            lw.wq(),
            lw.wqType(),
            queryDim,
            batchK,
            lw.wk(),
            lw.wkType(),
            keyDim,
            batchV,
            lw.wv(),
            lw.wvType(),
            valueDim,
            batchXNorm,
            batchSize,
            dim);
      } else {
        batchedMatmulDispatch(batchQ, batchXNorm, batchSize, lw.wq(), lw.wqType(), queryDim, dim);
        batchedMatmulDispatch(batchK, batchXNorm, batchSize, lw.wk(), lw.wkType(), keyDim, dim);
        batchedMatmulDispatch(batchV, batchXNorm, batchSize, lw.wv(), lw.wvType(), valueDim, dim);
      }

      boolean pruneFinalLayer =
          finalLayerPrefillPruning
              && batchLogits == null
              && layerObserver == null
              && layer == config.numLayers() - 1;
      boolean useStagedQuantizedLayer =
          !pruneFinalLayer
              && stagedQuantizedLayer
              && stagedQuantizedPlan != null
              && stagedQuantizedPlan.supportsLayer(lw);
      if (useStagedQuantizedLayer) {
        cache.reserveSequenceCapacity(startPosition + batchSize);
        stagedQuantizedPlan.executeLayer(lw, layer, startPosition, batchSize);
      } else {
        prepareBatchedAttention(lw, layer, startPosition, 0, batchSize);
        computeBatchedAttention(lw, layer, startPosition, 0, batchSize);
        batchedMatmulDispatch(
            batchAttnProjected,
            batchAttnOut,
            batchSize,
            lw.wo(),
            lw.woType(),
            dim,
            attentionOutputDim);
        normalizeProjectionBatch(batchAttnProjected, batchSize, dim, lw.attentionPostNorm());
        addActiveInPlace(batchX, batchAttnProjected, batchSize * dim);

        if (pruneFinalLayer) {
          if (computeFinalLogits) {
            finishFinalLayerFfnRow(lw, (batchSize - 1) * dim, dim, hiddenDim);
          }
          break;
        }

        for (int batch = 0; batch < batchSize; batch++) {
          int xOffset = batch * dim;
          TensorOps.rmsNorm(
              batchXNorm, xOffset, batchX, xOffset, lw.ffnNorm(), dim, config.rmsNormEps());
        }
        if (stagedQuantizedFfn
            && stagedQuantizedPlan != null
            && stagedQuantizedPlan.supportsFfn(lw)) {
          stagedQuantizedPlan.executeFfn(lw, batchSize);
        } else {
          if (groupedBatchedPrefill) {
            dualBatchedMatmulDispatch(
                batchFfnGate,
                lw.ffnGate(),
                lw.ffnGateType(),
                hiddenDim,
                batchFfnUp,
                lw.ffnUp(),
                lw.ffnUpType(),
                hiddenDim,
                batchXNorm,
                batchSize,
                dim);
          } else {
            batchedMatmulDispatch(
                batchFfnGate,
                batchXNorm,
                batchSize,
                lw.ffnGate(),
                lw.ffnGateType(),
                hiddenDim,
                dim);
            batchedMatmulDispatch(
                batchFfnUp, batchXNorm, batchSize, lw.ffnUp(), lw.ffnUpType(), hiddenDim, dim);
          }
          for (int batch = 0; batch < batchSize; batch++) {
            int hiddenOffset = batch * hiddenDim;
            activateFfn(
                batchFfnOut,
                hiddenOffset,
                batchFfnGate,
                hiddenOffset,
                batchFfnUp,
                hiddenOffset,
                hiddenDim);
          }
          batchedMatmulDispatch(
              batchFfnProjected,
              batchFfnOut,
              batchSize,
              lw.ffnDown(),
              lw.ffnDownType(),
              dim,
              hiddenDim);
        }
      }
      normalizeProjectionBatch(batchFfnProjected, batchSize, dim, lw.ffnPostNorm());
      addActiveInPlace(batchX, batchFfnProjected, batchSize * dim);
      if (layerObserver != null) {
        for (int batch = 0; batch < batchSize; batch++) {
          layerObserver.onLayerComplete(layer, startPosition + batch, batchX, batch * dim, dim);
        }
      }
    }

    if (batchLogits != null) {
      int vocabSize = config.vocabSize();
      for (int batch = 0; batch < batchSize; batch++) {
        int stateOffset = batch * dim;
        TensorOps.rmsNorm(
            xNorm, 0, batchX, stateOffset, weights.outputNormWeight(), dim, config.rmsNormEps());
        matmulDispatch(
            logits, xNorm, weights.outputSegment(), weights.outputType(), vocabSize, dim);
        softcapLogits(logits);
        System.arraycopy(logits, 0, batchLogits, batchLogitsOffset + batch * vocabSize, vocabSize);
      }
    } else if (computeFinalLogits) {
      int finalOffset = (batchSize - 1) * dim;
      TensorOps.rmsNorm(
          xNorm, 0, batchX, finalOffset, weights.outputNormWeight(), dim, config.rmsNormEps());
      matmulDispatch(
          logits, xNorm, weights.outputSegment(), weights.outputType(), config.vocabSize(), dim);
      softcapLogits(logits);
    }
  }

  private LogitBatch forwardIndependentSessionBatch(Session[] sessions, int[] tokens) {
    int batchSize = sessions.length;
    int dim = config.embeddingDim();
    int vocabSize = config.vocabSize();

    prepareIndependentSessionInputs(sessions, tokens, batchSize, dim);
    for (int layer = 0; layer < config.numLayers(); layer++) {
      executeIndependentSessionLayer(sessions, batchSize, layer);
    }
    projectIndependentSessionLogits(batchSize, dim, vocabSize);
    for (Session session : sessions) {
      session.nextPosition++;
    }
    return new LogitBatch(batchSize, vocabSize, sessionBatchLogits);
  }

  private void prepareIndependentSessionInputs(
      Session[] sessions, int[] tokens, int batchSize, int dim) {
    for (int batch = 0; batch < batchSize; batch++) {
      batchPositions[batch] = sessions[batch].nextPosition;
    }
    prepareRopePositions(batchPositions, batchSize);

    for (int batch = 0; batch < batchSize; batch++) {
      weights.embedToken(tokens[batch], x);
      scaleActive(x, 0, dim, config.embeddingScale());
      System.arraycopy(x, 0, batchX, batch * dim, dim);
    }
  }

  private void executeIndependentSessionLayer(Session[] sessions, int batchSize, int layer) {
    int dim = config.embeddingDim();
    int attentionOutputDim = config.attentionOutputDim();
    LlamaWeights.LayerWeights lw = weights.layer(layer);

    normalizeIndependentSessionBatch(batchXNorm, batchX, batchSize, dim, lw.attentionNorm());
    projectIndependentSessionQkv(lw, batchSize, dim);
    attendIndependentSessions(sessions, batchSize, layer, lw);

    batchedMatmulDispatch(
        batchAttnProjected, batchAttnOut, batchSize, lw.wo(), lw.woType(), dim, attentionOutputDim);
    normalizeProjectionBatch(batchAttnProjected, batchSize, dim, lw.attentionPostNorm());
    addActiveInPlace(batchX, batchAttnProjected, batchSize * dim);

    executeIndependentSessionFfn(lw, batchSize, dim);
    observeIndependentSessionLayer(sessions, batchSize, layer, dim);
  }

  private void normalizeIndependentSessionBatch(
      float[] target, float[] source, int batchSize, int dim, float[] norm) {
    for (int batch = 0; batch < batchSize; batch++) {
      int stateOffset = batch * dim;
      TensorOps.rmsNorm(target, stateOffset, source, stateOffset, norm, dim, config.rmsNormEps());
    }
  }

  private void projectIndependentSessionQkv(LlamaWeights.LayerWeights lw, int batchSize, int dim) {
    int queryDim = config.queryDim();
    int keyDim = config.keyDim();
    int valueDim = config.valueDim();
    if (groupedBatchedPrefill) {
      tripleBatchedMatmulDispatch(
          batchQ,
          lw.wq(),
          lw.wqType(),
          queryDim,
          batchK,
          lw.wk(),
          lw.wkType(),
          keyDim,
          batchV,
          lw.wv(),
          lw.wvType(),
          valueDim,
          batchXNorm,
          batchSize,
          dim);
      return;
    }
    batchedMatmulDispatch(batchQ, batchXNorm, batchSize, lw.wq(), lw.wqType(), queryDim, dim);
    batchedMatmulDispatch(batchK, batchXNorm, batchSize, lw.wk(), lw.wkType(), keyDim, dim);
    batchedMatmulDispatch(batchV, batchXNorm, batchSize, lw.wv(), lw.wvType(), valueDim, dim);
  }

  private void attendIndependentSessions(
      Session[] sessions, int batchSize, int layer, LlamaWeights.LayerWeights lw) {
    int queryDim = config.queryDim();
    int keyDim = config.keyDim();
    int valueDim = config.valueDim();
    int attentionOutputDim = config.attentionOutputDim();
    boolean usesRope = config.usesRope(layer);
    for (int batch = 0; batch < batchSize; batch++) {
      int queryOffset = batch * queryDim;
      int keyOffset = batch * keyDim;
      int valueOffset = batch * valueDim;
      addOptionalBias(batchQ, queryOffset, lw.qBias());
      addOptionalBias(batchK, keyOffset, lw.kBias());
      addOptionalBias(batchV, valueOffset, lw.vBias());
      for (int head = 0; head < config.numHeads(); head++) {
        int offset = queryOffset + head * config.keyLength();
        normalizeHead(batchQ, offset, lw.qNorm(), config.keyLength());
        if (usesRope) {
          applyRopeBatch(batchQ, offset, batch, layer);
        }
      }
      for (int head = 0; head < config.numKvHeads(); head++) {
        int offset = keyOffset + head * config.keyLength();
        normalizeHead(batchK, offset, lw.kNorm(), config.keyLength());
        if (usesRope) {
          applyRopeBatch(batchK, offset, batch, layer);
        }
      }

      Session session = sessions[batch];
      session.cache.store(layer, session.nextPosition, batchK, keyOffset, batchV, valueOffset);
      float[] scores = batchAttentionScores.length == 0 ? attentionScores : batchAttentionScores;
      int scoresOffset = batchAttentionScores.length == 0 ? 0 : batch * session.cache.maxSeqLen();
      groupedQueryAttention(
          batchQ,
          queryOffset,
          batchAttnOut,
          batch * attentionOutputDim,
          layer,
          session.nextPosition,
          session.cache,
          session.cache.keyBuffer(),
          session.cache.valueBuffer(),
          scores,
          scoresOffset);
    }
  }

  private void executeIndependentSessionFfn(LlamaWeights.LayerWeights lw, int batchSize, int dim) {
    int hiddenDim = config.hiddenDim();
    normalizeIndependentSessionBatch(batchXNorm, batchX, batchSize, dim, lw.ffnNorm());
    if (stagedQuantizedFfn && stagedQuantizedPlan != null && stagedQuantizedPlan.supportsFfn(lw)) {
      stagedQuantizedPlan.executeFfn(lw, batchSize);
    } else {
      projectIndependentSessionGateUp(lw, batchSize, dim, hiddenDim);
      batchedMatmulDispatch(
          batchFfnProjected,
          batchFfnOut,
          batchSize,
          lw.ffnDown(),
          lw.ffnDownType(),
          dim,
          hiddenDim);
    }
    normalizeProjectionBatch(batchFfnProjected, batchSize, dim, lw.ffnPostNorm());
    addActiveInPlace(batchX, batchFfnProjected, batchSize * dim);
  }

  private void projectIndependentSessionGateUp(
      LlamaWeights.LayerWeights lw, int batchSize, int dim, int hiddenDim) {
    if (groupedBatchedPrefill) {
      dualBatchedMatmulDispatch(
          batchFfnGate,
          lw.ffnGate(),
          lw.ffnGateType(),
          hiddenDim,
          batchFfnUp,
          lw.ffnUp(),
          lw.ffnUpType(),
          hiddenDim,
          batchXNorm,
          batchSize,
          dim);
    } else {
      batchedMatmulDispatch(
          batchFfnGate, batchXNorm, batchSize, lw.ffnGate(), lw.ffnGateType(), hiddenDim, dim);
      batchedMatmulDispatch(
          batchFfnUp, batchXNorm, batchSize, lw.ffnUp(), lw.ffnUpType(), hiddenDim, dim);
    }
    for (int batch = 0; batch < batchSize; batch++) {
      int hiddenOffset = batch * hiddenDim;
      activateFfn(
          batchFfnOut,
          hiddenOffset,
          batchFfnGate,
          hiddenOffset,
          batchFfnUp,
          hiddenOffset,
          hiddenDim);
    }
  }

  private void observeIndependentSessionLayer(
      Session[] sessions, int batchSize, int layer, int dim) {
    if (layerObserver == null) {
      return;
    }
    for (int batch = 0; batch < batchSize; batch++) {
      layerObserver.onLayerComplete(layer, sessions[batch].nextPosition, batchX, batch * dim, dim);
    }
  }

  private void projectIndependentSessionLogits(int batchSize, int dim, int vocabSize) {
    ensureSessionBatchLogits(Math.multiplyExact(batchSize, vocabSize));
    if (usesBatchedSessionOutputProjection(batchSize)) {
      normalizeIndependentSessionBatch(
          batchXNorm, batchX, batchSize, dim, weights.outputNormWeight());
      batchedMatmulDispatch(
          sessionBatchLogits,
          batchXNorm,
          batchSize,
          weights.outputSegment(),
          weights.outputType(),
          vocabSize,
          dim);
      for (int batch = 0; batch < batchSize; batch++) {
        softcapLogits(sessionBatchLogits, batch * vocabSize, vocabSize);
      }
      return;
    }
    for (int batch = 0; batch < batchSize; batch++) {
      int stateOffset = batch * dim;
      TensorOps.rmsNorm(
          xNorm, 0, batchX, stateOffset, weights.outputNormWeight(), dim, config.rmsNormEps());
      matmulDispatch(logits, xNorm, weights.outputSegment(), weights.outputType(), vocabSize, dim);
      softcapLogits(logits);
      System.arraycopy(logits, 0, sessionBatchLogits, batch * vocabSize, vocabSize);
    }
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
    int maxBatchSize = maxSessionBatchSize();
    if (sessions.length > maxBatchSize) {
      throw new IllegalArgumentException(
          "session batch exceeds capacity " + maxBatchSize + ": " + sessions.length);
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

  private void ensureSessionBatchLogits(int requiredLength) {
    if (sessionBatchLogits.length < requiredLength) {
      sessionBatchLogits = new float[requiredLength];
    }
  }

  /**
   * What a forward step should produce.
   *
   * <p>This was a {@code boolean computeLogits}, which conflated two independent decisions: whether
   * to project to the vocabulary, and whether the final layer's activation is needed at all. Prompt
   * prefill discards that activation, which is why it can skip most of the last layer; embedding
   * needs precisely that activation while skipping the projection. Three states, not two.
   */
  private enum Head {
    /** Project to the vocabulary and return logits. */
    LOGITS,
    /** Discard the activation; only key/value state must reach the cache. */
    NONE,
    /** Return the final normalized hidden state without the vocabulary projection. */
    HIDDEN
  }

  private float[] forwardInternal(int token, int position, Head head) {
    float[] result = forwardSequenceInternal(token, position, head, cache, nextPosition);
    nextPosition++;
    return result;
  }

  private float[] forwardSessionInternal(Session session, int token, int position, Head head) {
    float[] result =
        forwardSequenceInternal(token, position, head, session.cache, session.nextPosition);
    session.nextPosition++;
    return result;
  }

  private float[] forwardSequenceInternal(
      int token, int position, Head head, KvCache sequenceCache, int expectedPosition) {
    if (position != expectedPosition) {
      throw new IllegalArgumentException(
          "position must be sequential: expected " + expectedPosition + ", got " + position);
    }
    if (position >= sequenceCache.maxSeqLen()) {
      throw new IllegalArgumentException("position out of range: " + position);
    }

    int dim = config.embeddingDim();
    int keyLength = config.keyLength();
    int valueLength = config.valueLength();
    int numHeads = config.numHeads();
    int numKvHeads = config.numKvHeads();
    prepareRope(position);

    // Token embedding (dequantizes only the single row for this token)
    weights.embedToken(token, x);
    scaleActive(x, 0, dim, config.embeddingScale());

    // Process each transformer layer
    for (int layer = 0; layer < config.numLayers(); layer++) {
      LlamaWeights.LayerWeights lw = weights.layer(layer);

      // Attention norm
      TensorOps.rmsNorm(xNorm, x, lw.attentionNorm(), dim, config.rmsNormEps());

      // Only safe when the activation is genuinely discarded; HIDDEN needs the full final layer.
      if (finalLayerKvOnlyPrefill
          && head == Head.NONE
          && layerObserver == null
          && layer == config.numLayers() - 1) {
        finishFinalLayerKvOnlyToken(lw, layer, position, keyLength, numKvHeads, sequenceCache);
        return null;
      }

      // QKV projections
      if (groupedProjections) {
        tripleMatmulDispatch(
            q,
            lw.wq(),
            lw.wqType(),
            config.queryDim(),
            k,
            lw.wk(),
            lw.wkType(),
            config.keyDim(),
            v,
            lw.wv(),
            lw.wvType(),
            config.valueDim(),
            xNorm,
            dim);
      } else {
        matmulDispatch(q, xNorm, lw.wq(), lw.wqType(), config.queryDim(), dim);
        matmulDispatch(k, xNorm, lw.wk(), lw.wkType(), config.keyDim(), dim);
        matmulDispatch(v, xNorm, lw.wv(), lw.wvType(), config.valueDim(), dim);
      }
      addOptionalBias(q, lw.qBias());
      addOptionalBias(k, lw.kBias());
      addOptionalBias(v, lw.vBias());

      // RoPE for each head
      for (int h = 0; h < numHeads; h++) {
        int offset = h * keyLength;
        normalizeHead(q, offset, lw.qNorm(), keyLength);
        if (config.usesRope(layer)) {
          applyRope(q, offset, layer);
        }
      }
      for (int h = 0; h < numKvHeads; h++) {
        int offset = h * keyLength;
        normalizeHead(k, offset, lw.kNorm(), keyLength);
        if (config.usesRope(layer)) {
          applyRope(k, offset, layer);
        }
      }

      // Store K,V in cache
      sequenceCache.store(layer, position, k, v);
      float[] keyCache = sequenceCache.keyBuffer();
      float[] valueCache = sequenceCache.valueBuffer();

      // Grouped-query attention
      java.util.Arrays.fill(attnOut, 0.0f);
      int groupSize = numHeads / numKvHeads;
      float scale = (float) (1.0 / Math.sqrt(keyLength));

      for (int h = 0; h < numHeads; h++) {
        int kvHead = h / groupSize;
        int qOff = h * keyLength;
        int outputOffset = h * valueLength;
        int firstPosition = config.attentionStartPosition(layer, position);

        computeAttentionScores(
            q,
            qOff,
            layer,
            position,
            kvHead,
            sequenceCache,
            keyCache,
            keyLength,
            scale,
            attentionScores,
            0);

        // Softmax over scores
        TensorOps.softmax(attentionScores, firstPosition, position - firstPosition + 1);

        accumulateAttentionValues(
            attnOut,
            outputOffset,
            layer,
            position,
            kvHead,
            sequenceCache,
            valueCache,
            valueLength,
            attentionScores,
            0);
      }

      // Output projection
      matmulDispatch(
          attnProjected, attnOut, lw.wo(), lw.woType(), dim, config.attentionOutputDim());
      normalizeProjection(attnProjected, lw.attentionPostNorm(), dim);

      // Residual connection
      for (int i = 0; i < dim; i++) {
        x[i] += attnProjected[i];
      }

      // Second pruning shortcut: the last layer's FFN cannot change a discarded activation.
      // HIDDEN consumes that activation, so it must run the FFN.
      if (finalLayerPrefillPruning
          && head == Head.NONE
          && layerObserver == null
          && layer == config.numLayers() - 1) {
        return null;
      }

      // FFN norm
      TensorOps.rmsNorm(xNorm, x, lw.ffnNorm(), dim, config.rmsNormEps());

      // FFN: SwiGLU
      if (groupedProjections) {
        dualMatmulDispatch(
            ffnGate,
            lw.ffnGate(),
            lw.ffnGateType(),
            config.hiddenDim(),
            ffnUp,
            lw.ffnUp(),
            lw.ffnUpType(),
            config.hiddenDim(),
            xNorm,
            dim);
      } else {
        matmulDispatch(ffnGate, xNorm, lw.ffnGate(), lw.ffnGateType(), config.hiddenDim(), dim);
        matmulDispatch(ffnUp, xNorm, lw.ffnUp(), lw.ffnUpType(), config.hiddenDim(), dim);
      }
      activateFfn(ffnOut, 0, ffnGate, 0, ffnUp, 0, config.hiddenDim());

      // Down projection
      matmulDispatch(ffnProjected, ffnOut, lw.ffnDown(), lw.ffnDownType(), dim, config.hiddenDim());
      normalizeProjection(ffnProjected, lw.ffnPostNorm(), dim);

      // Residual connection
      for (int i = 0; i < dim; i++) {
        x[i] += ffnProjected[i];
      }
      if (layerObserver != null) {
        layerObserver.onLayerComplete(layer, position, x, 0, dim);
      }
    }

    if (head == Head.NONE) {
      return null;
    }
    TensorOps.rmsNorm(xNorm, x, weights.outputNormWeight(), dim, config.rmsNormEps());
    if (head == Head.HIDDEN) {
      // The normalized activation the vocabulary projection would consume. Skipping that matmul
      // is the entire saving: it is the widest one in the pass.
      return xNorm;
    }
    matmulDispatch(
        logits, xNorm, weights.outputSegment(), weights.outputType(), config.vocabSize(), dim);
    softcapLogits(logits);
    return logits;
  }

  /** Verifies a contiguous speculative continuation and returns logits for every consumed token. */
  public LogitBatch verify(int[] tokens, int startPosition) {
    return verifyTransient(tokens, startPosition).snapshot();
  }

  /**
   * Verifies a contiguous speculative continuation using logits storage reused by the next call.
   */
  public LogitBatch verifyTransient(int[] tokens, int startPosition) {
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
          "tokens exceed context length: " + (startPosition + (long) tokens.length));
    }

    int vocabSize = config.vocabSize();
    int resultLength = Math.multiplyExact(tokens.length, vocabSize);
    if (verificationLogits.length < resultLength) {
      verificationLogits = new float[resultLength];
    }
    int tokenOffset = 0;
    while (tokenOffset < tokens.length) {
      int batchSize =
          batchedPrefill ? Math.min(prefillBatchCapacity, tokens.length - tokenOffset) : 1;
      if (batchSize == 1) {
        float[] row =
            forwardInternal(
                tokens[tokenOffset], Math.addExact(startPosition, tokenOffset), Head.LOGITS);
        System.arraycopy(row, 0, verificationLogits, tokenOffset * vocabSize, vocabSize);
      } else {
        prefillBatch(
            tokens,
            tokenOffset,
            batchSize,
            Math.addExact(startPosition, tokenOffset),
            false,
            verificationLogits,
            tokenOffset * vocabSize);
        nextPosition += batchSize;
      }
      tokenOffset += batchSize;
    }
    return new LogitBatch(tokens.length, vocabSize, verificationLogits);
  }

  /** Returns the next sequence position for speculative rollback. */
  public int checkpoint() {
    return nextPosition;
  }

  /** Discards speculative cache entries at and after {@code checkpoint}. */
  public void rewind(int checkpoint) {
    if (checkpoint < 0 || checkpoint > nextPosition) {
      throw new IllegalArgumentException(
          "checkpoint must be between 0 and " + nextPosition + ": " + checkpoint);
    }
    cache.discardFrom(checkpoint);
    batchedAttentionKernel.rewind(checkpoint);
    nextPosition = checkpoint;
  }

  /** Clears the autoregressive key-value cache before processing a new sequence. */
  public void reset() {
    cache.clear();
    batchedAttentionKernel.reset();
    nextPosition = 0;
  }

  boolean usesBatchedPrefill() {
    return batchedPrefill;
  }

  boolean usesGroupedBatchedPrefill() {
    return groupedBatchedPrefill;
  }

  boolean usesFinalLayerPrefillPruning() {
    return finalLayerPrefillPruning;
  }

  boolean usesFinalLayerKvOnlyPrefill() {
    return finalLayerKvOnlyPrefill;
  }

  GgufQ6BatchedKernel q6BatchedKernel() {
    return q6BatchedKernel;
  }

  boolean usesBatchedAttentionValues() {
    return batchedAttentionValues;
  }

  boolean usesBatchedAttentionScores() {
    return batchedAttentionScores;
  }

  boolean usesStagedQuantizedFfn() {
    return stagedQuantizedFfn && stagedQuantizedPlan != null;
  }

  boolean usesStagedQuantizedLayer() {
    return stagedQuantizedLayer && stagedQuantizedPlan != null;
  }

  boolean usesBlockMajorQ8Activations() {
    return blockMajorQ8Activations && stagedQuantizedPlan != null;
  }

  boolean usesParallelQ8FfnPreparation() {
    return parallelQ8FfnPreparation && stagedQuantizedPlan != null;
  }

  boolean usesBatchedSessionOutputProjection() {
    return batchedSessionOutputProjection;
  }

  private boolean usesBatchedSessionOutputProjection(int batchSize) {
    return batchedSessionOutputProjection
        || batchedMatrixKernel.isEligible(
            weights.outputType(), batchSize, config.vocabSize(), config.embeddingDim());
  }

  int stagedQuantizedLayerStageCount() {
    return usesStagedQuantizedLayer() ? stagedQuantizedPlan.layerStageCount() : 0;
  }

  private void finishFinalLayerKvOnlyBatch(
      LlamaWeights.LayerWeights layer,
      int layerIndex,
      int batchSize,
      int startPosition,
      boolean computeFinalLogits) {
    int dim = config.embeddingDim();
    int queryDim = config.queryDim();
    int keyDim = config.keyDim();
    int valueDim = config.valueDim();
    int keyLength = config.keyLength();
    int numHeads = config.numHeads();
    int numKvHeads = config.numKvHeads();

    if (groupedBatchedPrefill) {
      dualBatchedMatmulDispatch(
          batchK,
          layer.wk(),
          layer.wkType(),
          keyDim,
          batchV,
          layer.wv(),
          layer.wvType(),
          valueDim,
          batchXNorm,
          batchSize,
          dim);
    } else {
      batchedMatmulDispatch(batchK, batchXNorm, batchSize, layer.wk(), layer.wkType(), keyDim, dim);
      batchedMatmulDispatch(
          batchV, batchXNorm, batchSize, layer.wv(), layer.wvType(), valueDim, dim);
    }

    for (int batch = 0; batch < batchSize; batch++) {
      int keyOffset = batch * keyDim;
      int valueOffset = batch * valueDim;
      addOptionalBias(batchK, keyOffset, layer.kBias());
      addOptionalBias(batchV, valueOffset, layer.vBias());
      for (int head = 0; head < numKvHeads; head++) {
        int offset = keyOffset + head * keyLength;
        normalizeHead(batchK, offset, layer.kNorm(), keyLength);
        if (config.usesRope(layerIndex)) {
          applyRopeBatch(batchK, offset, batch, layerIndex);
        }
      }
      cache.store(layerIndex, startPosition + batch, batchK, keyOffset, batchV, valueOffset);
    }

    if (!computeFinalLogits) {
      return;
    }

    int finalBatch = batchSize - 1;
    int stateOffset = finalBatch * dim;
    System.arraycopy(batchXNorm, stateOffset, xNorm, 0, dim);
    matmulDispatch(q, xNorm, layer.wq(), layer.wqType(), queryDim, dim);
    addOptionalBias(q, layer.qBias());
    for (int head = 0; head < numHeads; head++) {
      int offset = head * keyLength;
      normalizeHead(q, offset, layer.qNorm(), keyLength);
      if (config.usesRope(layerIndex)) {
        applyRopeBatch(q, offset, finalBatch, layerIndex);
      }
    }

    groupedQueryAttention(
        q,
        0,
        attnOut,
        0,
        layerIndex,
        startPosition + finalBatch,
        cache,
        cache.keyBuffer(),
        cache.valueBuffer(),
        attentionScores,
        0);
    matmulDispatch(
        attnProjected, attnOut, layer.wo(), layer.woType(), dim, config.attentionOutputDim());
    normalizeProjection(attnProjected, layer.attentionPostNorm(), dim);
    for (int index = 0; index < dim; index++) {
      batchX[stateOffset + index] += attnProjected[index];
    }
    finishFinalLayerFfnRow(layer, stateOffset, dim, config.hiddenDim());
  }

  private void finishFinalLayerKvOnlyToken(
      LlamaWeights.LayerWeights layer,
      int layerIndex,
      int position,
      int keyLength,
      int numKvHeads,
      KvCache sequenceCache) {
    if (groupedProjections) {
      dualMatmulDispatch(
          k,
          layer.wk(),
          layer.wkType(),
          config.keyDim(),
          v,
          layer.wv(),
          layer.wvType(),
          config.valueDim(),
          xNorm,
          config.embeddingDim());
    } else {
      matmulDispatch(k, xNorm, layer.wk(), layer.wkType(), config.keyDim(), config.embeddingDim());
      matmulDispatch(
          v, xNorm, layer.wv(), layer.wvType(), config.valueDim(), config.embeddingDim());
    }
    addOptionalBias(k, layer.kBias());
    addOptionalBias(v, layer.vBias());
    for (int head = 0; head < numKvHeads; head++) {
      int offset = head * keyLength;
      normalizeHead(k, offset, layer.kNorm(), keyLength);
      if (config.usesRope(layerIndex)) {
        applyRope(k, offset, layerIndex);
      }
    }
    sequenceCache.store(layerIndex, position, k, v);
  }

  private void finishFinalLayerFfnRow(
      LlamaWeights.LayerWeights layer, int stateOffset, int dim, int hiddenDim) {
    TensorOps.rmsNorm(xNorm, 0, batchX, stateOffset, layer.ffnNorm(), dim, config.rmsNormEps());
    if (groupedProjections) {
      dualMatmulDispatch(
          ffnGate,
          layer.ffnGate(),
          layer.ffnGateType(),
          hiddenDim,
          ffnUp,
          layer.ffnUp(),
          layer.ffnUpType(),
          hiddenDim,
          xNorm,
          dim);
    } else {
      matmulDispatch(ffnGate, xNorm, layer.ffnGate(), layer.ffnGateType(), hiddenDim, dim);
      matmulDispatch(ffnUp, xNorm, layer.ffnUp(), layer.ffnUpType(), hiddenDim, dim);
    }
    activateFfn(ffnOut, 0, ffnGate, 0, ffnUp, 0, hiddenDim);
    matmulDispatch(ffnProjected, ffnOut, layer.ffnDown(), layer.ffnDownType(), dim, hiddenDim);
    normalizeProjection(ffnProjected, layer.ffnPostNorm(), dim);
    for (int index = 0; index < dim; index++) {
      batchX[stateOffset + index] += ffnProjected[index];
    }
  }

  private void prepareBatchedAttention(
      LlamaWeights.LayerWeights layer,
      int layerIndex,
      int startPosition,
      int fromBatch,
      int toBatch) {
    int queryDim = config.queryDim();
    int keyDim = config.keyDim();
    int valueDim = config.valueDim();
    int keyLength = config.keyLength();
    for (int batch = fromBatch; batch < toBatch; batch++) {
      int qBase = batch * queryDim;
      int kBase = batch * keyDim;
      int vBase = batch * valueDim;
      addOptionalBias(batchQ, qBase, layer.qBias());
      addOptionalBias(batchK, kBase, layer.kBias());
      addOptionalBias(batchV, vBase, layer.vBias());

      for (int head = 0; head < config.numHeads(); head++) {
        int offset = qBase + head * keyLength;
        normalizeHead(batchQ, offset, layer.qNorm(), keyLength);
        if (config.usesRope(layerIndex)) {
          applyRopeBatch(batchQ, offset, batch, layerIndex);
        }
      }
      for (int head = 0; head < config.numKvHeads(); head++) {
        int offset = kBase + head * keyLength;
        normalizeHead(batchK, offset, layer.kNorm(), keyLength);
        if (config.usesRope(layerIndex)) {
          applyRopeBatch(batchK, offset, batch, layerIndex);
        }
      }
      cache.store(layerIndex, startPosition + batch, batchK, kBase, batchV, vBase);
    }
  }

  private void computeBatchedAttention(
      LlamaWeights.LayerWeights ignoredLayer,
      int layerIndex,
      int startPosition,
      int fromBatch,
      int toBatch) {
    int batchSize = toBatch - fromBatch;
    int chunkStartPosition = startPosition + fromBatch;
    int slidingWindow = config.usesSlidingWindow(layerIndex) ? config.slidingWindow() : 0;
    if (fromBatch == 0
        && batchedAttentionKernel.isEligible(
            layerIndex,
            chunkStartPosition,
            batchSize,
            config.numHeads(),
            config.numKvHeads(),
            config.keyLength(),
            config.valueLength(),
            cache.maxSeqLen(),
            slidingWindow)) {
      batchedAttentionKernel.attend(
          batchAttnOut,
          batchQ,
          batchK,
          batchV,
          layerIndex,
          chunkStartPosition,
          batchSize,
          config.numHeads(),
          config.numKvHeads(),
          config.keyLength(),
          config.valueLength(),
          cache.maxSeqLen(),
          slidingWindow);
      return;
    }
    float[] keyCache = cache.keyBuffer();
    float[] valueCache = cache.valueBuffer();
    boolean separateScores = batchAttentionScores.length != 0;
    int scoreStride = cache.maxSeqLen();
    for (int batch = fromBatch; batch < toBatch; batch++) {
      float[] scores = separateScores ? batchAttentionScores : attentionScores;
      int scoreOffset = separateScores ? batch * scoreStride : 0;
      groupedQueryAttention(
          batchQ,
          batch * config.queryDim(),
          batchAttnOut,
          batch * config.attentionOutputDim(),
          layerIndex,
          startPosition + batch,
          cache,
          keyCache,
          valueCache,
          scores,
          scoreOffset);
    }
  }

  private void groupedQueryAttention(
      float[] query,
      int queryOffset,
      float[] output,
      int outputOffset,
      int layer,
      int position,
      KvCache sequenceCache,
      float[] keyCache,
      float[] valueCache,
      float[] scores,
      int scoresOffset) {
    int keyLength = config.keyLength();
    int valueLength = config.valueLength();
    int numHeads = config.numHeads();
    int groupSize = numHeads / config.numKvHeads();
    float scale = (float) (1.0 / Math.sqrt(keyLength));
    java.util.Arrays.fill(output, outputOffset, outputOffset + config.attentionOutputDim(), 0.0f);

    for (int head = 0; head < numHeads; head++) {
      int kvHead = head / groupSize;
      int qOffset = queryOffset + head * keyLength;
      int headOutputOffset = outputOffset + head * valueLength;
      int firstPosition = config.attentionStartPosition(layer, position);
      computeAttentionScores(
          query,
          qOffset,
          layer,
          position,
          kvHead,
          sequenceCache,
          keyCache,
          keyLength,
          scale,
          scores,
          scoresOffset);

      TensorOps.softmax(scores, scoresOffset + firstPosition, position - firstPosition + 1);
      accumulateAttentionValues(
          output,
          headOutputOffset,
          layer,
          position,
          kvHead,
          sequenceCache,
          valueCache,
          valueLength,
          scores,
          scoresOffset);
    }
  }

  private void computeAttentionScores(
      float[] query,
      int queryOffset,
      int layer,
      int position,
      int kvHead,
      KvCache sequenceCache,
      float[] keyCache,
      int keyLength,
      float scale,
      float[] scores,
      int scoresOffset) {
    int firstPosition = config.attentionStartPosition(layer, position);
    int visiblePositions = position - firstPosition + 1;
    int firstKeyOffset = sequenceCache.keyOffset(layer, firstPosition) + kvHead * keyLength;
    if (batchedAttentionScores) {
      VectorUtil.batchDotProductExact(
          query,
          queryOffset,
          keyCache,
          firstKeyOffset,
          sequenceCache.keyDim(),
          visiblePositions,
          keyLength,
          scores,
          scoresOffset + firstPosition);
      for (int cachedPosition = firstPosition; cachedPosition <= position; cachedPosition++) {
        scores[scoresOffset + cachedPosition] *= scale;
      }
      return;
    }
    for (int cachedPosition = firstPosition; cachedPosition <= position; cachedPosition++) {
      int cacheOffset = firstKeyOffset + (cachedPosition - firstPosition) * sequenceCache.keyDim();
      float dot = VectorUtil.dotProduct(query, queryOffset, keyCache, cacheOffset, keyLength);
      scores[scoresOffset + cachedPosition] = dot * scale;
    }
  }

  private void accumulateAttentionValues(
      float[] output,
      int outputOffset,
      int layer,
      int position,
      int kvHead,
      KvCache sequenceCache,
      float[] valueCache,
      int valueLength,
      float[] scores,
      int scoresOffset) {
    int firstPosition = config.attentionStartPosition(layer, position);
    int visiblePositions = position - firstPosition + 1;
    if (batchedAttentionValues) {
      int firstValueOffset = sequenceCache.valueOffset(layer, firstPosition) + kvHead * valueLength;
      VectorUtil.addWeightedRowsInPlace(
          output,
          outputOffset,
          valueCache,
          firstValueOffset,
          sequenceCache.valueDim(),
          scores,
          scoresOffset + firstPosition,
          visiblePositions,
          valueLength);
      return;
    }
    for (int cachedPosition = firstPosition; cachedPosition <= position; cachedPosition++) {
      int cacheOffset = sequenceCache.valueOffset(layer, cachedPosition) + kvHead * valueLength;
      VectorUtil.addScaledInPlace(
          output,
          outputOffset,
          valueCache,
          cacheOffset,
          valueLength,
          scores[scoresOffset + cachedPosition]);
    }
  }

  private void normalizeHead(float[] vector, int offset, float[] weight, int headDim) {
    if (weight.length != 0) {
      TensorOps.rmsNorm(vector, offset, vector, offset, weight, headDim, config.rmsNormEps());
    }
  }

  private void prepareRope(int position) {
    globalRopeTable.prepare(position);
    if (slidingWindowRopeTable != globalRopeTable) {
      slidingWindowRopeTable.prepare(position);
    }
  }

  private void prepareRopeBatch(int startPosition, int batchSize) {
    globalRopeTable.prepareBatch(startPosition, batchSize);
    if (slidingWindowRopeTable != globalRopeTable) {
      slidingWindowRopeTable.prepareBatch(startPosition, batchSize);
    }
  }

  private void prepareRopePositions(int[] positions, int count) {
    globalRopeTable.preparePositions(positions, count);
    if (slidingWindowRopeTable != globalRopeTable) {
      slidingWindowRopeTable.preparePositions(positions, count);
    }
  }

  private RotaryTable ropeTable(int layer) {
    return config.usesSlidingWindow(layer) ? slidingWindowRopeTable : globalRopeTable;
  }

  private void applyRope(float[] vector, int offset, int layer) {
    ropeTable(layer).apply(vector, offset, config.usesNeoxRope());
  }

  private void applyRopeBatch(float[] vector, int offset, int batch, int layer) {
    ropeTable(layer).applyBatch(vector, offset, batch, config.usesNeoxRope());
  }

  private void normalizeProjection(float[] projection, float[] norm, int size) {
    if (norm.length != 0) {
      TensorOps.rmsNorm(projection, projection, norm, size, config.rmsNormEps());
    }
  }

  private void normalizeProjectionBatch(float[] projection, int batchSize, int size, float[] norm) {
    if (norm.length == 0) {
      return;
    }
    for (int batch = 0; batch < batchSize; batch++) {
      int offset = batch * size;
      TensorOps.rmsNorm(projection, offset, projection, offset, norm, size, config.rmsNormEps());
    }
  }

  private void activateFfn(
      float[] out,
      int outOffset,
      float[] gate,
      int gateOffset,
      float[] up,
      int upOffset,
      int size) {
    if (config.usesGeluFfn()) {
      TensorOps.geluGlu(out, outOffset, gate, gateOffset, up, upOffset, size);
    } else if (usesVectorizedSwiGlu()) {
      VectorUtil.swiGlu(out, outOffset, gate, gateOffset, up, upOffset, size);
    } else {
      TensorOps.swiGlu(out, outOffset, gate, gateOffset, up, upOffset, size);
    }
  }

  private boolean usesVectorizedSwiGlu() {
    return config.architecture() == DecoderArchitecture.QWEN2
        || config.architecture() == DecoderArchitecture.QWEN3;
  }

  private static void scaleActive(float[] values, int offset, int length, float scale) {
    if (scale == 1.0f) {
      return;
    }
    for (int index = 0; index < length; index++) {
      values[offset + index] *= scale;
    }
  }

  private void softcapLogits(float[] values) {
    softcapLogits(values, 0, values.length);
  }

  private void softcapLogits(float[] values, int offset, int length) {
    float softcap = config.finalLogitSoftcap();
    if (softcap == 0.0f) {
      return;
    }
    int limit = Math.addExact(offset, length);
    for (int index = offset; index < limit; index++) {
      values[index] = softcap * (float) Math.tanh(values[index] / softcap);
    }
  }

  private static void addOptionalBias(float[] vector, float[] bias) {
    if (bias.length != 0) {
      VectorUtil.addInPlace(vector, bias);
    }
  }

  private static void addOptionalBias(float[] vector, int offset, float[] bias) {
    for (int index = 0; index < bias.length; index++) {
      vector[offset + index] += bias[index];
    }
  }

  private static void addActiveInPlace(float[] target, float[] addend, int length) {
    for (int index = 0; index < length; index++) {
      target[index] += addend[index];
    }
  }

  private void matmulDispatch(
      float[] out, float[] input, MemorySegment weight, GgufTensorType type, int rows, int cols) {
    if (batchedMatrixKernel.isEligible(type, 1, rows, cols)) {
      batchedMatrixKernel.multiply(out, input, weight, type, 1, rows, cols);
      return;
    }
    TensorOps.ggufMatmul(
        out,
        input,
        weight,
        type,
        rows,
        cols,
        quantizedActivation,
        quantizedActivationScales,
        quantizedActivationZeroPointCorrections,
        quantizedActivationSums,
        q4Kernel);
  }

  private void dualMatmulDispatch(
      float[] firstOut,
      MemorySegment firstWeight,
      GgufTensorType firstType,
      int firstRows,
      float[] secondOut,
      MemorySegment secondWeight,
      GgufTensorType secondType,
      int secondRows,
      float[] input,
      int cols) {
    if (batchedMatrixKernel.isDualEligible(firstType, firstRows, secondType, secondRows, 1, cols)) {
      batchedMatrixKernel.multiplyDual(
          firstOut,
          firstWeight,
          firstType,
          firstRows,
          secondOut,
          secondWeight,
          secondType,
          secondRows,
          input,
          1,
          cols);
      return;
    }
    TensorOps.ggufDualMatmul(
        firstOut,
        firstWeight,
        firstType,
        firstRows,
        secondOut,
        secondWeight,
        secondType,
        secondRows,
        input,
        cols,
        quantizedActivation,
        quantizedActivationScales,
        quantizedActivationZeroPointCorrections,
        quantizedActivationSums,
        q4Kernel);
  }

  private void tripleMatmulDispatch(
      float[] firstOut,
      MemorySegment firstWeight,
      GgufTensorType firstType,
      int firstRows,
      float[] secondOut,
      MemorySegment secondWeight,
      GgufTensorType secondType,
      int secondRows,
      float[] thirdOut,
      MemorySegment thirdWeight,
      GgufTensorType thirdType,
      int thirdRows,
      float[] input,
      int cols) {
    if (batchedMatrixKernel.isTripleEligible(
        firstType, firstRows, secondType, secondRows, thirdType, thirdRows, 1, cols)) {
      batchedMatrixKernel.multiplyTriple(
          firstOut,
          firstWeight,
          firstType,
          firstRows,
          secondOut,
          secondWeight,
          secondType,
          secondRows,
          thirdOut,
          thirdWeight,
          thirdType,
          thirdRows,
          input,
          1,
          cols);
      return;
    }
    TensorOps.ggufTripleMatmul(
        firstOut,
        firstWeight,
        firstType,
        firstRows,
        secondOut,
        secondWeight,
        secondType,
        secondRows,
        thirdOut,
        thirdWeight,
        thirdType,
        thirdRows,
        input,
        cols,
        quantizedActivation,
        quantizedActivationScales,
        quantizedActivationZeroPointCorrections,
        quantizedActivationSums,
        q4Kernel,
        mixedKProjections);
  }

  private void dualBatchedMatmulDispatch(
      float[] firstOut,
      MemorySegment firstWeight,
      GgufTensorType firstType,
      int firstRows,
      float[] secondOut,
      MemorySegment secondWeight,
      GgufTensorType secondType,
      int secondRows,
      float[] input,
      int batchSize,
      int cols) {
    if (batchedMatrixKernel.isDualEligible(
        firstType, firstRows, secondType, secondRows, batchSize, cols)) {
      batchedMatrixKernel.multiplyDual(
          firstOut,
          firstWeight,
          firstType,
          firstRows,
          secondOut,
          secondWeight,
          secondType,
          secondRows,
          input,
          batchSize,
          cols);
      return;
    }
    TensorOps.ggufDualBatchedMatmul(
        firstOut,
        firstWeight,
        firstType,
        firstRows,
        secondOut,
        secondWeight,
        secondType,
        secondRows,
        input,
        batchSize,
        cols,
        batchQuantizedActivation,
        batchQuantizedActivationScales,
        batchQuantizedActivationZeroPointCorrections,
        batchQuantizedActivationSums,
        batchQ4LaneScratch,
        q4Kernel);
  }

  private void tripleBatchedMatmulDispatch(
      float[] firstOut,
      MemorySegment firstWeight,
      GgufTensorType firstType,
      int firstRows,
      float[] secondOut,
      MemorySegment secondWeight,
      GgufTensorType secondType,
      int secondRows,
      float[] thirdOut,
      MemorySegment thirdWeight,
      GgufTensorType thirdType,
      int thirdRows,
      float[] input,
      int batchSize,
      int cols) {
    if (batchedMatrixKernel.isTripleEligible(
        firstType, firstRows, secondType, secondRows, thirdType, thirdRows, batchSize, cols)) {
      batchedMatrixKernel.multiplyTriple(
          firstOut,
          firstWeight,
          firstType,
          firstRows,
          secondOut,
          secondWeight,
          secondType,
          secondRows,
          thirdOut,
          thirdWeight,
          thirdType,
          thirdRows,
          input,
          batchSize,
          cols);
      return;
    }
    TensorOps.ggufTripleBatchedMatmul(
        firstOut,
        firstWeight,
        firstType,
        firstRows,
        secondOut,
        secondWeight,
        secondType,
        secondRows,
        thirdOut,
        thirdWeight,
        thirdType,
        thirdRows,
        input,
        batchSize,
        cols,
        batchQuantizedActivation,
        batchQuantizedActivationScales,
        batchQuantizedActivationZeroPointCorrections,
        batchQuantizedActivationSums,
        batchQ4LaneScratch,
        q4Kernel,
        q6BatchedKernel,
        mixedKProjections);
  }

  private void batchedMatmulDispatch(
      float[] out,
      float[] input,
      int batchSize,
      MemorySegment weight,
      GgufTensorType type,
      int rows,
      int cols) {
    if (batchedMatrixKernel.isEligible(type, batchSize, rows, cols)) {
      batchedMatrixKernel.multiply(out, input, weight, type, batchSize, rows, cols);
      return;
    }
    TensorOps.ggufBatchedMatmul(
        out,
        input,
        weight,
        type,
        batchSize,
        rows,
        cols,
        batchQuantizedActivation,
        batchQuantizedActivationScales,
        batchQuantizedActivationZeroPointCorrections,
        batchQuantizedActivationSums,
        batchQ4LaneScratch,
        q4Kernel,
        q6BatchedKernel);
  }

  private static boolean usesProjectionType(
      LlamaConfig config, LlamaWeights weights, GgufTensorType type) {
    for (int layer = 0; layer < config.numLayers(); layer++) {
      LlamaWeights.LayerWeights layerWeights = weights.layer(layer);
      if (layerWeights.wqType() == type
          || layerWeights.wkType() == type
          || layerWeights.wvType() == type
          || layerWeights.woType() == type
          || layerWeights.ffnGateType() == type
          || layerWeights.ffnUpType() == type
          || layerWeights.ffnDownType() == type) {
        return true;
      }
    }
    return false;
  }

  private static boolean usesGroupedBatchedProjection(
      LlamaConfig config, LlamaWeights weights, boolean mixedKProjections) {
    for (int layer = 0; layer < config.numLayers(); layer++) {
      LlamaWeights.LayerWeights layerWeights = weights.layer(layer);
      if (layerWeights.ffnGateType() == layerWeights.ffnUpType()
          && TensorOps.supportsGroupedBatchedMatmul(layerWeights.ffnGateType())) {
        return true;
      }
      if (mixedKProjections
          && layerWeights.wqType() == GgufTensorType.Q4_K
          && layerWeights.wkType() == GgufTensorType.Q4_K
          && layerWeights.wvType() == GgufTensorType.Q6_K) {
        return true;
      }
      if ((layerWeights.wqType() == layerWeights.wkType()
              && TensorOps.supportsGroupedBatchedMatmul(layerWeights.wqType()))
          || (layerWeights.wqType() == layerWeights.wvType()
              && TensorOps.supportsGroupedBatchedMatmul(layerWeights.wqType()))
          || (layerWeights.wkType() == layerWeights.wvType()
              && TensorOps.supportsGroupedBatchedMatmul(layerWeights.wkType()))) {
        return true;
      }
    }
    return false;
  }

  private static int maxGroupedQ4ProjectionRows(LlamaConfig config, LlamaWeights weights) {
    int maxRows = 0;
    for (int layer = 0; layer < config.numLayers(); layer++) {
      LlamaWeights.LayerWeights layerWeights = weights.layer(layer);
      int qkvRows = 0;
      if (layerWeights.wqType() == GgufTensorType.Q4_0) {
        qkvRows = Math.addExact(qkvRows, config.queryDim());
      }
      if (layerWeights.wkType() == GgufTensorType.Q4_0) {
        qkvRows = Math.addExact(qkvRows, config.keyDim());
      }
      if (layerWeights.wvType() == GgufTensorType.Q4_0) {
        qkvRows = Math.addExact(qkvRows, config.valueDim());
      }
      maxRows = Math.max(maxRows, qkvRows);

      int gateUpRows = 0;
      if (layerWeights.ffnGateType() == GgufTensorType.Q4_0) {
        gateUpRows = Math.addExact(gateUpRows, config.hiddenDim());
      }
      if (layerWeights.ffnUpType() == GgufTensorType.Q4_0) {
        gateUpRows = Math.addExact(gateUpRows, config.hiddenDim());
      }
      maxRows = Math.max(maxRows, gateUpRows);
      if (layerWeights.woType() == GgufTensorType.Q4_0
          || layerWeights.ffnDownType() == GgufTensorType.Q4_0) {
        maxRows = Math.max(maxRows, config.embeddingDim());
      }
    }
    return maxRows;
  }

  private static float[] batchBuffer(int batchCapacity, int width) {
    return new float[Math.multiplyExact(batchCapacity, width)];
  }

  private static PureJavaExecutionPlan defaultPlan(LlamaConfig config, LlamaWeights weights) {
    return ExecutionPlanner.plan(
        RuntimeFingerprint.capture(),
        ModelTopology.from(config.architecture().metadataId(), config, weights),
        PureJavaPlanConfiguration.fromSystemProperties(Map.of()));
  }
}
