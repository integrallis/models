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
package com.integrallis.models.backend.purejava.lfm2;

import com.integrallis.models.backend.purejava.cache.SequenceGrowth;
import com.integrallis.models.backend.purejava.gguf.GgufFile;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.ops.ExpertRouting;
import com.integrallis.models.backend.purejava.ops.GroupedQueryAttentionKernel;
import com.integrallis.models.backend.purejava.ops.RotaryTable;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import com.integrallis.vectors.core.GgufQ4Kernel;
import com.integrallis.vectors.core.VectorUtil;
import java.lang.foreign.MemorySegment;
import java.util.Arrays;
import java.util.Objects;

/**
 * The LFM2 decoder: a hybrid of attention layers and gated short convolutions.
 *
 * <p>Every layer has the same outer shape -- norm, mixer, residual, norm, gated feed-forward,
 * residual -- and only the mixer differs. Six of LFM2.5-1.2B's sixteen layers attend; the rest run
 * {@link Lfm2ShortConv}, which carries a recurrent state instead of a key-value cache. That is the
 * reason this is its own decoder rather than an addition to the Llama one: two kinds of per-layer
 * state with different shapes and different update rules.
 *
 * <p>Serial and single-sequence on purpose. A batched prefill would have to advance the convolution
 * state token by token anyway -- the recurrence is inherently sequential -- so the batched form
 * saves only the projections, and getting it right is worth less than getting the recurrence right
 * first.
 *
 * <p>Rope is the NeoX half-split layout, read from llama.cpp's architecture table, and is applied
 * after the per-head query and key norms, matching the reference's order.
 */
public final class Lfm2ForwardPass {

  private final Lfm2Config config;
  private final Lfm2Weights weights;
  private final RotaryTable rotary;
  private final int maxSeqLen;

  private final float[] state;
  private final float[] normalized;
  private final float[] query;
  private final float[] key;
  private final float[] value;
  private final float[] attentionOut;
  private final float[] projected;
  private float[] scores;

  /**
   * Row stride into {@link #scores}: the allocated capacity, which grows, not {@code maxSeqLen}.
   */
  private int scoreStride;

  private int allocatedCapacity;
  private final float[] gate;
  private final float[] up;
  private final float[] ffnOut;
  private final float[] convProjected;

  /**
   * Per-layer residual observer, mirroring {@code LlamaForwardPass.LayerObserver}.
   *
   * <p>Added because this decoder had no way to see inside it, and that cost a diagnosis. The
   * {@code lfm2moe} port generated {@code " France is France is"} where llama.cpp generated {@code
   * " Paris. The city of Paris is known for its beautiful landmarks"}, and with the dense path
   * green on a real artifact and the routed path green on a toy one, there was nothing left to
   * narrow it with: a 24-layer model that is wrong somewhere needs a per-layer comparison, and this
   * decoder could not produce one.
   *
   * <p>Null when nobody is watching, so the hot path pays one null check per layer.
   */
  interface LayerObserver {
    void onLayerComplete(int layer, int position, float[] state, int offset, int length);
  }

  private LayerObserver layerObserver;

  /**
   * The injected batched matrix kernel, or {@link GgufBatchedMatrixKernel#none()}.
   *
   * <p><b>This decoder had no kernel seam at all.</b> Every projection went through {@code
   * TensorOps.ggufMatmul}, and {@code PureJavaBackend.loadLfm2} never passed the kernel it had
   * already resolved -- so an lfm2 or lfm2moe model ran with no native acceleration even when the
   * caller injected one and the diagnostics reported it. Measured 2026-11-01: adding the seam took
   * lfm2moe's decode from 8.94 to 23.80 tok/s, our own arm before and after on one host under one
   * protocol -- a paired within-arm A/B, not a gate result. Its admitted qualification, against
   * ollama at warmups 1 and iterations 3, is {@code decodeThroughputRatio} 0.3974 against a 0.80
   * floor, with prefill the worse half at 0.1833 (25.54 against 139.31 tok/s).
   */
  private final GgufBatchedMatrixKernel matrixKernel;

  // Grouped expert dispatch scratch, allocated once: a routed layer runs every token of every
  // layer, so an allocation here is an allocation in the hot path. Null on a dense lfm2.
  // Q, K and V share one input, so they are one grouped dispatch. Allocated once beside the
  // expert scratch for the same reason: a routed layer runs this every token of every sequence.
  private final float[][] qkvOutputs;
  private final MemorySegment[] qkvWeights;
  private final GgufTensorType[] qkvTypes;
  private final int[] qkvRows;

  private final float[][] groupedGateUp;
  private final MemorySegment[] groupedGateUpWeights;
  private final GgufTensorType[] groupedGateUpTypes;
  private final int[] groupedGateUpRows;
  private final float[][] groupedActivations;
  private final float[][] groupedDownOutputs;
  private final MemorySegment[] groupedDownWeights;
  private final GgufTensorType[] groupedDownTypes;
  private final int[] groupedDownRows;
  private final float[] routerLogits;
  private final int[] selectedExperts;
  private final float[] routingWeights;
  private final float[] expertOut;
  private final float[] convGated;
  private final float[] logits;

  private final byte[] quantizedActivation;
  private final float[] quantizedActivationScales;
  private final int[] quantizedActivationZeroPointCorrections;
  private final short[] quantizedActivationSums;

  /** Key and value history, allocated only for the layers that attend. */
  private final float[][] keyCache;

  private final float[][] valueCache;

  /** The convolution's shift register, allocated only for the layers that do not attend. */
  private final float[][] convState;

  private int nextPosition;

  /** Attaches a per-layer observer; pass null to detach. Diagnostics only. */
  void observeLayers(LayerObserver observer) {
    this.layerObserver = observer;
  }

  /**
   * Loads a decoder from a GGUF, keeping the weight layout package-private.
   *
   * @param file the parsed GGUF
   * @param config the parsed LFM2 contract
   * @param maxSeqLen the sequence limit to allocate for
   * @return a ready decoder
   */
  public static Lfm2ForwardPass fromGgufFile(GgufFile file, Lfm2Config config, int maxSeqLen) {
    return fromGgufFile(file, config, maxSeqLen, GgufBatchedMatrixKernel.none());
  }

  /** Loads a decoder that projects through the injected kernel where it is eligible. */
  public static Lfm2ForwardPass fromGgufFile(
      GgufFile file, Lfm2Config config, int maxSeqLen, GgufBatchedMatrixKernel matrixKernel) {
    return new Lfm2ForwardPass(
        config, Lfm2Weights.fromGgufFile(file, config), maxSeqLen, matrixKernel);
  }

  Lfm2ForwardPass(Lfm2Config config, Lfm2Weights weights, int maxSeqLen) {
    this(config, weights, maxSeqLen, GgufBatchedMatrixKernel.none());
  }

  Lfm2ForwardPass(
      Lfm2Config config, Lfm2Weights weights, int maxSeqLen, GgufBatchedMatrixKernel matrixKernel) {
    this.matrixKernel = Objects.requireNonNull(matrixKernel, "matrixKernel");
    this.config = Objects.requireNonNull(config, "config");
    this.weights = Objects.requireNonNull(weights, "weights");
    if (maxSeqLen <= 0 || maxSeqLen > config.contextLength()) {
      throw new IllegalArgumentException(
          "maxSeqLen must be in 1.." + config.contextLength() + ", was " + maxSeqLen);
    }
    this.maxSeqLen = maxSeqLen;
    this.rotary = new RotaryTable(config.headDim(), config.ropeTheta(), 1.0f);

    int dim = config.embeddingDim();
    this.state = new float[dim];
    this.normalized = new float[dim];
    this.query = new float[config.queryDim()];
    this.attentionOut = new float[config.queryDim()];
    this.projected = new float[dim];
    // One row per query head: scoreGroup writes a whole group at once, so the per-head rows
    // must coexist rather than share one buffer as the scalar loop's did.
    // Grown with the caches rather than sized at the ceiling: see ensureCapacity.
    this.scores = new float[0];
    this.scoreStride = 0;
    this.allocatedCapacity = 0;
    // The dense feed-forward is feed_forward_length wide and an expert is
    // expert_feed_forward_length wide, and on lfm2moe those differ (7168 against 1792). One buffer
    // sized for the wider of the two serves both branches; sizing it from hiddenDim alone would
    // overflow on a file whose experts are wider than its dense blocks.
    int ffnWidth =
        config.isMixtureOfExperts()
            ? Math.max(config.hiddenDim(), config.experts().hiddenDim())
            : config.hiddenDim();
    this.gate = new float[ffnWidth];
    this.up = new float[ffnWidth];
    // Hidden-width, not model-width: SwiGLU produces one value per hidden unit and only the down
    // projection returns to the model width.
    this.ffnOut = new float[ffnWidth];
    this.routerLogits = config.isMixtureOfExperts() ? new float[config.experts().count()] : null;
    this.selectedExperts =
        config.isMixtureOfExperts() ? new int[config.experts().usedCount()] : null;
    this.routingWeights =
        config.isMixtureOfExperts() ? new float[config.experts().usedCount()] : null;
    this.expertOut = config.isMixtureOfExperts() ? new float[config.embeddingDim()] : null;
    if (config.isMixtureOfExperts()) {
      int used = config.experts().usedCount();
      int expertWidth = config.experts().hiddenDim();
      this.groupedGateUp = new float[2 * used][expertWidth];
      this.groupedGateUpWeights = new MemorySegment[2 * used];
      this.groupedGateUpTypes = new GgufTensorType[2 * used];
      this.groupedGateUpRows = new int[2 * used];
      this.groupedActivations = new float[used][expertWidth];
      this.groupedDownOutputs = new float[used][config.embeddingDim()];
      this.groupedDownWeights = new MemorySegment[used];
      this.groupedDownTypes = new GgufTensorType[used];
      this.groupedDownRows = new int[used];
    } else {
      this.groupedGateUp = null;
      this.groupedGateUpWeights = null;
      this.groupedGateUpTypes = null;
      this.groupedGateUpRows = null;
      this.groupedActivations = null;
      this.groupedDownOutputs = null;
      this.groupedDownWeights = null;
      this.groupedDownTypes = null;
      this.groupedDownRows = null;
    }
    // Allocated whenever the file has attention layers at all: grouping is decided per call from
    // the quantisations, not from the architecture, so the scratch is cheap and the check is late.
    this.qkvOutputs = new float[3][];
    this.qkvWeights = new MemorySegment[3];
    this.qkvTypes = new GgufTensorType[3];
    this.qkvRows = new int[3];
    this.convProjected = new float[3 * dim];
    this.convGated = new float[dim];
    this.logits = new float[config.vocabSize()];

    int maxKeyDim = 0;
    for (int layer = 0; layer < config.numLayers(); layer++) {
      if (config.usesAttention(layer)) {
        maxKeyDim = Math.max(maxKeyDim, config.keyDim(layer));
      }
    }
    this.key = new float[maxKeyDim];
    this.value = new float[maxKeyDim];

    int widest = Math.max(Math.max(dim, config.hiddenDim()), config.queryDim());
    this.quantizedActivation = new byte[widest];
    this.quantizedActivationScales = new float[(widest + 31) / 32];
    this.quantizedActivationZeroPointCorrections = new int[(widest + 3) / 4];
    this.quantizedActivationSums = new short[(widest + 15) / 16];

    this.keyCache = new float[config.numLayers()][];
    this.valueCache = new float[config.numLayers()][];
    this.convState = new float[config.numLayers()][];
    // An attention layer's keys and values are grown by ensureCapacity at the first position
    // stored. Convolution state is sized by the architecture, so it is allocated once here.
    for (int layer = 0; layer < config.numLayers(); layer++) {
      if (!config.usesAttention(layer)) {
        convState[layer] = new float[config.shortConvStateSize()];
      }
    }
  }

  /**
   * Grows key, value and score storage so every position below {@code positions} can be stored.
   *
   * <p>Sixteen, doubling, never past {@code maxSeqLen}: the same policy as {@code KvCache} and
   * {@code LayeredKvCache}, which this graph does not route through. Note that {@link #scores} is
   * strided by the allocated capacity rather than by {@code maxSeqLen}, so growing it moves every
   * head's row -- which is why nothing may read scores across a growth, and nothing does: a score
   * row is written and consumed within one call to {@code attend}.
   */
  private void ensureCapacity(int positions) {
    if (positions <= allocatedCapacity) {
      return;
    }
    int grown = SequenceGrowth.nextCapacity(allocatedCapacity, positions, maxSeqLen);
    for (int layer = 0; layer < config.numLayers(); layer++) {
      if (config.usesAttention(layer)) {
        keyCache[layer] = growRows(keyCache[layer], grown, config.keyDim(layer));
        valueCache[layer] = growRows(valueCache[layer], grown, config.keyDim(layer));
      }
    }
    scores = new float[Math.multiplyExact(config.numHeads(), grown)];
    scoreStride = grown;
    allocatedCapacity = grown;
  }

  /** Reallocates one cache at {@code capacity} rows, carrying the stored rows over. */
  private static float[] growRows(float[] existing, int capacity, int dimension) {
    float[] grown = new float[Math.multiplyExact(capacity, dimension)];
    if (existing != null) {
      System.arraycopy(existing, 0, grown, 0, existing.length);
    }
    return grown;
  }

  /** Clears both kinds of state, so a new sequence starts from position zero. */
  public void reset() {
    nextPosition = 0;
    for (int layer = 0; layer < config.numLayers(); layer++) {
      if (convState[layer] != null) {
        java.util.Arrays.fill(convState[layer], 0.0f);
      }
    }
  }

  public int nextPosition() {
    return nextPosition;
  }

  /**
   * Advances one token and returns the vocabulary logits.
   *
   * @param token the token id
   * @param position the position, which must be the next one in sequence
   * @return the logits, valid until the next call
   */
  public float[] forward(int token, int position) {
    return forwardInternal(token, position, true);
  }

  /**
   * Advances one token and returns the final-norm activation instead of the vocabulary logits.
   *
   * <p>Same contract as {@code LlamaForwardPass.hiddenState}: the activation the output projection
   * would consume, with that projection skipped. Skipping it is the whole saving, since it is the
   * widest matmul in the pass, and it is what an embedding needs — LFM2.5-Embedding is an {@code
   * lfm2} decoder, so without this its seven published artifacts cannot be embedded at all.
   *
   * <p>The returned array is backend-owned scratch, valid until the next call. Copy it to keep it.
   *
   * @param token token id to advance with
   * @param position sequential position for the token
   * @return the final-norm activation, owned by this pass
   */
  public float[] hiddenState(int token, int position) {
    return forwardInternal(token, position, false);
  }

  private float[] forwardInternal(int token, int position, boolean projectLogits) {
    if (position != nextPosition) {
      throw new IllegalArgumentException(
          "position must be sequential: expected " + nextPosition + ", got " + position);
    }
    if (position >= maxSeqLen) {
      throw new IllegalArgumentException("position exceeds the sequence limit " + maxSeqLen);
    }
    ensureCapacity(position + 1);
    int dim = config.embeddingDim();
    weights.embedToken(token, state);

    for (int layer = 0; layer < config.numLayers(); layer++) {
      Lfm2Weights.LayerWeights layerWeights = weights.layer(layer);
      TensorOps.rmsNorm(
          normalized, state, layerWeights.attentionNorm(), dim, config.rmsNormEpsilon());
      // Both mixers leave their result in `projected`, so the residual is the same either way.
      if (config.usesAttention(layer)) {
        attend(layer, layerWeights, position);
      } else {
        convolve(layerWeights, layer);
      }
      add(state, projected, dim);

      TensorOps.rmsNorm(normalized, state, layerWeights.ffnNorm(), dim, config.rmsNormEpsilon());
      if (config.usesExperts(layer)) {
        route(layerWeights);
      } else {
        project(layerWeights.gateProjection(), normalized, gate);
        project(layerWeights.upProjection(), normalized, up);
        activate(ffnOut, 0, gate, 0, up, 0, config.hiddenDim());
        project(layerWeights.downProjection(), ffnOut, projected);
      }
      add(state, projected, dim);
      if (layerObserver != null) {
        layerObserver.onLayerComplete(layer, position, state, 0, dim);
      }
    }

    nextPosition++;
    // The final norm is the tensor named token_embd_norm; see Lfm2Weights.
    TensorOps.rmsNorm(normalized, state, weights.outputNorm(), dim, config.rmsNormEpsilon());
    if (!projectLogits) {
      return normalized;
    }
    project(
        weights.outputProjection(),
        weights.outputProjectionType(),
        config.vocabSize(),
        dim,
        normalized,
        logits);
    return logits;
  }

  /**
   * Q, K and V in one grouped dispatch instead of three, since all three read {@code normalized}.
   *
   * <p>MEASURED 2026-11-01. In lfm2moe's decode profile the NON-grouped kernel entry, {@code
   * NativeKernelLibrary.invokeBatched}, was the largest single consumer at 2469 native samples --
   * more than {@code invokeGrouped}'s 1556 and more than everything in Java. At batch 1 every
   * projection is a GEMV, so each separate downcall pays FFM crossing plus a fan-out and join
   * across sixteen worker threads for a matrix small enough that the fan-out is a material share of
   * the work. Fewer, larger calls is the lever; this is the same reasoning that took expert
   * dispatch from {@code 3 * usedCount} calls down to two.
   *
   * <p>{@code LlamaForwardPass} has had this since it gained {@code tripleMatmulDispatch}, gated on
   * {@code groupedProjections}. This decoder issued three {@code project} calls and so never got
   * it, which is the same shape of omission as the missing kernel seam and the scalar attention.
   *
   * <p>Grouping needs one shared input and compatible quantisations, so it declines rather than
   * forces: a file whose Q, K and V are not groupable together -- Q5_0 is grouped neither way --
   * falls back to the three calls, which is what the kernel would have done anyway.
   */
  private void projectQkv(Lfm2Weights.LayerWeights layerWeights) {
    Lfm2Weights.Matrix queryMatrix = layerWeights.queryProjection();
    Lfm2Weights.Matrix keyMatrix = layerWeights.keyProjection();
    Lfm2Weights.Matrix valueMatrix = layerWeights.valueProjection();
    if (qkvOutputs != null) {
      int columns = queryMatrix.columns();
      qkvWeights[0] = queryMatrix.data();
      qkvWeights[1] = keyMatrix.data();
      qkvWeights[2] = valueMatrix.data();
      qkvTypes[0] = queryMatrix.type();
      qkvTypes[1] = keyMatrix.type();
      qkvTypes[2] = valueMatrix.type();
      qkvRows[0] = queryMatrix.rows();
      qkvRows[1] = keyMatrix.rows();
      qkvRows[2] = valueMatrix.rows();
      if (keyMatrix.columns() == columns
          && valueMatrix.columns() == columns
          && matrixKernel.isGroupedEligible(qkvTypes, qkvRows, 3, 1, columns)) {
        qkvOutputs[0] = query;
        qkvOutputs[1] = key;
        qkvOutputs[2] = value;
        matrixKernel.multiplyGrouped(
            qkvOutputs, qkvWeights, qkvTypes, qkvRows, 3, normalized, 1, columns);
        return;
      }
    }
    project(queryMatrix, normalized, query);
    project(keyMatrix, normalized, key);
    project(valueMatrix, normalized, value);
  }

  private void attend(int layer, Lfm2Weights.LayerWeights layerWeights, int position) {
    int headDim = config.headDim();
    int numHeads = config.numHeads();
    int numKvHeads = config.numKvHeads(layer);
    int keyDim = config.keyDim(layer);

    projectQkv(layerWeights);

    // Per-head norms first, then rope: the reference's order.
    for (int head = 0; head < numHeads; head++) {
      int offset = head * headDim;
      TensorOps.rmsNorm(
          query, offset, query, offset, layerWeights.queryNorm(), headDim, config.rmsNormEpsilon());
    }
    for (int head = 0; head < numKvHeads; head++) {
      int offset = head * headDim;
      TensorOps.rmsNorm(
          key, offset, key, offset, layerWeights.keyNorm(), headDim, config.rmsNormEpsilon());
    }
    rotary.prepare(position);
    for (int head = 0; head < numHeads; head++) {
      rotary.apply(query, head * headDim, true);
    }
    for (int head = 0; head < numKvHeads; head++) {
      rotary.apply(key, head * headDim, true);
    }

    System.arraycopy(key, 0, keyCache[layer], position * keyDim, keyDim);
    System.arraycopy(value, 0, valueCache[layer], position * keyDim, keyDim);

    int groupSize = numHeads / numKvHeads;
    float scale = (float) (1.0 / Math.sqrt(headDim));
    float[] keys = keyCache[layer];
    float[] values = valueCache[layer];
    int rows = position + 1;
    // accumulateGroup reads its output and adds into it, so the destination starts at zero. The
    // scalar loop this replaces filled per head inside the loop; one fill up front is the same
    // thing for a buffer whose every head is about to be written.
    java.util.Arrays.fill(attentionOut, 0, numHeads * headDim, 0.0f);
    // One grouped-query call per KV head, not a scalar loop per query head.
    //
    // MEASURED 2026-11-01. The scalar triple loop this replaces was the single hottest frame in
    // lfm2moe's decode -- 460 of 1065 JFR self-time samples in `attend`, against 38 for every
    // native expert matmul put together. The expert dispatch was never the cost; this was. The
    // decoder had its own hand-rolled attention while GroupedQueryAttentionKernel, the vectorised
    // pinned-reduction kernel the Llama path already runs, sat unused in the same package -- the
    // same shape of defect as the missing kernel seam.
    //
    // Each K and V row is now read once for the whole group rather than once per query head, and
    // both reductions are pinned to eight logical lanes, so the result does not depend on the
    // host's vector width. It is NOT bit-identical to the scalar loop above: the score reduction
    // folds in a different order. That is acceptable here precisely because no qualified catalog
    // entry uses this decoder -- the catalog carries no lfm2 architecture, checked rather than
    // assumed -- so the parity that matters is against llama.cpp and not against our own previous
    // output, and it is re-measured on the greedy ladder after any change to this method.
    for (int kvHead = 0; kvHead < numKvHeads; kvHead++) {
      int headBase = kvHead * groupSize;
      GroupedQueryAttentionKernel.scoreGroup(
          query,
          headBase * headDim,
          headDim,
          groupSize,
          keys,
          kvHead * headDim,
          keyDim,
          rows,
          headDim,
          scale,
          scores,
          headBase * scoreStride,
          scoreStride);
      for (int offset = 0; offset < groupSize; offset++) {
        GroupedQueryAttentionKernel.softmax(scores, (headBase + offset) * scoreStride, rows);
      }
      GroupedQueryAttentionKernel.accumulateGroup(
          attentionOut,
          headBase * headDim,
          headDim,
          groupSize,
          values,
          kvHead * headDim,
          keyDim,
          rows,
          headDim,
          scores,
          headBase * scoreStride,
          scoreStride);
    }
    project(layerWeights.attentionOutputProjection(), attentionOut, projected);
  }

  /**
   * SwiGLU through the native kernel, falling back to the vectorised Java one.
   *
   * <p>MEASURED 2026-11-01: {@code TensorOps.swiGlu} is a scalar loop calling {@code Math.exp} once
   * per element, and it was 351 of 1065 JFR self-time samples in lfm2moe's decode -- second only to
   * attention, and nine times every native expert matmul combined. This decoder called it at all
   * three of its activation sites while the Rust kernel's {@code trySwiGlu} and {@code
   * VectorUtil.swiGlu} were both already reachable.
   *
   * <p>On the Llama path the same choice is gated by {@code usesVectorizedSwiGlu()} to QWEN2 and
   * QWEN3. That whitelist is about output parity, not speed: the vectorised form evaluates {@code
   * exp} by polynomial rather than by {@code Math.exp}, so an architecture switched onto it stops
   * reproducing the greedy outputs its published records were measured with, and every architecture
   * there keeps the activation it was qualified on. lfm2 is under no such constraint -- the catalog
   * carries no lfm2 entry -- so it takes the fast path, and the greedy ladder against llama.cpp is
   * what holds it, that being the reference that actually matters.
   */
  private void activate(
      float[] out,
      int outOffset,
      float[] gate,
      int gateOffset,
      float[] up,
      int upOffset,
      int size) {
    if (matrixKernel.trySwiGlu(out, outOffset, gate, gateOffset, up, upOffset, size)) {
      return;
    }
    VectorUtil.swiGlu(out, outOffset, gate, gateOffset, up, upOffset, size);
  }

  /**
   * The routed feed-forward: sigmoid gating with a selection-only bias, then a weighted sum.
   *
   * <p>Routing goes through {@link ExpertRouting#selectExpertsWithSigmoidGating} rather than a
   * local copy, because {@code expert_gating_func = 2} is the whole reason #283 was blocked and a
   * second implementation of it is a second thing to get subtly wrong. {@code lfm2moe} publishes no
   * {@code expert_weights_norm} and no {@code expert_weights_scale}, so the weights are
   * renormalised with a unit scale — which is what upstream's defaults give it.
   *
   * <p>Each expert is evaluated in turn and accumulated, rather than gathered into one batched
   * matmul. That is the slower shape and it is deliberate for a first port: the expert matrices are
   * already non-contiguous slices of one tensor, and a gather would add a copy whose correctness is
   * harder to see than the loop's. The kernel seam is unchanged, so the batched form is a later
   * change measurable against this one.
   */
  private void route(Lfm2Weights.LayerWeights layerWeights) {
    Lfm2Config.Experts experts = config.experts();
    int dim = config.embeddingDim();
    int expertWidth = experts.hiddenDim();

    project(layerWeights.router(), normalized, routerLogits);
    ExpertRouting.selectExpertsWithSigmoidGating(
        routerLogits, layerWeights.selectionBias(), selectedExperts, routingWeights, true, 1.0f);

    Arrays.fill(projected, 0, dim, 0.0f);
    if (routeGrouped(layerWeights, dim, expertWidth)) {
      return;
    }
    for (int slot = 0; slot < selectedExperts.length; slot++) {
      int expert = selectedExperts[slot];
      project(layerWeights.expertGate()[expert], normalized, gate);
      project(layerWeights.expertUp()[expert], normalized, up);
      activate(ffnOut, 0, gate, 0, up, 0, expertWidth);
      project(layerWeights.expertDown()[expert], ffnOut, expertOut);
      float weight = routingWeights[slot];
      for (int i = 0; i < dim; i++) {
        projected[i] += weight * expertOut[i];
      }
    }
  }

  /**
   * The selected experts in two native dispatches instead of {@code 3 * usedCount}.
   *
   * <p>Same change as {@code LlamaForwardPass.routeExpertsGrouped}, for this decoder. The loop in
   * {@link #route} issues one dispatch per projection per expert -- twelve at {@code
   * expert_used_count = 4} -- where ggml issues one {@code mul_mat_id}. Measured 2026-11-01: this
   * and the missing kernel seam together took lfm2moe's decode from 8.94 to 23.80 tok/s, our own
   * arm before and after. That paired figure is what this change is worth; the gate ratios first
   * quoted beside it were withdrawn, because the comparator arm they divided carried no artifact
   * SHA-256 and the policy had excluded it. The admitted ollama comparison is {@code
   * decodeThroughputRatio} 0.3974 -- still FAILED_RELATIVE_GATE against a 0.80 floor.
   *
   * <p>The two groups are decided separately, which is not symmetry for its own sake: mellum lost
   * half its layers to a coupled check because one of its three expert families is Q5_0, the single
   * quantisation the shim groups neither way. lfm2moe's are all Q4_0, so both groups are expected
   * to take — but the check is written so a future file with a mixed set keeps whichever half it
   * can.
   *
   * <p>Returns false when the gate/up group is declined, leaving the loop to run.
   */
  private boolean routeGrouped(Lfm2Weights.LayerWeights layerWeights, int dim, int expertWidth) {
    if (groupedGateUp == null) {
      return false;
    }
    int used = selectedExperts.length;
    for (int slot = 0; slot < used; slot++) {
      int expert = selectedExperts[slot];
      Lfm2Weights.Matrix gateMatrix = layerWeights.expertGate()[expert];
      Lfm2Weights.Matrix upMatrix = layerWeights.expertUp()[expert];
      Lfm2Weights.Matrix downMatrix = layerWeights.expertDown()[expert];
      groupedGateUpWeights[2 * slot] = gateMatrix.data();
      groupedGateUpWeights[2 * slot + 1] = upMatrix.data();
      groupedGateUpTypes[2 * slot] = gateMatrix.type();
      groupedGateUpTypes[2 * slot + 1] = upMatrix.type();
      groupedGateUpRows[2 * slot] = gateMatrix.rows();
      groupedGateUpRows[2 * slot + 1] = upMatrix.rows();
      groupedDownWeights[slot] = downMatrix.data();
      groupedDownTypes[slot] = downMatrix.type();
      groupedDownRows[slot] = downMatrix.rows();
    }
    if (!matrixKernel.isGroupedEligible(groupedGateUpTypes, groupedGateUpRows, 2 * used, 1, dim)) {
      return false;
    }
    boolean groupDown =
        matrixKernel.isIndependentEligible(groupedDownTypes, groupedDownRows, used, 1, expertWidth);
    matrixKernel.multiplyGrouped(
        groupedGateUp,
        groupedGateUpWeights,
        groupedGateUpTypes,
        groupedGateUpRows,
        2 * used,
        normalized,
        1,
        dim);
    for (int slot = 0; slot < used; slot++) {
      activate(
          groupedActivations[slot],
          0,
          groupedGateUp[2 * slot],
          0,
          groupedGateUp[2 * slot + 1],
          0,
          expertWidth);
    }
    if (groupDown) {
      matrixKernel.multiplyIndependent(
          groupedDownOutputs,
          groupedDownWeights,
          groupedDownTypes,
          groupedDownRows,
          used,
          groupedActivations,
          1,
          expertWidth);
    } else {
      for (int slot = 0; slot < used; slot++) {
        project(
            groupedDownWeights[slot],
            groupedDownTypes[slot],
            dim,
            expertWidth,
            groupedActivations[slot],
            groupedDownOutputs[slot]);
      }
    }
    for (int slot = 0; slot < used; slot++) {
      float weight = routingWeights[slot];
      float[] contribution = groupedDownOutputs[slot];
      for (int index = 0; index < dim; index++) {
        projected[index] += weight * contribution[index];
      }
    }
    return true;
  }

  private void convolve(Lfm2Weights.LayerWeights layerWeights, int layer) {
    project(layerWeights.shortConvInProjection(), normalized, convProjected);
    Lfm2ShortConv.applyToken(
        convProjected,
        config.embeddingDim(),
        layerWeights.shortConvKernel(),
        config.shortConvCache(),
        convState[layer],
        convGated);
    project(layerWeights.shortConvOutProjection(), convGated, projected);
  }

  private static void add(float[] target, float[] addend, int size) {
    for (int index = 0; index < size; index++) {
      target[index] += addend[index];
    }
  }

  private void project(Lfm2Weights.Matrix matrix, float[] input, float[] output) {
    project(matrix.data(), matrix.type(), matrix.rows(), matrix.columns(), input, output);
  }

  private void project(
      MemorySegment matrix,
      GgufTensorType type,
      int rows,
      int columns,
      float[] input,
      float[] output) {
    // The injected kernel first, exactly as LlamaForwardPass.matmulDispatch does. Before
    // 2026-11-01 this method went straight to the Java path and the kernel was never consulted,
    // so an lfm2 model ran unaccelerated while reporting that a kernel was loaded.
    if (matrixKernel.isEligible(type, 1, rows, columns)) {
      matrixKernel.multiply(output, input, matrix, type, 1, rows, columns);
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
}
