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

import com.integrallis.models.backend.purejava.cache.KvCache;
import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import com.integrallis.vectors.core.VectorUtil;
import java.lang.foreign.MemorySegment;
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

  private static final String PREFILL_BATCH_SIZE_PROPERTY = "models.purejava.prefillBatchSize";
  private static final int DEFAULT_PREFILL_BATCH_SIZE = 32;

  private final LlamaConfig config;
  private final LlamaWeights weights;
  private final KvCache cache;
  private final RopeTable ropeTable;
  private final LayerObserver layerObserver;
  private final boolean batchedPrefill;
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
  private final short[] quantizedActivationSums;
  private final float[] batchX;
  private final float[] batchXNorm;
  private final float[] batchQ;
  private final float[] batchK;
  private final float[] batchV;
  private final float[] batchAttnOut;
  private final float[] batchAttnProjected;
  private final float[] batchFfnGate;
  private final float[] batchFfnUp;
  private final float[] batchFfnOut;
  private final float[] batchFfnProjected;
  private final byte[] batchQuantizedActivation;
  private final float[] batchQuantizedActivationScales;
  private final float[] batchQ4LaneScratch;
  private int nextPosition;

  public LlamaForwardPass(LlamaConfig config, LlamaWeights weights, KvCache cache) {
    this(config, weights, cache, null);
  }

  LlamaForwardPass(
      LlamaConfig config, LlamaWeights weights, KvCache cache, LayerObserver layerObserver) {
    this.config = config;
    this.weights = weights;
    this.cache = cache;
    this.layerObserver = layerObserver;
    this.ropeTable =
        new RopeTable(config.keyLength(), config.ropeTheta(), config.ropeFrequencyScale());

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
    this.quantizedActivationSums = new short[(maxProjectionInput + 15) / 16];

    int requestedBatchSize = configuredPrefillBatchSize();
    boolean compatible = supportsBatchedPrefill(config, weights);
    int capacity = compatible ? Math.min(requestedBatchSize, cache.maxSeqLen()) : 0;
    this.batchedPrefill = capacity > 1;
    this.prefillBatchCapacity = batchedPrefill ? capacity : 0;
    this.batchX = batchBuffer(prefillBatchCapacity, dim);
    this.batchXNorm = batchBuffer(prefillBatchCapacity, dim);
    this.batchQ = batchBuffer(prefillBatchCapacity, config.queryDim());
    this.batchK = batchBuffer(prefillBatchCapacity, config.keyDim());
    this.batchV = batchBuffer(prefillBatchCapacity, config.valueDim());
    this.batchAttnOut = batchBuffer(prefillBatchCapacity, config.attentionOutputDim());
    this.batchAttnProjected = batchBuffer(prefillBatchCapacity, dim);
    this.batchFfnGate = batchBuffer(prefillBatchCapacity, hiddenDim);
    this.batchFfnUp = batchBuffer(prefillBatchCapacity, hiddenDim);
    this.batchFfnOut = batchBuffer(prefillBatchCapacity, hiddenDim);
    this.batchFfnProjected = batchBuffer(prefillBatchCapacity, dim);
    this.batchQuantizedActivation =
        new byte[Math.multiplyExact(prefillBatchCapacity, maxProjectionInput)];
    this.batchQuantizedActivationScales =
        new float[Math.multiplyExact(prefillBatchCapacity, (maxProjectionInput + 31) / 32)];
    int maxProjectionOutput =
        Math.max(
            Math.max(dim, hiddenDim),
            Math.max(config.queryDim(), Math.max(config.keyDim(), config.valueDim())));
    this.batchQ4LaneScratch =
        new float
            [Math.multiplyExact(Math.multiplyExact(prefillBatchCapacity, maxProjectionOutput), 8)];
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
    return forwardInternal(token, position, true);
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
      forwardInternal(tokens[index], Math.addExact(startPosition, index), false);
    }
    return forwardInternal(tokens[finalIndex], Math.addExact(startPosition, finalIndex), true);
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
        return forwardInternal(tokens[tokenOffset], startPosition + tokenOffset, true);
      }
      prefillBatch(tokens, tokenOffset, batchSize, startPosition + tokenOffset, finalBatch);
      nextPosition += batchSize;
      tokenOffset += batchSize;
    }
    return logits;
  }

  private void prefillBatch(
      int[] tokens, int tokenOffset, int batchSize, int startPosition, boolean computeLogits) {
    int dim = config.embeddingDim();
    int queryDim = config.queryDim();
    int keyDim = config.keyDim();
    int valueDim = config.valueDim();
    int attentionOutputDim = config.attentionOutputDim();
    int hiddenDim = config.hiddenDim();
    int keyLength = config.keyLength();
    int numHeads = config.numHeads();
    int numKvHeads = config.numKvHeads();
    ropeTable.prepareBatch(startPosition, batchSize);

    for (int batch = 0; batch < batchSize; batch++) {
      weights.embedToken(tokens[tokenOffset + batch], x);
      System.arraycopy(x, 0, batchX, batch * dim, dim);
    }

    for (int layer = 0; layer < config.numLayers(); layer++) {
      LlamaWeights.LayerWeights lw = weights.layer(layer);
      for (int batch = 0; batch < batchSize; batch++) {
        int xOffset = batch * dim;
        TensorOps.rmsNorm(
            batchXNorm, xOffset, batchX, xOffset, lw.attentionNorm(), dim, config.rmsNormEps());
      }

      batchedMatmulDispatch(batchQ, batchXNorm, batchSize, lw.wq(), lw.wqType(), queryDim, dim);
      batchedMatmulDispatch(batchK, batchXNorm, batchSize, lw.wk(), lw.wkType(), keyDim, dim);
      batchedMatmulDispatch(batchV, batchXNorm, batchSize, lw.wv(), lw.wvType(), valueDim, dim);

      for (int batch = 0; batch < batchSize; batch++) {
        int qBase = batch * queryDim;
        int kBase = batch * keyDim;
        int vBase = batch * valueDim;
        addOptionalBias(batchQ, qBase, lw.qBias());
        addOptionalBias(batchK, kBase, lw.kBias());
        addOptionalBias(batchV, vBase, lw.vBias());

        for (int head = 0; head < numHeads; head++) {
          int offset = qBase + head * keyLength;
          normalizeHead(batchQ, offset, lw.qNorm(), keyLength);
          if (config.usesRope(layer)) {
            ropeTable.applyBatch(batchQ, offset, batch, config.usesNeoxRope());
          }
        }
        for (int head = 0; head < numKvHeads; head++) {
          int offset = kBase + head * keyLength;
          normalizeHead(batchK, offset, lw.kNorm(), keyLength);
          if (config.usesRope(layer)) {
            ropeTable.applyBatch(batchK, offset, batch, config.usesNeoxRope());
          }
        }
        cache.store(layer, startPosition + batch, batchK, kBase, batchV, vBase);
      }

      float[] keyCache = cache.keyBuffer();
      float[] valueCache = cache.valueBuffer();
      for (int batch = 0; batch < batchSize; batch++) {
        groupedQueryAttention(
            batchQ,
            batch * queryDim,
            batchAttnOut,
            batch * attentionOutputDim,
            layer,
            startPosition + batch,
            keyCache,
            valueCache);
      }

      batchedMatmulDispatch(
          batchAttnProjected,
          batchAttnOut,
          batchSize,
          lw.wo(),
          lw.woType(),
          dim,
          attentionOutputDim);
      addActiveInPlace(batchX, batchAttnProjected, batchSize * dim);

      for (int batch = 0; batch < batchSize; batch++) {
        int xOffset = batch * dim;
        TensorOps.rmsNorm(
            batchXNorm, xOffset, batchX, xOffset, lw.ffnNorm(), dim, config.rmsNormEps());
      }
      batchedMatmulDispatch(
          batchFfnGate, batchXNorm, batchSize, lw.ffnGate(), lw.ffnGateType(), hiddenDim, dim);
      batchedMatmulDispatch(
          batchFfnUp, batchXNorm, batchSize, lw.ffnUp(), lw.ffnUpType(), hiddenDim, dim);
      for (int batch = 0; batch < batchSize; batch++) {
        int hiddenOffset = batch * hiddenDim;
        TensorOps.swiGlu(
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
      addActiveInPlace(batchX, batchFfnProjected, batchSize * dim);
      if (layerObserver != null) {
        for (int batch = 0; batch < batchSize; batch++) {
          layerObserver.onLayerComplete(layer, startPosition + batch, batchX, batch * dim, dim);
        }
      }
    }

    if (computeLogits) {
      int finalOffset = (batchSize - 1) * dim;
      TensorOps.rmsNorm(
          xNorm, 0, batchX, finalOffset, weights.outputNormWeight(), dim, config.rmsNormEps());
      matmulDispatch(
          logits, xNorm, weights.outputSegment(), weights.outputType(), config.vocabSize(), dim);
    }
  }

  private float[] forwardInternal(int token, int position, boolean computeLogits) {
    if (position != nextPosition) {
      throw new IllegalArgumentException(
          "position must be sequential: expected " + nextPosition + ", got " + position);
    }
    if (position >= cache.maxSeqLen()) {
      throw new IllegalArgumentException("position out of range: " + position);
    }

    int dim = config.embeddingDim();
    int keyLength = config.keyLength();
    int valueLength = config.valueLength();
    int numHeads = config.numHeads();
    int numKvHeads = config.numKvHeads();
    ropeTable.prepare(position);

    // Token embedding (dequantizes only the single row for this token)
    weights.embedToken(token, x);

    // Process each transformer layer
    for (int layer = 0; layer < config.numLayers(); layer++) {
      LlamaWeights.LayerWeights lw = weights.layer(layer);

      // Attention norm
      TensorOps.rmsNorm(xNorm, x, lw.attentionNorm(), dim, config.rmsNormEps());

      // QKV projections
      matmulDispatch(q, xNorm, lw.wq(), lw.wqType(), config.queryDim(), dim);
      matmulDispatch(k, xNorm, lw.wk(), lw.wkType(), config.keyDim(), dim);
      matmulDispatch(v, xNorm, lw.wv(), lw.wvType(), config.valueDim(), dim);
      addOptionalBias(q, lw.qBias());
      addOptionalBias(k, lw.kBias());
      addOptionalBias(v, lw.vBias());

      // RoPE for each head
      for (int h = 0; h < numHeads; h++) {
        int offset = h * keyLength;
        normalizeHead(q, offset, lw.qNorm(), keyLength);
        if (config.usesRope(layer)) {
          applyRope(q, offset);
        }
      }
      for (int h = 0; h < numKvHeads; h++) {
        int offset = h * keyLength;
        normalizeHead(k, offset, lw.kNorm(), keyLength);
        if (config.usesRope(layer)) {
          applyRope(k, offset);
        }
      }

      // Store K,V in cache
      cache.store(layer, position, k, v);
      float[] keyCache = cache.keyBuffer();
      float[] valueCache = cache.valueBuffer();

      // Grouped-query attention
      java.util.Arrays.fill(attnOut, 0.0f);
      int groupSize = numHeads / numKvHeads;
      float scale = (float) (1.0 / Math.sqrt(keyLength));

      for (int h = 0; h < numHeads; h++) {
        int kvHead = h / groupSize;
        int qOff = h * keyLength;
        int outputOffset = h * valueLength;

        // Compute attention scores over all cached positions
        for (int p = 0; p <= position; p++) {
          int cacheOffset = cache.keyOffset(layer, p) + kvHead * keyLength;
          float dot = VectorUtil.dotProduct(q, qOff, keyCache, cacheOffset, keyLength);
          attentionScores[p] = dot * scale;
        }

        // Softmax over scores
        TensorOps.softmax(attentionScores, 0, position + 1);

        // Weighted sum of values
        for (int p = 0; p <= position; p++) {
          int cacheOffset = cache.valueOffset(layer, p) + kvHead * valueLength;
          float weight = attentionScores[p];
          VectorUtil.addScaledInPlace(
              attnOut, outputOffset, valueCache, cacheOffset, valueLength, weight);
        }
      }

      // Output projection
      matmulDispatch(
          attnProjected, attnOut, lw.wo(), lw.woType(), dim, config.attentionOutputDim());

      // Residual connection
      for (int i = 0; i < dim; i++) {
        x[i] += attnProjected[i];
      }

      // FFN norm
      TensorOps.rmsNorm(xNorm, x, lw.ffnNorm(), dim, config.rmsNormEps());

      // FFN: SwiGLU
      matmulDispatch(ffnGate, xNorm, lw.ffnGate(), lw.ffnGateType(), config.hiddenDim(), dim);
      matmulDispatch(ffnUp, xNorm, lw.ffnUp(), lw.ffnUpType(), config.hiddenDim(), dim);
      TensorOps.swiGlu(ffnOut, ffnGate, ffnUp, config.hiddenDim());

      // Down projection
      matmulDispatch(ffnProjected, ffnOut, lw.ffnDown(), lw.ffnDownType(), dim, config.hiddenDim());

      // Residual connection
      for (int i = 0; i < dim; i++) {
        x[i] += ffnProjected[i];
      }
      if (layerObserver != null) {
        layerObserver.onLayerComplete(layer, position, x, 0, dim);
      }
    }

    if (computeLogits) {
      TensorOps.rmsNorm(xNorm, x, weights.outputNormWeight(), dim, config.rmsNormEps());
      matmulDispatch(
          logits, xNorm, weights.outputSegment(), weights.outputType(), config.vocabSize(), dim);
    }

    nextPosition++;
    return computeLogits ? logits : null;
  }

  /** Clears the autoregressive key-value cache before processing a new sequence. */
  public void reset() {
    cache.clear();
    nextPosition = 0;
  }

  boolean usesBatchedPrefill() {
    return batchedPrefill;
  }

  private void groupedQueryAttention(
      float[] query,
      int queryOffset,
      float[] output,
      int outputOffset,
      int layer,
      int position,
      float[] keyCache,
      float[] valueCache) {
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
      for (int cachedPosition = 0; cachedPosition <= position; cachedPosition++) {
        int cacheOffset = cache.keyOffset(layer, cachedPosition) + kvHead * keyLength;
        float dot = VectorUtil.dotProduct(query, qOffset, keyCache, cacheOffset, keyLength);
        attentionScores[cachedPosition] = dot * scale;
      }

      TensorOps.softmax(attentionScores, 0, position + 1);
      for (int cachedPosition = 0; cachedPosition <= position; cachedPosition++) {
        int cacheOffset = cache.valueOffset(layer, cachedPosition) + kvHead * valueLength;
        VectorUtil.addScaledInPlace(
            output,
            headOutputOffset,
            valueCache,
            cacheOffset,
            valueLength,
            attentionScores[cachedPosition]);
      }
    }
  }

  private void normalizeHead(float[] vector, int offset, float[] weight, int headDim) {
    if (weight.length != 0) {
      TensorOps.rmsNorm(vector, offset, vector, offset, weight, headDim, config.rmsNormEps());
    }
  }

  private void applyRope(float[] vector, int offset) {
    ropeTable.apply(vector, offset, config.usesNeoxRope());
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
    TensorOps.ggufMatmul(
        out,
        input,
        weight,
        type,
        rows,
        cols,
        quantizedActivation,
        quantizedActivationScales,
        quantizedActivationSums);
  }

  private void batchedMatmulDispatch(
      float[] out,
      float[] input,
      int batchSize,
      MemorySegment weight,
      GgufTensorType type,
      int rows,
      int cols) {
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
        batchQ4LaneScratch);
  }

  private static int configuredPrefillBatchSize() {
    int batchSize = Integer.getInteger(PREFILL_BATCH_SIZE_PROPERTY, DEFAULT_PREFILL_BATCH_SIZE);
    if (batchSize < 1) {
      throw new IllegalArgumentException(
          PREFILL_BATCH_SIZE_PROPERTY + " must be >= 1: " + batchSize);
    }
    return batchSize;
  }

  private static boolean supportsBatchedPrefill(LlamaConfig config, LlamaWeights weights) {
    for (int layer = 0; layer < config.numLayers(); layer++) {
      LlamaWeights.LayerWeights layerWeights = weights.layer(layer);
      if (!TensorOps.supportsBatchedMatmul(layerWeights.wqType())
          || !TensorOps.supportsBatchedMatmul(layerWeights.wkType())
          || !TensorOps.supportsBatchedMatmul(layerWeights.wvType())
          || !TensorOps.supportsBatchedMatmul(layerWeights.woType())
          || !TensorOps.supportsBatchedMatmul(layerWeights.ffnGateType())
          || !TensorOps.supportsBatchedMatmul(layerWeights.ffnUpType())
          || !TensorOps.supportsBatchedMatmul(layerWeights.ffnDownType())) {
        return false;
      }
    }
    return true;
  }

  private static float[] batchBuffer(int batchCapacity, int width) {
    return new float[Math.multiplyExact(batchCapacity, width)];
  }
}
