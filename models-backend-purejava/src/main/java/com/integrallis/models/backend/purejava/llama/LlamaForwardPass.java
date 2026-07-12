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

/**
 * Single-token forward pass for Llama-family models. Implements the full transformer decoder
 * pipeline: embed, (RMSNorm → QKV → RoPE → GQA attention → residual → RMSNorm → SwiGLU FFN →
 * residual) per layer, final RMSNorm → output logits.
 */
public final class LlamaForwardPass {

  private final LlamaConfig config;
  private final LlamaWeights weights;
  private final KvCache cache;

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
  private int nextPosition;

  public LlamaForwardPass(LlamaConfig config, LlamaWeights weights, KvCache cache) {
    this.config = config;
    this.weights = weights;
    this.cache = cache;

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
  }

  /** Runs a single forward pass for the given token at the given position. Returns logits. */
  public float[] forward(int token, int position) {
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
        applyRope(q, offset, position, keyLength);
      }
      for (int h = 0; h < numKvHeads; h++) {
        int offset = h * keyLength;
        normalizeHead(k, offset, lw.kNorm(), keyLength);
        applyRope(k, offset, position, keyLength);
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
    }

    // Final RMSNorm
    TensorOps.rmsNorm(xNorm, x, weights.outputNormWeight(), dim, config.rmsNormEps());

    // Output logits
    matmulDispatch(
        logits, xNorm, weights.outputSegment(), weights.outputType(), config.vocabSize(), dim);

    nextPosition++;
    return logits.clone();
  }

  /** Clears the autoregressive key-value cache before processing a new sequence. */
  public void reset() {
    cache.clear();
    nextPosition = 0;
  }

  private void normalizeHead(float[] vector, int offset, float[] weight, int headDim) {
    if (weight.length != 0) {
      TensorOps.rmsNorm(vector, offset, vector, offset, weight, headDim, config.rmsNormEps());
    }
  }

  private void applyRope(float[] vector, int offset, int position, int headDim) {
    if (config.usesNeoxRope()) {
      TensorOps.ropeNeox(
          vector, offset, position, headDim, config.ropeTheta(), config.ropeFrequencyScale());
    } else {
      TensorOps.rope(
          vector, offset, position, headDim, config.ropeTheta(), config.ropeFrequencyScale());
    }
  }

  private static void addOptionalBias(float[] vector, float[] bias) {
    if (bias.length != 0) {
      VectorUtil.addInPlace(vector, bias);
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
}
