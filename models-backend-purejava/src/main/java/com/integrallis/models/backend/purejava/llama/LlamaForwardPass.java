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
  private final float[] ffnGate;
  private final float[] ffnUp;
  private final float[] ffnOut;
  private final float[] logits;

  public LlamaForwardPass(LlamaConfig config, LlamaWeights weights, KvCache cache) {
    this.config = config;
    this.weights = weights;
    this.cache = cache;

    int dim = config.embeddingDim();
    int kvDim = config.kvDim();
    int hiddenDim = config.hiddenDim();
    int vocabSize = config.vocabSize();

    this.x = new float[dim];
    this.xNorm = new float[dim];
    this.q = new float[dim];
    this.k = new float[kvDim];
    this.v = new float[kvDim];
    this.attnOut = new float[dim];
    this.ffnGate = new float[hiddenDim];
    this.ffnUp = new float[hiddenDim];
    this.ffnOut = new float[hiddenDim];
    this.logits = new float[vocabSize];
  }

  /** Runs a single forward pass for the given token at the given position. Returns logits. */
  public float[] forward(int token, int position) {
    int dim = config.embeddingDim();
    int headDim = config.headDim();
    int numHeads = config.numHeads();
    int numKvHeads = config.numKvHeads();
    int kvDim = config.kvDim();

    // Token embedding (dequantizes only the single row for this token)
    weights.embedToken(token, x);

    // Process each transformer layer
    for (int layer = 0; layer < config.numLayers(); layer++) {
      LlamaWeights.LayerWeights lw = weights.layer(layer);

      // Attention norm
      TensorOps.rmsNorm(xNorm, x, lw.attentionNorm(), dim, config.rmsNormEps());

      // QKV projections
      matmulDispatch(q, xNorm, lw.wq(), lw.wqType(), dim, dim);
      matmulDispatch(k, xNorm, lw.wk(), lw.wkType(), kvDim, dim);
      matmulDispatch(v, xNorm, lw.wv(), lw.wvType(), kvDim, dim);

      // RoPE for each head
      for (int h = 0; h < numHeads; h++) {
        int qOff = h * headDim;
        int kOff = (h % numKvHeads) * headDim;
        float[] qHead = new float[headDim];
        float[] kHead = new float[headDim];
        System.arraycopy(q, qOff, qHead, 0, headDim);
        if (h < numKvHeads) {
          System.arraycopy(k, kOff, kHead, 0, headDim);
        }
        TensorOps.rope(qHead, kHead, position, headDim, config.ropeTheta());
        System.arraycopy(qHead, 0, q, qOff, headDim);
        if (h < numKvHeads) {
          System.arraycopy(kHead, 0, k, kOff, headDim);
        }
      }

      // Store K,V in cache
      cache.store(layer, position, k, v);

      // Grouped-query attention
      java.util.Arrays.fill(attnOut, 0.0f);
      int groupSize = numHeads / numKvHeads;
      float scale = (float) (1.0 / Math.sqrt(headDim));

      for (int h = 0; h < numHeads; h++) {
        int kvHead = h / groupSize;
        int qOff = h * headDim;

        // Compute attention scores over all cached positions
        float[] scores = new float[position + 1];
        for (int p = 0; p <= position; p++) {
          float[] cachedK = cache.key(layer, p);
          float dot = 0;
          for (int d = 0; d < headDim; d++) {
            dot += q[qOff + d] * cachedK[kvHead * headDim + d];
          }
          scores[p] = dot * scale;
        }

        // Softmax over scores
        TensorOps.softmax(scores, 0, position + 1);

        // Weighted sum of values
        for (int p = 0; p <= position; p++) {
          float[] cachedV = cache.value(layer, p);
          float weight = scores[p];
          for (int d = 0; d < headDim; d++) {
            attnOut[qOff + d] += weight * cachedV[kvHead * headDim + d];
          }
        }
      }

      // Output projection
      float[] attnProjected = new float[dim];
      matmulDispatch(attnProjected, attnOut, lw.wo(), lw.woType(), dim, dim);

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
      float[] ffnProjected = new float[dim];
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

    return logits.clone();
  }

  private void matmulDispatch(
      float[] out, float[] input, MemorySegment weight, GgufTensorType type, int rows, int cols) {
    if (type == GgufTensorType.F32) {
      // Read F32 weight into temporary buffer
      float[] w = new float[rows * cols];
      for (int i = 0; i < w.length; i++) {
        w[i] =
            weight.get(
                java.lang.foreign.ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(
                    java.nio.ByteOrder.LITTLE_ENDIAN),
                (long) i * 4);
      }
      TensorOps.matmul(out, input, w, rows, cols);
    } else {
      TensorOps.quantizedMatmul(out, input, weight, type, rows, cols);
    }
  }
}
