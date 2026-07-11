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
package com.integrallis.models.backend.purejava.ops;

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.vectors.core.VectorUtil;
import java.lang.foreign.MemorySegment;

/** Core tensor operations for transformer inference. */
public final class TensorOps {

  private TensorOps() {}

  /** RMS normalization: out[i] = x[i] / rms(x) * weight[i]. */
  public static void rmsNorm(float[] out, float[] x, float[] weight, int size, float eps) {
    rmsNorm(out, 0, x, 0, weight, size, eps);
  }

  /** Offset-aware RMS normalization for attention heads stored in contiguous buffers. */
  public static void rmsNorm(
      float[] out, int outOffset, float[] x, int xOffset, float[] weight, int size, float eps) {
    float sumSq = VectorUtil.dotProduct(x, xOffset, x, xOffset, size);
    float rms = (float) Math.sqrt(sumSq / size + eps);
    float scale = 1.0f / rms;
    for (int i = 0; i < size; i++) {
      out[outOffset + i] = x[xOffset + i] * scale * weight[i];
    }
  }

  /** Matrix-vector multiplication: out = weight * x where weight is [rows x cols] row-major. */
  public static void matmul(float[] out, float[] x, float[] weight, int rows, int cols) {
    VectorUtil.batchDotProduct(x, weight, rows, cols, out);
  }

  /**
   * Matrix-vector multiplication over a mapped GGUF tensor. Uses vectors-core fused kernels so rows
   * are not materialized as temporary F32 buffers.
   */
  public static void ggufMatmul(
      float[] out, float[] x, MemorySegment qWeight, GgufTensorType type, int rows, int cols) {
    switch (type) {
      case F32 -> VectorUtil.ggufF32BatchDotProduct(x, qWeight, rows, cols, out);
      case Q4_0 -> VectorUtil.ggufQ4_0BatchDotProduct(x, qWeight, rows, cols, out);
      case Q8_0 -> VectorUtil.ggufQ8_0BatchDotProduct(x, qWeight, rows, cols, out);
      case Q6_K -> VectorUtil.ggufQ6_KBatchDotProduct(x, qWeight, rows, cols, out);
      default -> throw new UnsupportedOperationException("GGUF matmul not supported for: " + type);
    }
  }

  /** Matrix-vector multiplication with a quantized GGUF weight. */
  public static void quantizedMatmul(
      float[] out, float[] x, MemorySegment qWeight, GgufTensorType type, int rows, int cols) {
    ggufMatmul(out, x, qWeight, type, rows, cols);
  }

  /** In-place numerically stable softmax over x[offset..offset+size). */
  public static void softmax(float[] x, int offset, int size) {
    float max = Float.NEGATIVE_INFINITY;
    for (int i = 0; i < size; i++) {
      if (x[offset + i] > max) {
        max = x[offset + i];
      }
    }
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
      x[offset + i] = (float) Math.exp(x[offset + i] - max);
      sum += x[offset + i];
    }
    float invSum = 1.0f / sum;
    for (int i = 0; i < size; i++) {
      x[offset + i] *= invSum;
    }
  }

  /** In-place rotary position embedding on q and k vectors. */
  public static void rope(float[] q, float[] k, int position, int headDim, float ropeTheta) {
    rope(q, 0, k, 0, position, headDim, ropeTheta);
  }

  /** In-place rotary position embedding on q and k sub-vectors. */
  public static void rope(
      float[] q, int qOffset, float[] k, int kOffset, int position, int headDim, float ropeTheta) {
    for (int i = 0; i < headDim; i += 2) {
      float freq = (float) (1.0 / Math.pow(ropeTheta, (double) i / headDim));
      float angle = position * freq;
      float cos = (float) Math.cos(angle);
      float sin = (float) Math.sin(angle);

      rotatePair(q, qOffset + i, cos, sin);
      rotatePair(k, kOffset + i, cos, sin);
    }
  }

  /** In-place rotary position embedding on one sub-vector. */
  public static void rope(float[] vector, int offset, int position, int headDim, float ropeTheta) {
    for (int i = 0; i < headDim; i += 2) {
      float freq = (float) (1.0 / Math.pow(ropeTheta, (double) i / headDim));
      float angle = position * freq;
      float cos = (float) Math.cos(angle);
      float sin = (float) Math.sin(angle);

      rotatePair(vector, offset + i, cos, sin);
    }
  }

  /** In-place NeoX rotary embedding whose coordinate pairs are separated by half a head. */
  public static void ropeNeox(
      float[] vector, int offset, int position, int headDim, float ropeTheta) {
    int half = headDim / 2;
    for (int i = 0; i < half; i++) {
      float freq = (float) (1.0 / Math.pow(ropeTheta, (double) (2 * i) / headDim));
      float angle = position * freq;
      float cos = (float) Math.cos(angle);
      float sin = (float) Math.sin(angle);

      rotateSplitPair(vector, offset + i, offset + half + i, cos, sin);
    }
  }

  /** SwiGLU activation: out[i] = silu(gate[i]) * up[i]. */
  public static void swiGlu(float[] out, float[] gate, float[] up, int size) {
    for (int i = 0; i < size; i++) {
      float x = gate[i];
      float silu = x / (1.0f + (float) Math.exp(-x));
      out[i] = silu * up[i];
    }
  }

  private static void rotatePair(float[] vector, int offset, float cos, float sin) {
    float x0 = vector[offset];
    float x1 = vector[offset + 1];
    vector[offset] = x0 * cos - x1 * sin;
    vector[offset + 1] = x0 * sin + x1 * cos;
  }

  private static void rotateSplitPair(float[] vector, int first, int second, float cos, float sin) {
    float x0 = vector[first];
    float x1 = vector[second];
    vector[first] = x0 * cos - x1 * sin;
    vector[second] = x0 * sin + x1 * cos;
  }
}
