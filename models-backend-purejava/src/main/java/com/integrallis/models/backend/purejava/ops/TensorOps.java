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
import com.integrallis.models.backend.purejava.quant.Dequantizer;
import com.integrallis.models.backend.purejava.quant.Q4_0Dequantizer;
import com.integrallis.models.backend.purejava.quant.Q8_0Dequantizer;
import com.integrallis.vectors.core.VectorUtil;
import java.lang.foreign.MemorySegment;

/** Core tensor operations for transformer inference. */
public final class TensorOps {

  private static final Q4_0Dequantizer Q4_0 = new Q4_0Dequantizer();
  private static final Q8_0Dequantizer Q8_0 = new Q8_0Dequantizer();

  private TensorOps() {}

  /** RMS normalization: out[i] = x[i] / rms(x) * weight[i]. */
  public static void rmsNorm(float[] out, float[] x, float[] weight, int size, float eps) {
    float sumSq = 0.0f;
    for (int i = 0; i < size; i++) {
      sumSq += x[i] * x[i];
    }
    float rms = (float) Math.sqrt(sumSq / size + eps);
    float scale = 1.0f / rms;
    for (int i = 0; i < size; i++) {
      out[i] = x[i] * scale * weight[i];
    }
  }

  /** Matrix-vector multiplication: out = weight * x where weight is [rows x cols] row-major. */
  public static void matmul(float[] out, float[] x, float[] weight, int rows, int cols) {
    VectorUtil.batchDotProduct(x, weight, rows, cols, out);
  }

  /**
   * Matrix-vector multiplication with quantized weight. Dequantizes one row at a time and computes
   * the dot product.
   */
  public static void quantizedMatmul(
      float[] out, float[] x, MemorySegment qWeight, GgufTensorType type, int rows, int cols) {
    Dequantizer dequant = getDequantizer(type);
    long rowBytes = rowByteSize(type, cols);
    float[] rowBuf = new float[cols];

    for (int row = 0; row < rows; row++) {
      long rowOffset = (long) row * rowBytes;
      dequant.dequantize(qWeight, rowOffset, rowBuf, 0, cols);
      float sum = 0.0f;
      for (int col = 0; col < cols; col++) {
        sum += rowBuf[col] * x[col];
      }
      out[row] = sum;
    }
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
    for (int i = 0; i < headDim; i += 2) {
      float freq = (float) (1.0 / Math.pow(ropeTheta, (double) i / headDim));
      float angle = position * freq;
      float cos = (float) Math.cos(angle);
      float sin = (float) Math.sin(angle);

      float q0 = q[i];
      float q1 = q[i + 1];
      q[i] = q0 * cos - q1 * sin;
      q[i + 1] = q0 * sin + q1 * cos;

      float k0 = k[i];
      float k1 = k[i + 1];
      k[i] = k0 * cos - k1 * sin;
      k[i + 1] = k0 * sin + k1 * cos;
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

  private static Dequantizer getDequantizer(GgufTensorType type) {
    return switch (type) {
      case Q4_0 -> Q4_0;
      case Q8_0 -> Q8_0;
      default ->
          throw new UnsupportedOperationException("Quantized matmul not supported for: " + type);
    };
  }

  private static long rowByteSize(GgufTensorType type, int cols) {
    long blocks = (cols + type.blockSize() - 1) / type.blockSize();
    return blocks * type.typeSize();
  }
}
