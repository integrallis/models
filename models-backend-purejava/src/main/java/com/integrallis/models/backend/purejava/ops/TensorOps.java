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

  enum GroupedProjectionPlan {
    NONE,
    ALL,
    FIRST_SECOND,
    FIRST_THIRD,
    SECOND_THIRD
  }

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
    if (type == GgufTensorType.Q4_0
        || type == GgufTensorType.Q5_0
        || type == GgufTensorType.Q8_0
        || type == GgufTensorType.Q4_K
        || type == GgufTensorType.Q5_K
        || type == GgufTensorType.Q6_K) {
      int activationBlockSize =
          type == GgufTensorType.Q4_K || type == GgufTensorType.Q5_K || type == GgufTensorType.Q6_K
              ? 256
              : 32;
      ggufMatmul(
          out,
          x,
          qWeight,
          type,
          rows,
          cols,
          new byte[cols],
          new float[cols / activationBlockSize],
          new short[(cols + 15) / 16]);
      return;
    }
    ggufMatmul(out, x, qWeight, type, rows, cols, null, null, null);
  }

  /**
   * Matrix-vector multiplication with reusable Q8 activation scratch for GGML quantized kernels.
   */
  public static void ggufMatmul(
      float[] out,
      float[] x,
      MemorySegment qWeight,
      GgufTensorType type,
      int rows,
      int cols,
      byte[] quantizedActivation,
      float[] quantizedActivationScales) {
    ggufMatmul(
        out,
        x,
        qWeight,
        type,
        rows,
        cols,
        quantizedActivation,
        quantizedActivationScales,
        type == GgufTensorType.Q4_K || type == GgufTensorType.Q5_K
            ? new short[(cols + 15) / 16]
            : null);
  }

  /** Matrix-vector multiplication with reusable Q8 activation and Q8_K sum scratch. */
  public static void ggufMatmul(
      float[] out,
      float[] x,
      MemorySegment qWeight,
      GgufTensorType type,
      int rows,
      int cols,
      byte[] quantizedActivation,
      float[] quantizedActivationScales,
      short[] quantizedActivationSums) {
    switch (type) {
      case F32 -> VectorUtil.ggufF32BatchDotProduct(x, qWeight, rows, cols, out);
      case Q4_0 ->
          VectorUtil.ggufQ4_0Q8_0BatchDotProduct(
              x, qWeight, rows, cols, out, quantizedActivation, quantizedActivationScales);
      case Q5_0 ->
          VectorUtil.ggufQ5_0Q8_0BatchDotProduct(
              x, qWeight, rows, cols, out, quantizedActivation, quantizedActivationScales);
      case Q8_0 ->
          VectorUtil.ggufQ8_0Q8_0BatchDotProduct(
              x, qWeight, rows, cols, out, quantizedActivation, quantizedActivationScales);
      case Q4_K ->
          VectorUtil.ggufQ4_KQ8_KBatchDotProduct(
              x,
              qWeight,
              rows,
              cols,
              out,
              quantizedActivation,
              quantizedActivationScales,
              quantizedActivationSums);
      case Q5_K ->
          VectorUtil.ggufQ5_KQ8_KBatchDotProduct(
              x,
              qWeight,
              rows,
              cols,
              out,
              quantizedActivation,
              quantizedActivationScales,
              quantizedActivationSums);
      case Q6_K ->
          VectorUtil.ggufQ6_KQ8_KBatchDotProduct(
              x, qWeight, rows, cols, out, quantizedActivation, quantizedActivationScales);
      default -> throw new UnsupportedOperationException("GGUF matmul not supported for: " + type);
    }
  }

  /**
   * Multiplies two matrices by one activation, sharing quantization and row dispatch when the
   * tensor formats support it.
   */
  public static void ggufDualMatmul(
      float[] firstOut,
      MemorySegment firstWeight,
      GgufTensorType firstType,
      int firstRows,
      float[] secondOut,
      MemorySegment secondWeight,
      GgufTensorType secondType,
      int secondRows,
      float[] input,
      int cols,
      byte[] quantizedActivation,
      float[] quantizedActivationScales,
      short[] quantizedActivationSums) {
    if (firstType == secondType && supportsGroupedMatmul(firstType)) {
      switch (firstType) {
        case Q4_0 ->
            VectorUtil.ggufQ4_0Q8_0DualBatchDotProduct(
                input,
                firstWeight,
                firstRows,
                firstOut,
                secondWeight,
                secondRows,
                secondOut,
                cols,
                quantizedActivation,
                quantizedActivationScales);
        case Q8_0 ->
            VectorUtil.ggufQ8_0Q8_0DualBatchDotProduct(
                input,
                firstWeight,
                firstRows,
                firstOut,
                secondWeight,
                secondRows,
                secondOut,
                cols,
                quantizedActivation,
                quantizedActivationScales);
        case Q4_K ->
            VectorUtil.ggufQ4_KQ8_KDualBatchDotProduct(
                input,
                firstWeight,
                firstRows,
                firstOut,
                secondWeight,
                secondRows,
                secondOut,
                cols,
                quantizedActivation,
                quantizedActivationScales,
                quantizedActivationSums);
        case Q5_K ->
            VectorUtil.ggufQ5_KQ8_KDualBatchDotProduct(
                input,
                firstWeight,
                firstRows,
                firstOut,
                secondWeight,
                secondRows,
                secondOut,
                cols,
                quantizedActivation,
                quantizedActivationScales,
                quantizedActivationSums);
        default -> throw new IllegalStateException("Unsupported grouped matmul type: " + firstType);
      }
      return;
    }

    ggufMatmul(
        firstOut,
        input,
        firstWeight,
        firstType,
        firstRows,
        cols,
        quantizedActivation,
        quantizedActivationScales,
        quantizedActivationSums);
    ggufMatmul(
        secondOut,
        input,
        secondWeight,
        secondType,
        secondRows,
        cols,
        quantizedActivation,
        quantizedActivationScales,
        quantizedActivationSums);
  }

  /**
   * Multiplies three matrices by one activation, sharing quantization and row dispatch when the
   * tensor formats support it.
   */
  public static void ggufTripleMatmul(
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
      int cols,
      byte[] quantizedActivation,
      float[] quantizedActivationScales,
      short[] quantizedActivationSums) {
    switch (groupedProjectionPlan(firstType, secondType, thirdType)) {
      case ALL -> {
        switch (firstType) {
          case Q4_0 ->
              VectorUtil.ggufQ4_0Q8_0TripleBatchDotProduct(
                  input,
                  firstWeight,
                  firstRows,
                  firstOut,
                  secondWeight,
                  secondRows,
                  secondOut,
                  thirdWeight,
                  thirdRows,
                  thirdOut,
                  cols,
                  quantizedActivation,
                  quantizedActivationScales);
          case Q4_K ->
              VectorUtil.ggufQ4_KQ8_KTripleBatchDotProduct(
                  input,
                  firstWeight,
                  firstRows,
                  firstOut,
                  secondWeight,
                  secondRows,
                  secondOut,
                  thirdWeight,
                  thirdRows,
                  thirdOut,
                  cols,
                  quantizedActivation,
                  quantizedActivationScales,
                  quantizedActivationSums);
          case Q5_K ->
              VectorUtil.ggufQ5_KQ8_KTripleBatchDotProduct(
                  input,
                  firstWeight,
                  firstRows,
                  firstOut,
                  secondWeight,
                  secondRows,
                  secondOut,
                  thirdWeight,
                  thirdRows,
                  thirdOut,
                  cols,
                  quantizedActivation,
                  quantizedActivationScales,
                  quantizedActivationSums);
          default ->
              throw new IllegalStateException("Unsupported grouped matmul type: " + firstType);
        }
        return;
      }
      case FIRST_SECOND -> {
        ggufDualMatmul(
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
            quantizedActivationSums);
        ggufMatmul(
            thirdOut,
            input,
            thirdWeight,
            thirdType,
            thirdRows,
            cols,
            quantizedActivation,
            quantizedActivationScales,
            quantizedActivationSums);
        return;
      }
      case FIRST_THIRD -> {
        ggufDualMatmul(
            firstOut,
            firstWeight,
            firstType,
            firstRows,
            thirdOut,
            thirdWeight,
            thirdType,
            thirdRows,
            input,
            cols,
            quantizedActivation,
            quantizedActivationScales,
            quantizedActivationSums);
        ggufMatmul(
            secondOut,
            input,
            secondWeight,
            secondType,
            secondRows,
            cols,
            quantizedActivation,
            quantizedActivationScales,
            quantizedActivationSums);
        return;
      }
      case SECOND_THIRD -> {
        ggufDualMatmul(
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
            quantizedActivationSums);
        ggufMatmul(
            firstOut,
            input,
            firstWeight,
            firstType,
            firstRows,
            cols,
            quantizedActivation,
            quantizedActivationScales,
            quantizedActivationSums);
        return;
      }
      case NONE -> {
        // Fall through to independent format-specific projections.
      }
    }

    ggufMatmul(
        firstOut,
        input,
        firstWeight,
        firstType,
        firstRows,
        cols,
        quantizedActivation,
        quantizedActivationScales,
        quantizedActivationSums);
    ggufMatmul(
        secondOut,
        input,
        secondWeight,
        secondType,
        secondRows,
        cols,
        quantizedActivation,
        quantizedActivationScales,
        quantizedActivationSums);
    ggufMatmul(
        thirdOut,
        input,
        thirdWeight,
        thirdType,
        thirdRows,
        cols,
        quantizedActivation,
        quantizedActivationScales,
        quantizedActivationSums);
  }

  /**
   * Returns whether equal-format projections can share activation quantization and row dispatch.
   */
  public static boolean supportsGroupedMatmul(GgufTensorType type) {
    return type == GgufTensorType.Q4_0
        || type == GgufTensorType.Q8_0
        || type == GgufTensorType.Q4_K
        || type == GgufTensorType.Q5_K;
  }

  private static boolean supportsGroupedTripleMatmul(GgufTensorType type) {
    return type == GgufTensorType.Q4_0
        || type == GgufTensorType.Q4_K
        || type == GgufTensorType.Q5_K;
  }

  static GroupedProjectionPlan groupedProjectionPlan(
      GgufTensorType firstType, GgufTensorType secondType, GgufTensorType thirdType) {
    if (supportsGroupedTripleMatmul(firstType)) {
      if (firstType == secondType && firstType == thirdType) {
        return GroupedProjectionPlan.ALL;
      }
      if (firstType == secondType) {
        return GroupedProjectionPlan.FIRST_SECOND;
      }
      if (firstType == thirdType) {
        return GroupedProjectionPlan.FIRST_THIRD;
      }
    }
    if (secondType == thirdType && supportsGroupedTripleMatmul(secondType)) {
      return GroupedProjectionPlan.SECOND_THIRD;
    }
    return GroupedProjectionPlan.NONE;
  }

  /** Returns whether the mapped tensor type has a weight-reusing batched prefill kernel. */
  public static boolean supportsBatchedMatmul(GgufTensorType type) {
    return type == GgufTensorType.Q4_0 || type == GgufTensorType.Q4_K;
  }

  /** Matrix multiplication over batch-major activations using caller-owned quantization scratch. */
  public static void ggufBatchedMatmul(
      float[] out,
      float[] x,
      MemorySegment qWeight,
      GgufTensorType type,
      int batchSize,
      int rows,
      int cols,
      byte[] quantizedActivations,
      float[] quantizedActivationScales,
      short[] quantizedActivationSums,
      float[] q4LaneScratch) {
    if (!supportsBatchedMatmul(type)) {
      throw new UnsupportedOperationException("GGUF batched matmul not supported for: " + type);
    }
    switch (type) {
      case Q4_0 ->
          VectorUtil.ggufQ4_0Q8_0BatchedMatmul(
              x,
              qWeight,
              batchSize,
              rows,
              cols,
              out,
              quantizedActivations,
              quantizedActivationScales,
              q4LaneScratch);
      case Q4_K ->
          VectorUtil.ggufQ4_KQ8_KBatchedMatmul(
              x,
              qWeight,
              batchSize,
              rows,
              cols,
              out,
              quantizedActivations,
              quantizedActivationScales,
              quantizedActivationSums);
      default -> throw new AssertionError("unhandled batched matmul type: " + type);
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
    rope(q, qOffset, k, kOffset, position, headDim, ropeTheta, 1.0f);
  }

  /** Offset-aware rotary embedding with a GGUF frequency scale. */
  public static void rope(
      float[] q,
      int qOffset,
      float[] k,
      int kOffset,
      int position,
      int headDim,
      float ropeTheta,
      float frequencyScale) {
    float scaledPosition = position * frequencyScale;
    for (int i = 0; i < headDim; i += 2) {
      float freq = (float) (1.0 / Math.pow(ropeTheta, (double) i / headDim));
      float angle = scaledPosition * freq;
      float cos = (float) Math.cos(angle);
      float sin = (float) Math.sin(angle);

      rotatePair(q, qOffset + i, cos, sin);
      rotatePair(k, kOffset + i, cos, sin);
    }
  }

  /** In-place rotary position embedding on one sub-vector. */
  public static void rope(float[] vector, int offset, int position, int headDim, float ropeTheta) {
    rope(vector, offset, position, headDim, ropeTheta, 1.0f);
  }

  /** In-place rotary position embedding on one sub-vector with a GGUF frequency scale. */
  public static void rope(
      float[] vector,
      int offset,
      int position,
      int headDim,
      float ropeTheta,
      float frequencyScale) {
    float scaledPosition = position * frequencyScale;
    for (int i = 0; i < headDim; i += 2) {
      float freq = (float) (1.0 / Math.pow(ropeTheta, (double) i / headDim));
      float angle = scaledPosition * freq;
      float cos = (float) Math.cos(angle);
      float sin = (float) Math.sin(angle);

      rotatePair(vector, offset + i, cos, sin);
    }
  }

  /** Applies standard rotary embedding using caller-precomputed pair factors. */
  public static void rope(float[] vector, int offset, float[] cosine, float[] sine) {
    rope(vector, offset, cosine, sine, 0, cosine.length);
  }

  /** Applies standard rotary embedding from an offset in precomputed pair factors. */
  public static void rope(
      float[] vector,
      int vectorOffset,
      float[] cosine,
      float[] sine,
      int factorOffset,
      int pairCount) {
    for (int pair = 0; pair < pairCount; pair++) {
      rotatePair(
          vector, vectorOffset + pair * 2, cosine[factorOffset + pair], sine[factorOffset + pair]);
    }
  }

  /** In-place NeoX rotary embedding whose coordinate pairs are separated by half a head. */
  public static void ropeNeox(
      float[] vector, int offset, int position, int headDim, float ropeTheta) {
    ropeNeox(vector, offset, position, headDim, ropeTheta, 1.0f);
  }

  /** In-place NeoX rotary embedding with a GGUF frequency scale. */
  public static void ropeNeox(
      float[] vector,
      int offset,
      int position,
      int headDim,
      float ropeTheta,
      float frequencyScale) {
    int half = headDim / 2;
    float scaledPosition = position * frequencyScale;
    for (int i = 0; i < half; i++) {
      float freq = (float) (1.0 / Math.pow(ropeTheta, (double) (2 * i) / headDim));
      float angle = scaledPosition * freq;
      float cos = (float) Math.cos(angle);
      float sin = (float) Math.sin(angle);

      rotateSplitPair(vector, offset + i, offset + half + i, cos, sin);
    }
  }

  /** Applies NeoX rotary embedding using caller-precomputed pair factors. */
  public static void ropeNeox(float[] vector, int offset, float[] cosine, float[] sine) {
    ropeNeox(vector, offset, cosine, sine, 0, cosine.length);
  }

  /** Applies NeoX rotary embedding from an offset in precomputed pair factors. */
  public static void ropeNeox(
      float[] vector,
      int vectorOffset,
      float[] cosine,
      float[] sine,
      int factorOffset,
      int pairCount) {
    for (int pair = 0; pair < pairCount; pair++) {
      rotateSplitPair(
          vector,
          vectorOffset + pair,
          vectorOffset + pairCount + pair,
          cosine[factorOffset + pair],
          sine[factorOffset + pair]);
    }
  }

  /** SwiGLU activation: out[i] = silu(gate[i]) * up[i]. */
  public static void swiGlu(float[] out, float[] gate, float[] up, int size) {
    swiGlu(out, 0, gate, 0, up, 0, size);
  }

  /** Offset-aware SwiGLU activation over flat batch buffers. */
  public static void swiGlu(
      float[] out,
      int outOffset,
      float[] gate,
      int gateOffset,
      float[] up,
      int upOffset,
      int size) {
    for (int i = 0; i < size; i++) {
      float x = gate[gateOffset + i];
      float silu = x / (1.0f + (float) Math.exp(-x));
      out[outOffset + i] = silu * up[upOffset + i];
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
