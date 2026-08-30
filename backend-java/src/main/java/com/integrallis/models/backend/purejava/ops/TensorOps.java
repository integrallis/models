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
import com.integrallis.vectors.core.BFloat16Matrix;
import com.integrallis.vectors.core.GgufQ4Kernel;
import com.integrallis.vectors.core.GgufQ6BatchedKernel;
import com.integrallis.vectors.core.VectorUtil;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

/** Core tensor operations for transformer inference. */
public final class TensorOps {

  private static final float TANH_TABLE_LIMIT = 10.0f;
  private static final int TANH_TABLE_SIZE = 1 << 16;
  private static final float TANH_TABLE_SCALE = TANH_TABLE_SIZE / (2.0f * TANH_TABLE_LIMIT);
  private static final float[] TANH_TABLE = createTanhTable();

  enum GroupedProjectionPlan {
    NONE,
    ALL,
    MIXED_Q4_K_Q4_K_Q6_K,
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

  /** Layer normalization with learned scale and bias over one contiguous row. */
  public static void layerNorm(
      float[] out,
      int outOffset,
      float[] x,
      int xOffset,
      float[] weight,
      float[] bias,
      int size,
      float eps) {
    Objects.requireNonNull(out, "out");
    Objects.requireNonNull(x, "x");
    Objects.requireNonNull(weight, "weight");
    Objects.requireNonNull(bias, "bias");
    Objects.checkFromIndexSize(outOffset, size, out.length);
    Objects.checkFromIndexSize(xOffset, size, x.length);
    if (weight.length != size || bias.length != size) {
      throw new IllegalArgumentException(
          "layer-norm scale and bias must match row size: "
              + weight.length
              + ", "
              + bias.length
              + " != "
              + size);
    }
    if (size == 0) {
      throw new IllegalArgumentException("layer-norm row must not be empty");
    }
    if (!(eps >= 0.0f) || !Float.isFinite(eps)) {
      throw new IllegalArgumentException("layer-norm epsilon must be finite and >= 0: " + eps);
    }

    float sum = 0.0f;
    for (int index = 0; index < size; index++) {
      sum += x[xOffset + index];
    }
    float mean = sum / size;
    float variance = 0.0f;
    for (int index = 0; index < size; index++) {
      float centered = x[xOffset + index] - mean;
      variance += centered * centered;
    }
    float scale = 1.0f / (float) Math.sqrt(variance / size + eps);
    for (int index = 0; index < size; index++) {
      float centered = x[xOffset + index] - mean;
      out[outOffset + index] = centered * scale * weight[index] + bias[index];
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
          new int[(cols + 3) / 4],
          new short[(cols + 15) / 16],
          GgufQ4Kernel.WIDENED);
      return;
    }
    ggufMatmul(out, x, qWeight, type, rows, cols, null, null, null, null, GgufQ4Kernel.WIDENED);
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
        new int[(cols + 3) / 4],
        type == GgufTensorType.Q4_K || type == GgufTensorType.Q5_K
            ? new short[(cols + 15) / 16]
            : null,
        GgufQ4Kernel.WIDENED);
  }

  /** Matrix-vector multiplication with caller-owned scratch and an explicit Q4 policy. */
  public static void ggufMatmul(
      float[] out,
      float[] x,
      MemorySegment qWeight,
      GgufTensorType type,
      int rows,
      int cols,
      byte[] quantizedActivation,
      float[] quantizedActivationScales,
      int[] quantizedActivationZeroPointCorrections,
      short[] quantizedActivationSums,
      GgufQ4Kernel q4Kernel) {
    Objects.requireNonNull(q4Kernel, "q4Kernel");
    switch (type) {
      case F32 -> VectorUtil.ggufF32BatchDotProduct(x, qWeight, rows, cols, out);
      case BF16 -> BFloat16Matrix.of(qWeight, rows, cols).multiply(x, out);
      case Q4_0 ->
          VectorUtil.ggufQ4_0Q8_0BatchDotProduct(
              x,
              qWeight,
              rows,
              cols,
              out,
              quantizedActivation,
              quantizedActivationScales,
              quantizedActivationZeroPointCorrections,
              q4Kernel);
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

  /** Two projections sharing quantization and row dispatch under an explicit Q4 policy. */
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
      int[] quantizedActivationZeroPointCorrections,
      short[] quantizedActivationSums,
      GgufQ4Kernel q4Kernel) {
    Objects.requireNonNull(q4Kernel, "q4Kernel");
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
                quantizedActivationScales,
                quantizedActivationZeroPointCorrections,
                q4Kernel);
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
        quantizedActivationZeroPointCorrections,
        quantizedActivationSums,
        q4Kernel);
    ggufMatmul(
        secondOut,
        input,
        secondWeight,
        secondType,
        secondRows,
        cols,
        quantizedActivation,
        quantizedActivationScales,
        quantizedActivationZeroPointCorrections,
        quantizedActivationSums,
        q4Kernel);
  }

  /** Three projections with an explicit Q4 arithmetic policy. */
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
      int[] quantizedActivationZeroPointCorrections,
      short[] quantizedActivationSums,
      GgufQ4Kernel q4Kernel) {
    ggufTripleMatmul(
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
        true);
  }

  /** Triple projection with explicit Q4 and heterogeneous K-quant policies. */
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
      int[] quantizedActivationZeroPointCorrections,
      short[] quantizedActivationSums,
      GgufQ4Kernel q4Kernel,
      boolean mixedKProjections) {
    Objects.requireNonNull(q4Kernel, "q4Kernel");
    GroupedProjectionPlan projectionPlan = groupedProjectionPlan(firstType, secondType, thirdType);
    if (!mixedKProjections && projectionPlan == GroupedProjectionPlan.MIXED_Q4_K_Q4_K_Q6_K) {
      projectionPlan = GroupedProjectionPlan.FIRST_SECOND;
    }
    switch (projectionPlan) {
      case MIXED_Q4_K_Q4_K_Q6_K -> {
        VectorUtil.ggufQ4_KQ4_KQ6_KQ8_KTripleBatchDotProduct(
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
        return;
      }
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
                  quantizedActivationScales,
                  quantizedActivationZeroPointCorrections,
                  q4Kernel);
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
            quantizedActivationZeroPointCorrections,
            quantizedActivationSums,
            q4Kernel);
        ggufMatmul(
            thirdOut,
            input,
            thirdWeight,
            thirdType,
            thirdRows,
            cols,
            quantizedActivation,
            quantizedActivationScales,
            quantizedActivationZeroPointCorrections,
            quantizedActivationSums,
            q4Kernel);
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
            quantizedActivationZeroPointCorrections,
            quantizedActivationSums,
            q4Kernel);
        ggufMatmul(
            secondOut,
            input,
            secondWeight,
            secondType,
            secondRows,
            cols,
            quantizedActivation,
            quantizedActivationScales,
            quantizedActivationZeroPointCorrections,
            quantizedActivationSums,
            q4Kernel);
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
            quantizedActivationZeroPointCorrections,
            quantizedActivationSums,
            q4Kernel);
        ggufMatmul(
            firstOut,
            input,
            firstWeight,
            firstType,
            firstRows,
            cols,
            quantizedActivation,
            quantizedActivationScales,
            quantizedActivationZeroPointCorrections,
            quantizedActivationSums,
            q4Kernel);
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
        quantizedActivationZeroPointCorrections,
        quantizedActivationSums,
        q4Kernel);
    ggufMatmul(
        secondOut,
        input,
        secondWeight,
        secondType,
        secondRows,
        cols,
        quantizedActivation,
        quantizedActivationScales,
        quantizedActivationZeroPointCorrections,
        quantizedActivationSums,
        q4Kernel);
    ggufMatmul(
        thirdOut,
        input,
        thirdWeight,
        thirdType,
        thirdRows,
        cols,
        quantizedActivation,
        quantizedActivationScales,
        quantizedActivationZeroPointCorrections,
        quantizedActivationSums,
        q4Kernel);
  }

  /** Two batched projections sharing quantization and row dispatch under an explicit Q4 policy. */
  public static void ggufDualBatchedMatmul(
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
      int cols,
      byte[] quantizedActivations,
      float[] quantizedActivationScales,
      int[] quantizedActivationZeroPointCorrections,
      short[] quantizedActivationSums,
      float[] q4LaneScratch,
      GgufQ4Kernel q4Kernel) {
    ggufDualBatchedMatmul(
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
        quantizedActivations,
        quantizedActivationScales,
        quantizedActivationZeroPointCorrections,
        quantizedActivationSums,
        q4LaneScratch,
        q4Kernel,
        GgufQ6BatchedKernel.ONE_QUERY_BLOCK);
  }

  /** Two batched projections with explicit Q4 and Q6_K kernel policies. */
  public static void ggufDualBatchedMatmul(
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
      int cols,
      byte[] quantizedActivations,
      float[] quantizedActivationScales,
      int[] quantizedActivationZeroPointCorrections,
      short[] quantizedActivationSums,
      float[] q4LaneScratch,
      GgufQ4Kernel q4Kernel,
      GgufQ6BatchedKernel q6BatchedKernel) {
    Objects.requireNonNull(q4Kernel, "q4Kernel");
    Objects.requireNonNull(q6BatchedKernel, "q6BatchedKernel");
    if (firstType == secondType) {
      switch (firstType) {
        case Q4_0 -> {
          VectorUtil.ggufQ4_0Q8_0DualBatchedMatmul(
              input,
              firstWeight,
              firstRows,
              firstOut,
              secondWeight,
              secondRows,
              secondOut,
              batchSize,
              cols,
              quantizedActivations,
              quantizedActivationScales,
              quantizedActivationZeroPointCorrections,
              q4LaneScratch,
              q4Kernel);
          return;
        }
        case Q4_K -> {
          VectorUtil.ggufQ4_KQ8_KDualBatchedMatmul(
              input,
              firstWeight,
              firstRows,
              firstOut,
              secondWeight,
              secondRows,
              secondOut,
              batchSize,
              cols,
              quantizedActivations,
              quantizedActivationScales,
              quantizedActivationSums);
          return;
        }
        case Q8_0 -> {
          VectorUtil.ggufQ8_0Q8_0DualBatchedMatmul(
              input,
              firstWeight,
              firstRows,
              firstOut,
              secondWeight,
              secondRows,
              secondOut,
              batchSize,
              cols,
              quantizedActivations,
              quantizedActivationScales);
          return;
        }
        default -> {
          // Fall through to independent batched projections.
        }
      }
    }

    ggufBatchedMatmul(
        firstOut,
        input,
        firstWeight,
        firstType,
        batchSize,
        firstRows,
        cols,
        quantizedActivations,
        quantizedActivationScales,
        quantizedActivationZeroPointCorrections,
        quantizedActivationSums,
        q4LaneScratch,
        q4Kernel,
        q6BatchedKernel);
    ggufBatchedMatmul(
        secondOut,
        input,
        secondWeight,
        secondType,
        batchSize,
        secondRows,
        cols,
        quantizedActivations,
        quantizedActivationScales,
        quantizedActivationZeroPointCorrections,
        quantizedActivationSums,
        q4LaneScratch,
        q4Kernel,
        q6BatchedKernel);
  }

  /** Three batched projections with explicit Q4 and heterogeneous K-quant policies. */
  public static void ggufTripleBatchedMatmul(
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
      int cols,
      byte[] quantizedActivations,
      float[] quantizedActivationScales,
      int[] quantizedActivationZeroPointCorrections,
      short[] quantizedActivationSums,
      float[] q4LaneScratch,
      GgufQ4Kernel q4Kernel,
      boolean mixedKProjections) {
    ggufTripleBatchedMatmul(
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
        quantizedActivations,
        quantizedActivationScales,
        quantizedActivationZeroPointCorrections,
        quantizedActivationSums,
        q4LaneScratch,
        q4Kernel,
        GgufQ6BatchedKernel.ONE_QUERY_BLOCK,
        mixedKProjections);
  }

  /** Three batched projections with explicit Q4, Q6_K, and heterogeneous K-quant policies. */
  public static void ggufTripleBatchedMatmul(
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
      int cols,
      byte[] quantizedActivations,
      float[] quantizedActivationScales,
      int[] quantizedActivationZeroPointCorrections,
      short[] quantizedActivationSums,
      float[] q4LaneScratch,
      GgufQ4Kernel q4Kernel,
      GgufQ6BatchedKernel q6BatchedKernel,
      boolean mixedKProjections) {
    Objects.requireNonNull(q4Kernel, "q4Kernel");
    Objects.requireNonNull(q6BatchedKernel, "q6BatchedKernel");
    GroupedProjectionPlan projectionPlan = groupedProjectionPlan(firstType, secondType, thirdType);
    if (!mixedKProjections && projectionPlan == GroupedProjectionPlan.MIXED_Q4_K_Q4_K_Q6_K) {
      projectionPlan = GroupedProjectionPlan.FIRST_SECOND;
    }
    if (projectionPlan == GroupedProjectionPlan.MIXED_Q4_K_Q4_K_Q6_K) {
      VectorUtil.ggufQ4_KQ4_KQ6_KQ8_KTripleBatchedMatmul(
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
          batchSize,
          cols,
          quantizedActivations,
          quantizedActivationScales,
          quantizedActivationSums,
          q6BatchedKernel);
      return;
    }

    switch (projectionPlan) {
      case ALL -> {
        if (firstType == GgufTensorType.Q4_0) {
          VectorUtil.ggufQ4_0Q8_0TripleBatchedMatmul(
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
              batchSize,
              cols,
              quantizedActivations,
              quantizedActivationScales,
              quantizedActivationZeroPointCorrections,
              q4LaneScratch,
              q4Kernel);
          return;
        }
        ggufDualBatchedMatmul(
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
            quantizedActivations,
            quantizedActivationScales,
            quantizedActivationZeroPointCorrections,
            quantizedActivationSums,
            q4LaneScratch,
            q4Kernel,
            q6BatchedKernel);
        ggufBatchedMatmul(
            thirdOut,
            input,
            thirdWeight,
            thirdType,
            batchSize,
            thirdRows,
            cols,
            quantizedActivations,
            quantizedActivationScales,
            quantizedActivationZeroPointCorrections,
            quantizedActivationSums,
            q4LaneScratch,
            q4Kernel,
            q6BatchedKernel);
        return;
      }
      case FIRST_SECOND -> {
        ggufDualBatchedMatmul(
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
            quantizedActivations,
            quantizedActivationScales,
            quantizedActivationZeroPointCorrections,
            quantizedActivationSums,
            q4LaneScratch,
            q4Kernel,
            q6BatchedKernel);
        ggufBatchedMatmul(
            thirdOut,
            input,
            thirdWeight,
            thirdType,
            batchSize,
            thirdRows,
            cols,
            quantizedActivations,
            quantizedActivationScales,
            quantizedActivationZeroPointCorrections,
            quantizedActivationSums,
            q4LaneScratch,
            q4Kernel,
            q6BatchedKernel);
        return;
      }
      case FIRST_THIRD -> {
        ggufDualBatchedMatmul(
            firstOut,
            firstWeight,
            firstType,
            firstRows,
            thirdOut,
            thirdWeight,
            thirdType,
            thirdRows,
            input,
            batchSize,
            cols,
            quantizedActivations,
            quantizedActivationScales,
            quantizedActivationZeroPointCorrections,
            quantizedActivationSums,
            q4LaneScratch,
            q4Kernel,
            q6BatchedKernel);
        ggufBatchedMatmul(
            secondOut,
            input,
            secondWeight,
            secondType,
            batchSize,
            secondRows,
            cols,
            quantizedActivations,
            quantizedActivationScales,
            quantizedActivationZeroPointCorrections,
            quantizedActivationSums,
            q4LaneScratch,
            q4Kernel,
            q6BatchedKernel);
        return;
      }
      case SECOND_THIRD -> {
        ggufDualBatchedMatmul(
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
            quantizedActivations,
            quantizedActivationScales,
            quantizedActivationZeroPointCorrections,
            quantizedActivationSums,
            q4LaneScratch,
            q4Kernel,
            q6BatchedKernel);
        ggufBatchedMatmul(
            firstOut,
            input,
            firstWeight,
            firstType,
            batchSize,
            firstRows,
            cols,
            quantizedActivations,
            quantizedActivationScales,
            quantizedActivationZeroPointCorrections,
            quantizedActivationSums,
            q4LaneScratch,
            q4Kernel,
            q6BatchedKernel);
        return;
      }
      case NONE -> {
        // Fall through to independent format-specific projections.
      }
      case MIXED_Q4_K_Q4_K_Q6_K -> throw new AssertionError("mixed K projection not handled");
    }

    ggufBatchedMatmul(
        firstOut,
        input,
        firstWeight,
        firstType,
        batchSize,
        firstRows,
        cols,
        quantizedActivations,
        quantizedActivationScales,
        quantizedActivationZeroPointCorrections,
        quantizedActivationSums,
        q4LaneScratch,
        q4Kernel,
        q6BatchedKernel);
    ggufBatchedMatmul(
        secondOut,
        input,
        secondWeight,
        secondType,
        batchSize,
        secondRows,
        cols,
        quantizedActivations,
        quantizedActivationScales,
        quantizedActivationZeroPointCorrections,
        quantizedActivationSums,
        q4LaneScratch,
        q4Kernel,
        q6BatchedKernel);
    ggufBatchedMatmul(
        thirdOut,
        input,
        thirdWeight,
        thirdType,
        batchSize,
        thirdRows,
        cols,
        quantizedActivations,
        quantizedActivationScales,
        quantizedActivationZeroPointCorrections,
        quantizedActivationSums,
        q4LaneScratch,
        q4Kernel,
        q6BatchedKernel);
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

  /** Returns whether the format has a retained multi-projection batched prefill kernel. */
  public static boolean supportsGroupedBatchedMatmul(GgufTensorType type) {
    return type == GgufTensorType.Q4_0
        || type == GgufTensorType.Q8_0
        || type == GgufTensorType.Q4_K;
  }

  /** Returns whether three equal-format projections can share one row dispatch. */
  public static boolean supportsGroupedTripleMatmul(GgufTensorType type) {
    return type == GgufTensorType.Q4_0
        || type == GgufTensorType.Q4_K
        || type == GgufTensorType.Q5_K;
  }

  /** Returns whether three projection formats can share one activation and row dispatch. */
  public static boolean supportsGroupedTripleMatmul(
      GgufTensorType firstType, GgufTensorType secondType, GgufTensorType thirdType) {
    return (firstType == secondType
            && firstType == thirdType
            && supportsGroupedTripleMatmul(firstType))
        || (firstType == GgufTensorType.Q4_K
            && secondType == GgufTensorType.Q4_K
            && thirdType == GgufTensorType.Q6_K);
  }

  static GroupedProjectionPlan groupedProjectionPlan(
      GgufTensorType firstType, GgufTensorType secondType, GgufTensorType thirdType) {
    if (firstType == GgufTensorType.Q4_K
        && secondType == GgufTensorType.Q4_K
        && thirdType == GgufTensorType.Q6_K) {
      return GroupedProjectionPlan.MIXED_Q4_K_Q4_K_Q6_K;
    }
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
    return type == GgufTensorType.Q4_0
        || type == GgufTensorType.BF16
        || type == GgufTensorType.Q5_0
        || type == GgufTensorType.Q8_0
        || type == GgufTensorType.Q4_K
        || type == GgufTensorType.Q5_K
        || type == GgufTensorType.Q6_K;
  }

  /** Batched matrix multiplication with caller-owned scratch and an explicit Q4 policy. */
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
      int[] quantizedActivationZeroPointCorrections,
      short[] quantizedActivationSums,
      float[] q4LaneScratch,
      GgufQ4Kernel q4Kernel) {
    ggufBatchedMatmul(
        out,
        x,
        qWeight,
        type,
        batchSize,
        rows,
        cols,
        quantizedActivations,
        quantizedActivationScales,
        quantizedActivationZeroPointCorrections,
        quantizedActivationSums,
        q4LaneScratch,
        q4Kernel,
        GgufQ6BatchedKernel.ONE_QUERY_BLOCK);
  }

  /** Batched matrix multiplication with explicit Q4 and Q6_K kernel policies. */
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
      int[] quantizedActivationZeroPointCorrections,
      short[] quantizedActivationSums,
      float[] q4LaneScratch,
      GgufQ4Kernel q4Kernel,
      GgufQ6BatchedKernel q6BatchedKernel) {
    Objects.requireNonNull(q4Kernel, "q4Kernel");
    Objects.requireNonNull(q6BatchedKernel, "q6BatchedKernel");
    if (!supportsBatchedMatmul(type)) {
      throw new UnsupportedOperationException("GGUF batched matmul not supported for: " + type);
    }
    switch (type) {
      case BF16 -> BFloat16Matrix.of(qWeight, rows, cols).multiplyBatch(x, 0, batchSize, out, 0);
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
              quantizedActivationZeroPointCorrections,
              q4LaneScratch,
              q4Kernel);
      case Q5_0 ->
          VectorUtil.ggufQ5_0Q8_0BatchedMatmul(
              x,
              qWeight,
              batchSize,
              rows,
              cols,
              out,
              quantizedActivations,
              quantizedActivationScales);
      case Q8_0 ->
          VectorUtil.ggufQ8_0Q8_0BatchedMatmul(
              x,
              qWeight,
              batchSize,
              rows,
              cols,
              out,
              quantizedActivations,
              quantizedActivationScales);
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
      case Q5_K ->
          VectorUtil.ggufQ5_KQ8_KBatchedMatmul(
              x,
              qWeight,
              batchSize,
              rows,
              cols,
              out,
              quantizedActivations,
              quantizedActivationScales,
              quantizedActivationSums);
      case Q6_K ->
          VectorUtil.ggufQ6_KQ8_KBatchedMatmul(
              x,
              qWeight,
              batchSize,
              rows,
              cols,
              out,
              quantizedActivations,
              quantizedActivationScales,
              q6BatchedKernel);
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
    Objects.requireNonNull(x, "x");
    if (size <= 0) {
      throw new IllegalArgumentException("size must be positive: " + size);
    }
    Objects.checkFromIndexSize(offset, size, x.length);
    float max = Float.NEGATIVE_INFINITY;
    for (int i = 0; i < size; i++) {
      float value = x[offset + i];
      if (Float.isNaN(value)) {
        throw new IllegalArgumentException("softmax input contains NaN at index " + (offset + i));
      }
      if (value == Float.POSITIVE_INFINITY) {
        throw new IllegalArgumentException(
            "softmax input contains positive infinity at index " + (offset + i));
      }
      if (value > max) {
        max = value;
      }
    }
    if (!Float.isFinite(max)) {
      throw new IllegalArgumentException("softmax requires at least one finite input");
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

  /**
   * In-place stable softmax for attention scores with one learned sink logit whose value vector is
   * zero. The returned scores therefore sum to at most one; the remaining probability belongs to
   * the sink and contributes no value to the attention output.
   */
  public static void softmaxWithZeroValueSink(
      float[] scores, int offset, int size, float sinkLogit) {
    Objects.requireNonNull(scores, "scores");
    if (size <= 0) {
      throw new IllegalArgumentException("size must be positive: " + size);
    }
    Objects.checkFromIndexSize(offset, size, scores.length);
    if (!Float.isFinite(sinkLogit)) {
      throw new IllegalArgumentException("sink logit must be finite: " + sinkLogit);
    }

    float maximum = sinkLogit;
    for (int index = 0; index < size; index++) {
      float score = scores[offset + index];
      if (Float.isNaN(score)) {
        throw new IllegalArgumentException(
            "attention score contains NaN at index " + (offset + index));
      }
      if (score == Float.POSITIVE_INFINITY) {
        throw new IllegalArgumentException(
            "attention score contains positive infinity at index " + (offset + index));
      }
      maximum = Math.max(maximum, score);
    }

    float denominator = (float) Math.exp(sinkLogit - maximum);
    for (int index = 0; index < size; index++) {
      float probability = (float) Math.exp(scores[offset + index] - maximum);
      scores[offset + index] = probability;
      denominator += probability;
    }
    float inverseDenominator = 1.0f / denominator;
    for (int index = 0; index < size; index++) {
      scores[offset + index] *= inverseDenominator;
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

  /** GELU-gated activation using llama.cpp's tanh approximation. */
  public static void geluGlu(float[] out, float[] gate, float[] up, int size) {
    geluGlu(out, 0, gate, 0, up, 0, size);
  }

  /** GELU activation using llama.cpp's tanh approximation. */
  public static void gelu(float[] out, int outOffset, float[] input, int inputOffset, int size) {
    Objects.requireNonNull(out, "out");
    Objects.requireNonNull(input, "input");
    Objects.checkFromIndexSize(outOffset, size, out.length);
    Objects.checkFromIndexSize(inputOffset, size, input.length);
    for (int index = 0; index < size; index++) {
      float value = input[inputOffset + index];
      out[outOffset + index] =
          0.5f
              * value
              * (1.0f
                  + tableTanh(0.7978845608028654f * value * (1.0f + 0.044715f * value * value)));
    }
  }

  /** Offset-aware GELU-gated activation over flat batch buffers. */
  public static void geluGlu(
      float[] out,
      int outOffset,
      float[] gate,
      int gateOffset,
      float[] up,
      int upOffset,
      int size) {
    for (int index = 0; index < size; index++) {
      float value = gate[gateOffset + index];
      float gelu =
          0.5f
              * value
              * (1.0f
                  + tableTanh(0.7978845608028654f * value * (1.0f + 0.044715f * value * value)));
      out[outOffset + index] = gelu * up[upOffset + index];
    }
  }

  private static float tableTanh(float value) {
    if (value <= -TANH_TABLE_LIMIT) {
      return -1.0f;
    }
    if (value >= TANH_TABLE_LIMIT) {
      return 1.0f;
    }
    float tablePosition = (value + TANH_TABLE_LIMIT) * TANH_TABLE_SCALE;
    int index = (int) tablePosition;
    float lower = TANH_TABLE[index];
    return lower + (tablePosition - index) * (TANH_TABLE[index + 1] - lower);
  }

  private static float[] createTanhTable() {
    float[] table = new float[TANH_TABLE_SIZE + 1];
    for (int index = 0; index <= TANH_TABLE_SIZE; index++) {
      float value = -TANH_TABLE_LIMIT + index / TANH_TABLE_SCALE;
      table[index] = (float) Math.tanh(value);
    }
    return table;
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
