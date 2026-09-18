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
package com.integrallis.models.backend.tornado;

import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;
import uk.ac.manchester.tornado.api.types.arrays.IntArray;

/**
 * Java-authored TornadoVM kernels for production-compatible K-quant by Q8_K projections.
 *
 * <p>Q4_K_M catalog models mix two super-block formats in one tensor set: Q4_K for most projections
 * and Q6_K for the value and a share of the down projections. Both consume the same Q8_K activation
 * preparation, so one host-side quantization feeds every kernel here.
 *
 * <p><b>GGUF super-block layouts implemented here</b> (little-endian, 256 values per super-block):
 *
 * <ul>
 *   <li><b>Q4_K</b>, 144 bytes: {@code d} (fp16) at 0, {@code dmin} (fp16) at 2, twelve bytes of
 *       six-bit packed scales and mins at 4, and 128 bytes of 4-bit quants at 16. The 256 values
 *       are eight groups of 32; group {@code g} reads the 32 bytes at {@code 16 + (g >>> 1) * 32}
 *       and takes the low nibble when {@code g} is even and the high nibble when it is odd. Group
 *       values are unsigned 0..15 and the offset is carried by the per-group six-bit minimum.
 *   <li><b>Q6_K</b>, 210 bytes: 128 bytes of low nibbles, 64 bytes of high two-bit pairs, 16 signed
 *       int8 group scales, then {@code d} (fp16) at 208. Quants are {@code (ql | qh << 4) - 32}.
 * </ul>
 *
 * <p><b>Arithmetic contract.</b> Every per-super-block reduction is accumulated in {@code int}, so
 * the quantized dot products and the Q4_K minimum corrections are exact and independent of work
 * scheduling. Only the per-super-block scale application is floating point, and it is applied in
 * ascending super-block order with the same two-step form the vectors-core CPU kernels use ({@code
 * sum + d * quantizedSum}, then {@code sum - dMin * minimumSum}). The CPU kernels fuse those two
 * steps with {@code Math.fma}; this kernel uses a plain multiply and add, matching the operation
 * set the existing Q4_0 kernel is known to compile to PTX with. The two agree bit-for-bit whenever
 * the products are exactly representable and differ by at most one fused-multiply-add rounding per
 * super-block otherwise.
 */
public final class KQuantProjectionKernel {
  /** Values per K-quant super-block. */
  public static final int SUPER_BLOCK_VALUES = 256;

  /** Values per Q8_K partial-sum block; Q4_K's minimum correction reads two of these per group. */
  public static final int SUM_BLOCK_VALUES = 16;

  /** Q8_K partial-sum entries per super-block. */
  public static final int SUMS_PER_SUPER_BLOCK = SUPER_BLOCK_VALUES / SUM_BLOCK_VALUES;

  /** Bytes per Q4_K super-block. */
  public static final int Q4_K_BLOCK_BYTES = 144;

  /** Bytes per Q6_K super-block. */
  public static final int Q6_K_BLOCK_BYTES = 210;

  private static final int Q4_K_SCALES_OFFSET = 4;
  private static final int Q4_K_QUANTS_OFFSET = 16;
  private static final int Q4_K_GROUPS = 8;
  private static final int Q4_K_GROUP_VALUES = 32;

  private static final int Q6_K_QL_BYTES = 128;
  private static final int Q6_K_QH_BYTES = 64;
  private static final int Q6_K_SCALE_BYTES = 16;
  private static final int Q6_K_SCALE_OFFSET = Q6_K_QL_BYTES + Q6_K_QH_BYTES;
  private static final int Q6_K_DELTA_OFFSET = Q6_K_SCALE_OFFSET + Q6_K_SCALE_BYTES;

  private KQuantProjectionKernel() {}

  /** Runs one work item per batch/output-row pair over host-prepared Q8_K activations. */
  public static void multiplyQ4K(
      ByteArray weights,
      ByteArray activations,
      FloatArray activationScales,
      IntArray activationSums,
      FloatArray output,
      int batchSize,
      int rows,
      int cols) {
    int blocks = cols / SUPER_BLOCK_VALUES;
    int sumsPerBatch = cols / SUM_BLOCK_VALUES;
    for (@Parallel int outputIndex = 0; outputIndex < batchSize * rows; outputIndex++) {
      int batch = outputIndex / rows;
      int row = outputIndex - batch * rows;
      output.set(
          outputIndex,
          q4kRowDot(
              weights,
              row,
              activations,
              batch * cols,
              activationScales,
              batch * blocks,
              activationSums,
              batch * sumsPerBatch,
              blocks));
    }
  }

  /** Computes two Q4_K projections in one dispatch over one prepared activation. */
  public static void multiplyQ4KDual(
      ByteArray firstWeights,
      int firstRows,
      ByteArray secondWeights,
      int secondRows,
      ByteArray activations,
      FloatArray activationScales,
      IntArray activationSums,
      FloatArray firstOutput,
      FloatArray secondOutput,
      int batchSize,
      int cols) {
    int blocks = cols / SUPER_BLOCK_VALUES;
    int sumsPerBatch = cols / SUM_BLOCK_VALUES;
    int combinedRows = firstRows + secondRows;
    for (@Parallel int outputIndex = 0; outputIndex < batchSize * combinedRows; outputIndex++) {
      int batch = outputIndex / combinedRows;
      int combinedRow = outputIndex - batch * combinedRows;
      if (combinedRow < firstRows) {
        firstOutput.set(
            batch * firstRows + combinedRow,
            q4kRowDot(
                firstWeights,
                combinedRow,
                activations,
                batch * cols,
                activationScales,
                batch * blocks,
                activationSums,
                batch * sumsPerBatch,
                blocks));
      } else {
        int row = combinedRow - firstRows;
        secondOutput.set(
            batch * secondRows + row,
            q4kRowDot(
                secondWeights,
                row,
                activations,
                batch * cols,
                activationScales,
                batch * blocks,
                activationSums,
                batch * sumsPerBatch,
                blocks));
      }
    }
  }

  /** Computes three Q4_K projections in one dispatch over one prepared activation. */
  public static void multiplyQ4KTriple(
      ByteArray firstWeights,
      int firstRows,
      ByteArray secondWeights,
      int secondRows,
      ByteArray thirdWeights,
      int thirdRows,
      ByteArray activations,
      FloatArray activationScales,
      IntArray activationSums,
      FloatArray firstOutput,
      FloatArray secondOutput,
      FloatArray thirdOutput,
      int batchSize,
      int cols) {
    int blocks = cols / SUPER_BLOCK_VALUES;
    int sumsPerBatch = cols / SUM_BLOCK_VALUES;
    int firstAndSecondRows = firstRows + secondRows;
    int combinedRows = firstAndSecondRows + thirdRows;
    for (@Parallel int outputIndex = 0; outputIndex < batchSize * combinedRows; outputIndex++) {
      int batch = outputIndex / combinedRows;
      int combinedRow = outputIndex - batch * combinedRows;
      if (combinedRow < firstRows) {
        firstOutput.set(
            batch * firstRows + combinedRow,
            q4kRowDot(
                firstWeights,
                combinedRow,
                activations,
                batch * cols,
                activationScales,
                batch * blocks,
                activationSums,
                batch * sumsPerBatch,
                blocks));
      } else if (combinedRow < firstAndSecondRows) {
        int row = combinedRow - firstRows;
        secondOutput.set(
            batch * secondRows + row,
            q4kRowDot(
                secondWeights,
                row,
                activations,
                batch * cols,
                activationScales,
                batch * blocks,
                activationSums,
                batch * sumsPerBatch,
                blocks));
      } else {
        int row = combinedRow - firstAndSecondRows;
        thirdOutput.set(
            batch * thirdRows + row,
            q4kRowDot(
                thirdWeights,
                row,
                activations,
                batch * cols,
                activationScales,
                batch * blocks,
                activationSums,
                batch * sumsPerBatch,
                blocks));
      }
    }
  }

  /** Runs one work item per batch/output-row pair over host-prepared Q8_K activations. */
  public static void multiplyQ6K(
      ByteArray weights,
      ByteArray activations,
      FloatArray activationScales,
      FloatArray output,
      int batchSize,
      int rows,
      int cols) {
    int blocks = cols / SUPER_BLOCK_VALUES;
    for (@Parallel int outputIndex = 0; outputIndex < batchSize * rows; outputIndex++) {
      int batch = outputIndex / rows;
      int row = outputIndex - batch * rows;
      output.set(
          outputIndex,
          q6kRowDot(
              weights, row, activations, batch * cols, activationScales, batch * blocks, blocks));
    }
  }

  /**
   * Computes the Q4_K/Q4_K/Q6_K attention projection group in one dispatch.
   *
   * <p>This is the shape a Q4_K_M model presents for query, key, and value: llama.cpp promotes the
   * value projection to Q6_K while the query and key projections stay Q4_K.
   */
  public static void multiplyMixedTriple(
      ByteArray firstWeights,
      int firstRows,
      ByteArray secondWeights,
      int secondRows,
      ByteArray thirdWeights,
      int thirdRows,
      ByteArray activations,
      FloatArray activationScales,
      IntArray activationSums,
      FloatArray firstOutput,
      FloatArray secondOutput,
      FloatArray thirdOutput,
      int batchSize,
      int cols) {
    int blocks = cols / SUPER_BLOCK_VALUES;
    int sumsPerBatch = cols / SUM_BLOCK_VALUES;
    int firstAndSecondRows = firstRows + secondRows;
    int combinedRows = firstAndSecondRows + thirdRows;
    for (@Parallel int outputIndex = 0; outputIndex < batchSize * combinedRows; outputIndex++) {
      int batch = outputIndex / combinedRows;
      int combinedRow = outputIndex - batch * combinedRows;
      if (combinedRow < firstRows) {
        firstOutput.set(
            batch * firstRows + combinedRow,
            q4kRowDot(
                firstWeights,
                combinedRow,
                activations,
                batch * cols,
                activationScales,
                batch * blocks,
                activationSums,
                batch * sumsPerBatch,
                blocks));
      } else if (combinedRow < firstAndSecondRows) {
        int row = combinedRow - firstRows;
        secondOutput.set(
            batch * secondRows + row,
            q4kRowDot(
                secondWeights,
                row,
                activations,
                batch * cols,
                activationScales,
                batch * blocks,
                activationSums,
                batch * sumsPerBatch,
                blocks));
      } else {
        int row = combinedRow - firstAndSecondRows;
        thirdOutput.set(
            batch * thirdRows + row,
            q6kRowDot(
                thirdWeights,
                row,
                activations,
                batch * cols,
                activationScales,
                batch * blocks,
                blocks));
      }
    }
  }

  private static float q4kRowDot(
      ByteArray weights,
      int row,
      ByteArray activations,
      int activationOffset,
      FloatArray activationScales,
      int scaleOffset,
      IntArray activationSums,
      int sumOffset,
      int blocks) {
    int rowOffset = row * blocks * Q4_K_BLOCK_BYTES;
    float sum = 0.0f;
    for (int block = 0; block < blocks; block++) {
      int blockOffset = rowOffset + block * Q4_K_BLOCK_BYTES;
      float activationScale = activationScales.get(scaleOffset + block);
      float delta = weights.getHalfFloat(blockOffset).getFloat32() * activationScale;
      float deltaMin = weights.getHalfFloat(blockOffset + 2).getFloat32() * activationScale;
      int scalesOffset = blockOffset + Q4_K_SCALES_OFFSET;
      int quantsOffset = blockOffset + Q4_K_QUANTS_OFFSET;
      int blockActivationOffset = activationOffset + block * SUPER_BLOCK_VALUES;
      int blockSumOffset = sumOffset + block * SUMS_PER_SUPER_BLOCK;
      int quantizedSum = 0;
      int minimumSum = 0;
      for (int group = 0; group < Q4_K_GROUPS; group++) {
        int packedOffset = quantsOffset + (group >>> 1) * Q4_K_GROUP_VALUES;
        int shift = (group & 1) * 4;
        int groupActivationOffset = blockActivationOffset + group * Q4_K_GROUP_VALUES;
        int groupDot = 0;
        for (int index = 0; index < Q4_K_GROUP_VALUES; index++) {
          int packed = weights.get(packedOffset + index) & 0xFF;
          int quant = (packed >>> shift) & 0x0F;
          groupDot += quant * activations.get(groupActivationOffset + index);
        }
        quantizedSum += groupScale(weights, scalesOffset, group) * groupDot;
        int groupSumOffset = blockSumOffset + group * 2;
        minimumSum +=
            groupMinimum(weights, scalesOffset, group)
                * (activationSums.get(groupSumOffset) + activationSums.get(groupSumOffset + 1));
      }
      sum = sum + delta * quantizedSum;
      sum = sum - deltaMin * minimumSum;
    }
    return sum;
  }

  private static float q6kRowDot(
      ByteArray weights,
      int row,
      ByteArray activations,
      int activationOffset,
      FloatArray activationScales,
      int scaleOffset,
      int blocks) {
    int rowOffset = row * blocks * Q6_K_BLOCK_BYTES;
    float sum = 0.0f;
    for (int block = 0; block < blocks; block++) {
      int blockOffset = rowOffset + block * Q6_K_BLOCK_BYTES;
      float delta =
          weights.getHalfFloat(blockOffset + Q6_K_DELTA_OFFSET).getFloat32()
              * activationScales.get(scaleOffset + block);
      int blockActivationOffset = activationOffset + block * SUPER_BLOCK_VALUES;
      int blockSum = 0;
      for (int half = 0; half < 2; half++) {
        int lowBase = blockOffset + half * 64;
        int highBase = blockOffset + Q6_K_QL_BYTES + half * 32;
        int scaleBase = blockOffset + Q6_K_SCALE_OFFSET + half * 8;
        int quantBase = blockActivationOffset + half * 128;
        for (int index = 0; index < 32; index++) {
          int scaleIndex = index >>> 4;
          int lowFirst = weights.get(lowBase + index) & 0xFF;
          int lowSecond = weights.get(lowBase + 32 + index) & 0xFF;
          int high = weights.get(highBase + index) & 0xFF;
          int first = ((lowFirst & 0x0F) | ((high & 0x03) << 4)) - 32;
          int second = ((lowSecond & 0x0F) | (((high >>> 2) & 0x03) << 4)) - 32;
          int third = ((lowFirst >>> 4) | (((high >>> 4) & 0x03) << 4)) - 32;
          int fourth = ((lowSecond >>> 4) | (((high >>> 6) & 0x03) << 4)) - 32;
          blockSum +=
              weights.get(scaleBase + scaleIndex) * first * activations.get(quantBase + index)
                  + weights.get(scaleBase + scaleIndex + 2)
                      * second
                      * activations.get(quantBase + index + 32)
                  + weights.get(scaleBase + scaleIndex + 4)
                      * third
                      * activations.get(quantBase + index + 64)
                  + weights.get(scaleBase + scaleIndex + 6)
                      * fourth
                      * activations.get(quantBase + index + 96);
        }
      }
      sum = sum + delta * blockSum;
    }
    return sum;
  }

  /** Decodes one of the eight six-bit group scales packed into a Q4_K super-block header. */
  private static int groupScale(ByteArray weights, int scalesOffset, int group) {
    if (group < 4) {
      return weights.get(scalesOffset + group) & 0x3F;
    }
    int low = weights.get(scalesOffset + group + 4) & 0x0F;
    int high = (weights.get(scalesOffset + group - 4) & 0xFF) >>> 6;
    return low | (high << 4);
  }

  /** Decodes one of the eight six-bit group minima packed into a Q4_K super-block header. */
  private static int groupMinimum(ByteArray weights, int scalesOffset, int group) {
    if (group < 4) {
      return weights.get(scalesOffset + group + 4) & 0x3F;
    }
    int low = (weights.get(scalesOffset + group + 4) & 0xFF) >>> 4;
    int high = (weights.get(scalesOffset + group) & 0xFF) >>> 6;
    return low | (high << 4);
  }

  /**
   * Reproduces the production Q8_K activation preparation.
   *
   * <p>Unlike Q8_0, the Q8_K scale is not rounded through fp16 and the block is scaled by the
   * signed extremum rather than the absolute maximum. The per-16-value sums are what Q4_K's minimum
   * correction consumes; they always fit in a {@code short} on the CPU side, so widening them to
   * {@code int} for the device is value-preserving.
   */
  public static void quantize(
      float[] input, byte[] activations, float[] scales, int[] sums, int batchSize, int cols) {
    int blocks = cols / SUPER_BLOCK_VALUES;
    for (int batch = 0; batch < batchSize; batch++) {
      int rowOffset = batch * cols;
      int rowScaleOffset = batch * blocks;
      int rowSumOffset = batch * (cols / SUM_BLOCK_VALUES);
      for (int block = 0; block < blocks; block++) {
        int offset = rowOffset + block * SUPER_BLOCK_VALUES;
        float extremum = 0.0f;
        float absoluteMax = 0.0f;
        for (int index = 0; index < SUPER_BLOCK_VALUES; index++) {
          float value = input[offset + index];
          float absolute = Math.abs(value);
          if (absolute > absoluteMax) {
            absoluteMax = absolute;
            extremum = value;
          }
        }
        int blockSumOffset = rowSumOffset + block * SUMS_PER_SUPER_BLOCK;
        if (absoluteMax == 0.0f) {
          for (int index = 0; index < SUPER_BLOCK_VALUES; index++) {
            activations[offset + index] = 0;
          }
          for (int index = 0; index < SUMS_PER_SUPER_BLOCK; index++) {
            sums[blockSumOffset + index] = 0;
          }
          scales[rowScaleOffset + block] = 0.0f;
          continue;
        }
        float inverseScale = -127.0f / extremum;
        int partialSum = 0;
        for (int index = 0; index < SUPER_BLOCK_VALUES; index++) {
          int quant = ggmlNearestInt(inverseScale * input[offset + index]);
          byte stored = (byte) Math.min(127, quant);
          activations[offset + index] = stored;
          partialSum += stored;
          if ((index + 1) % SUM_BLOCK_VALUES == 0) {
            sums[blockSumOffset + index / SUM_BLOCK_VALUES] = partialSum;
            partialSum = 0;
          }
        }
        scales[rowScaleOffset + block] = 1.0f / inverseScale;
      }
    }
  }

  /** Validates one Q4_K projection's device storage before a task graph is built. */
  public static void validateQ4K(
      ByteArray weights,
      ByteArray activations,
      FloatArray activationScales,
      IntArray activationSums,
      FloatArray output,
      int batchSize,
      int rows,
      int cols) {
    validateShape(batchSize, rows, cols);
    validateWeights(weights, rows, cols, Q4_K_BLOCK_BYTES);
    validateActivations(activations, activationScales, batchSize, cols);
    if (activationSums == null
        || activationSums.getSize() != Math.multiplyExact(batchSize, cols / SUM_BLOCK_VALUES)) {
      throw new IllegalArgumentException("activation sums do not match the projection shape");
    }
    validateOutput(output, batchSize, rows);
  }

  /** Validates one Q6_K projection's device storage before a task graph is built. */
  public static void validateQ6K(
      ByteArray weights,
      ByteArray activations,
      FloatArray activationScales,
      FloatArray output,
      int batchSize,
      int rows,
      int cols) {
    validateShape(batchSize, rows, cols);
    validateWeights(weights, rows, cols, Q6_K_BLOCK_BYTES);
    validateActivations(activations, activationScales, batchSize, cols);
    validateOutput(output, batchSize, rows);
  }

  private static void validateShape(int batchSize, int rows, int cols) {
    if (batchSize < 1) {
      throw new IllegalArgumentException("batchSize must be positive");
    }
    if (rows < 1) {
      throw new IllegalArgumentException("rows must be positive");
    }
    if (cols < SUPER_BLOCK_VALUES || cols % SUPER_BLOCK_VALUES != 0) {
      throw new IllegalArgumentException("cols must be a positive multiple of 256");
    }
  }

  private static void validateWeights(ByteArray weights, int rows, int cols, int blockBytes) {
    if (weights == null) {
      throw new IllegalArgumentException("weights do not match the projection shape");
    }
    int blocks = cols / SUPER_BLOCK_VALUES;
    long expected = (long) rows * blocks * blockBytes;
    if (expected > Integer.MAX_VALUE || weights.getSize() != (int) expected) {
      throw new IllegalArgumentException("weights do not match the projection shape");
    }
  }

  private static void validateActivations(
      ByteArray activations, FloatArray activationScales, int batchSize, int cols) {
    if (activations == null || activations.getSize() != Math.multiplyExact(batchSize, cols)) {
      throw new IllegalArgumentException("activations do not match the projection shape");
    }
    if (activationScales == null
        || activationScales.getSize() != Math.multiplyExact(batchSize, cols / SUPER_BLOCK_VALUES)) {
      throw new IllegalArgumentException("activation scales do not match the projection shape");
    }
  }

  private static void validateOutput(FloatArray output, int batchSize, int rows) {
    if (output == null || output.getSize() != Math.multiplyExact(batchSize, rows)) {
      throw new IllegalArgumentException("output does not match the projection shape");
    }
  }

  private static int ggmlNearestInt(float value) {
    int bits = Float.floatToRawIntBits(value + 12_582_912.0f);
    return (bits & 0x007F_FFFF) - 0x0040_0000;
  }
}
