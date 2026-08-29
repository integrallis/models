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
package com.integrallis.models.accelerator;

import uk.ac.manchester.tornado.api.annotations.Parallel;
import uk.ac.manchester.tornado.api.types.arrays.ByteArray;
import uk.ac.manchester.tornado.api.types.arrays.FloatArray;

/** Java-authored TornadoVM experiment for production-compatible Q4_0 by Q8_0 projections. */
final class Q4ProjectionKernel {
  private static final int BLOCK_VALUES = 32;
  private static final int BLOCK_BYTES = 18;

  private Q4ProjectionKernel() {}

  /** Runs one work item per batch/output-row pair over host-prepared Q8_0 activations. */
  static void multiply(
      ByteArray weights,
      ByteArray activations,
      FloatArray activationScales,
      FloatArray output,
      int batchSize,
      int rows,
      int cols) {
    int blocks = cols / BLOCK_VALUES;
    for (@Parallel int outputIndex = 0; outputIndex < batchSize * rows; outputIndex++) {
      int batch = outputIndex / rows;
      int row = outputIndex - batch * rows;
      int activationOffset = batch * cols;
      int activationScaleOffset = batch * blocks;
      output.set(
          outputIndex,
          rowDotLow(
                  weights,
                  row,
                  activations,
                  activationOffset,
                  activationScales,
                  activationScaleOffset,
                  blocks)
              + rowDotHigh(
                  weights,
                  row,
                  activations,
                  activationOffset,
                  activationScales,
                  activationScaleOffset,
                  blocks));
    }
  }

  /** Computes two projections in one dispatch over one prepared activation. */
  static void multiplyDual(
      ByteArray firstWeights,
      int firstRows,
      ByteArray secondWeights,
      int secondRows,
      ByteArray activations,
      FloatArray activationScales,
      FloatArray firstOutput,
      FloatArray secondOutput,
      int batchSize,
      int cols) {
    int blocks = cols / BLOCK_VALUES;
    int combinedRows = firstRows + secondRows;
    for (@Parallel int outputIndex = 0; outputIndex < batchSize * combinedRows; outputIndex++) {
      int batch = outputIndex / combinedRows;
      int combinedRow = outputIndex - batch * combinedRows;
      int activationOffset = batch * cols;
      int activationScaleOffset = batch * blocks;
      if (combinedRow < firstRows) {
        firstOutput.set(
            batch * firstRows + combinedRow,
            rowDotLow(
                    firstWeights,
                    combinedRow,
                    activations,
                    activationOffset,
                    activationScales,
                    activationScaleOffset,
                    blocks)
                + rowDotHigh(
                    firstWeights,
                    combinedRow,
                    activations,
                    activationOffset,
                    activationScales,
                    activationScaleOffset,
                    blocks));
      } else {
        int row = combinedRow - firstRows;
        secondOutput.set(
            batch * secondRows + row,
            rowDotLow(
                    secondWeights,
                    row,
                    activations,
                    activationOffset,
                    activationScales,
                    activationScaleOffset,
                    blocks)
                + rowDotHigh(
                    secondWeights,
                    row,
                    activations,
                    activationOffset,
                    activationScales,
                    activationScaleOffset,
                    blocks));
      }
    }
  }

  /** Computes three projections in one dispatch over one prepared activation. */
  static void multiplyTriple(
      ByteArray firstWeights,
      int firstRows,
      ByteArray secondWeights,
      int secondRows,
      ByteArray thirdWeights,
      int thirdRows,
      ByteArray activations,
      FloatArray activationScales,
      FloatArray firstOutput,
      FloatArray secondOutput,
      FloatArray thirdOutput,
      int batchSize,
      int cols) {
    int blocks = cols / BLOCK_VALUES;
    int firstAndSecondRows = firstRows + secondRows;
    int combinedRows = firstAndSecondRows + thirdRows;
    for (@Parallel int outputIndex = 0; outputIndex < batchSize * combinedRows; outputIndex++) {
      int batch = outputIndex / combinedRows;
      int combinedRow = outputIndex - batch * combinedRows;
      int activationOffset = batch * cols;
      int activationScaleOffset = batch * blocks;
      if (combinedRow < firstRows) {
        firstOutput.set(
            batch * firstRows + combinedRow,
            rowDotLow(
                    firstWeights,
                    combinedRow,
                    activations,
                    activationOffset,
                    activationScales,
                    activationScaleOffset,
                    blocks)
                + rowDotHigh(
                    firstWeights,
                    combinedRow,
                    activations,
                    activationOffset,
                    activationScales,
                    activationScaleOffset,
                    blocks));
      } else if (combinedRow < firstAndSecondRows) {
        int row = combinedRow - firstRows;
        secondOutput.set(
            batch * secondRows + row,
            rowDotLow(
                    secondWeights,
                    row,
                    activations,
                    activationOffset,
                    activationScales,
                    activationScaleOffset,
                    blocks)
                + rowDotHigh(
                    secondWeights,
                    row,
                    activations,
                    activationOffset,
                    activationScales,
                    activationScaleOffset,
                    blocks));
      } else {
        int row = combinedRow - firstAndSecondRows;
        thirdOutput.set(
            batch * thirdRows + row,
            rowDotLow(
                    thirdWeights,
                    row,
                    activations,
                    activationOffset,
                    activationScales,
                    activationScaleOffset,
                    blocks)
                + rowDotHigh(
                    thirdWeights,
                    row,
                    activations,
                    activationOffset,
                    activationScales,
                    activationScaleOffset,
                    blocks));
      }
    }
  }

  private static float rowDotLow(
      ByteArray weights,
      int row,
      ByteArray activations,
      int activationOffset,
      FloatArray activationScales,
      int activationScaleOffset,
      int blocks) {
    int rowBlockOffset = row * blocks;
    float sum = 0.0f;
    for (int block = 0; block < blocks; block++) {
      int weightOffset = (rowBlockOffset + block) * BLOCK_BYTES;
      int quantOffset = activationOffset + block * BLOCK_VALUES;
      int integerSum = pairProductSumLow(weights, weightOffset, activations, quantOffset);
      float scale =
          weights.getHalfFloat(weightOffset).getFloat32()
              * activationScales.get(activationScaleOffset + block);
      sum += scale * integerSum;
    }
    return sum;
  }

  private static float rowDotHigh(
      ByteArray weights,
      int row,
      ByteArray activations,
      int activationOffset,
      FloatArray activationScales,
      int activationScaleOffset,
      int blocks) {
    int rowBlockOffset = row * blocks;
    float sum = 0.0f;
    for (int block = 0; block < blocks; block++) {
      int weightOffset = (rowBlockOffset + block) * BLOCK_BYTES;
      int quantOffset = activationOffset + block * BLOCK_VALUES;
      int integerSum = pairProductSumHigh(weights, weightOffset, activations, quantOffset);
      float scale =
          weights.getHalfFloat(weightOffset).getFloat32()
              * activationScales.get(activationScaleOffset + block);
      sum += scale * integerSum;
    }
    return sum;
  }

  private static int pairProductSumLow(
      ByteArray weights, int weightOffset, ByteArray activations, int activationOffset) {
    return pairProduct(weights, weightOffset, activations, activationOffset, 0)
        + pairProduct(weights, weightOffset, activations, activationOffset, 1)
        + pairProduct(weights, weightOffset, activations, activationOffset, 2)
        + pairProduct(weights, weightOffset, activations, activationOffset, 3)
        + pairProduct(weights, weightOffset, activations, activationOffset, 4)
        + pairProduct(weights, weightOffset, activations, activationOffset, 5)
        + pairProduct(weights, weightOffset, activations, activationOffset, 6)
        + pairProduct(weights, weightOffset, activations, activationOffset, 7);
  }

  private static int pairProductSumHigh(
      ByteArray weights, int weightOffset, ByteArray activations, int activationOffset) {
    return pairProduct(weights, weightOffset, activations, activationOffset, 8)
        + pairProduct(weights, weightOffset, activations, activationOffset, 9)
        + pairProduct(weights, weightOffset, activations, activationOffset, 10)
        + pairProduct(weights, weightOffset, activations, activationOffset, 11)
        + pairProduct(weights, weightOffset, activations, activationOffset, 12)
        + pairProduct(weights, weightOffset, activations, activationOffset, 13)
        + pairProduct(weights, weightOffset, activations, activationOffset, 14)
        + pairProduct(weights, weightOffset, activations, activationOffset, 15);
  }

  private static int pairProduct(
      ByteArray weights, int weightOffset, ByteArray activations, int activationOffset, int index) {
    int packed = weights.get(weightOffset + 2 + index) & 0xFF;
    int low = (packed & 0x0F) - 8;
    int high = ((packed >>> 4) & 0x0F) - 8;
    return low * activations.get(activationOffset + index)
        + high * activations.get(activationOffset + index + 16);
  }

  /** Reproduces the production Q8_0 activation preparation, including FP16 scale rounding. */
  static void quantize(float[] input, byte[] activations, float[] scales, int batchSize, int cols) {
    int blocks = cols / BLOCK_VALUES;
    for (int batch = 0; batch < batchSize; batch++) {
      for (int block = 0; block < blocks; block++) {
        int offset = batch * cols + block * BLOCK_VALUES;
        float absoluteMax = 0.0f;
        for (int index = 0; index < BLOCK_VALUES; index++) {
          absoluteMax = Math.max(absoluteMax, Math.abs(input[offset + index]));
        }
        float scale = absoluteMax / 127.0f;
        float inverseScale = absoluteMax == 0.0f ? 0.0f : 127.0f / absoluteMax;
        scales[batch * blocks + block] = Float.float16ToFloat(Float.floatToFloat16(scale));
        for (int index = 0; index < BLOCK_VALUES; index++) {
          activations[offset + index] = (byte) ggmlNearestInt(input[offset + index] * inverseScale);
        }
      }
    }
  }

  static void validate(
      ByteArray weights,
      ByteArray activations,
      FloatArray activationScales,
      FloatArray output,
      int batchSize,
      int rows,
      int cols) {
    if (batchSize < 1) {
      throw new IllegalArgumentException("batchSize must be positive");
    }
    if (rows < 1) {
      throw new IllegalArgumentException("rows must be positive");
    }
    if (cols < BLOCK_VALUES || cols % BLOCK_VALUES != 0) {
      throw new IllegalArgumentException("cols must be a positive multiple of 32");
    }
    int blocks = cols / BLOCK_VALUES;
    int expectedWeightBytes = Math.multiplyExact(Math.multiplyExact(rows, blocks), BLOCK_BYTES);
    if (weights.getSize() != expectedWeightBytes) {
      throw new IllegalArgumentException("weights do not match the projection shape");
    }
    if (activations.getSize() != Math.multiplyExact(batchSize, cols)) {
      throw new IllegalArgumentException("activations do not match the projection shape");
    }
    if (activationScales.getSize() != Math.multiplyExact(batchSize, blocks)) {
      throw new IllegalArgumentException("activation scales do not match the projection shape");
    }
    if (output.getSize() != Math.multiplyExact(batchSize, rows)) {
      throw new IllegalArgumentException("output does not match the projection shape");
    }
  }

  private static int ggmlNearestInt(float value) {
    int bits = Float.floatToRawIntBits(value + 12_582_912.0f);
    return (bits & 0x007F_FFFF) - 0x0040_0000;
  }
}
