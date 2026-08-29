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

/** Java-authored TornadoVM experiment for row-major GGUF Q8_0 projections. */
final class Q8ProjectionKernel {
  private static final int BLOCK_VALUES = 32;
  private static final int BLOCK_BYTES = 34;

  private Q8ProjectionKernel() {}

  /** Runs one work item per batch/output-row pair. */
  static void multiply(
      ByteArray weights, FloatArray input, FloatArray output, int batchSize, int rows, int cols) {
    for (@Parallel int outputIndex = 0; outputIndex < batchSize * rows; outputIndex++) {
      int batch = outputIndex / rows;
      int row = outputIndex - batch * rows;
      int rowBlockOffset = row * (cols / BLOCK_VALUES);
      int inputOffset = batch * cols;
      float sum = 0.0f;
      for (int col = 0; col < cols; col++) {
        int block = col / BLOCK_VALUES;
        int withinBlock = col - block * BLOCK_VALUES;
        int blockByteOffset = (rowBlockOffset + block) * BLOCK_BYTES;
        float scale = weights.getHalfFloat(blockByteOffset).getFloat32();
        byte quantized = weights.get(blockByteOffset + 2 + withinBlock);
        sum += ((float) quantized * scale) * input.get(inputOffset + col);
      }
      output.set(outputIndex, sum);
    }
  }

  static void validate(
      ByteArray weights, FloatArray input, FloatArray output, int batchSize, int rows, int cols) {
    if (batchSize < 1) {
      throw new IllegalArgumentException("batchSize must be positive");
    }
    if (rows < 1) {
      throw new IllegalArgumentException("rows must be positive");
    }
    if (cols < BLOCK_VALUES || cols % BLOCK_VALUES != 0) {
      throw new IllegalArgumentException("cols must be a positive multiple of 32");
    }
    int expectedWeightBytes =
        Math.multiplyExact(Math.multiplyExact(rows, cols / BLOCK_VALUES), BLOCK_BYTES);
    if (weights.getSize() != expectedWeightBytes) {
      throw new IllegalArgumentException("weights do not match the projection shape");
    }
    if (input.getSize() != Math.multiplyExact(batchSize, cols)) {
      throw new IllegalArgumentException("input does not match the projection shape");
    }
    if (output.getSize() != Math.multiplyExact(batchSize, rows)) {
      throw new IllegalArgumentException("output does not match the projection shape");
    }
  }
}
