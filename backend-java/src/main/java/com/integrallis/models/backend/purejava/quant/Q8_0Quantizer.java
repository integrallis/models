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
package com.integrallis.models.backend.purejava.quant;

import java.lang.foreign.Arena;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;

/**
 * Converts float32 weight arrays to GGUF Q8_0 format for use with native kernels.
 *
 * <p>Each block of 32 float32 values becomes 34 bytes:
 *
 * <ul>
 *   <li>2 bytes: f16 scale = max_abs / 127.0f
 *   <li>32 bytes: signed int8 quantized values round(value / scale)
 * </ul>
 *
 * <p>This is a load-time operation; the quantized weights are then passed to {@code
 * GgufBatchedMatrixKernel} which routes to Rust SIMD kernels.
 */
public final class Q8_0Quantizer {

  private static final int BLOCK_SIZE = 32;
  private static final int BLOCK_BYTES = 34;

  /** Maximum quantized value for Q8_0 (one less than 2^7 to preserve symmetry). */
  private static final int Q8_0_MAX = 127;

  private static final ValueLayout.OfShort LE_SHORT =
      ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  private Q8_0Quantizer() {}

  /**
   * Quantizes a float32 weight matrix to Q8_0 in a new native MemorySegment.
   *
   * @param weights row-major float32 weights, length = rows * columns
   * @param rows number of output rows
   * @param columns number of input columns (must be a multiple of 32)
   * @param arena arena to allocate the native segment in
   * @return a native MemorySegment containing packed Q8_0 blocks
   */
  public static MemorySegment quantize(float[] weights, int rows, int columns, Arena arena) {
    if (columns % BLOCK_SIZE != 0) {
      throw new IllegalArgumentException(
          "columns must be a multiple of " + BLOCK_SIZE + ": " + columns);
    }
    if (weights.length != rows * columns) {
      throw new IllegalArgumentException(
          "weights length " + weights.length + " != rows * columns " + (rows * columns));
    }

    int blocksPerRow = columns / BLOCK_SIZE;
    long totalBlocks = (long) rows * blocksPerRow;
    long bytes = totalBlocks * BLOCK_BYTES;
    MemorySegment segment = arena.allocate(bytes);

    for (int row = 0; row < rows; row++) {
      int rowOffset = row * columns;
      for (int block = 0; block < blocksPerRow; block++) {
        int blockOffset = rowOffset + block * BLOCK_SIZE;

        // Find max absolute value in this block
        float maxAbs = 0.0f;
        for (int i = 0; i < BLOCK_SIZE; i++) {
          float abs = Math.abs(weights[blockOffset + i]);
          if (abs > maxAbs) {
            maxAbs = abs;
          }
        }

        // Compute scale
        float scale = maxAbs > 0.0f ? maxAbs / Q8_0_MAX : 1.0f;
        short scaleBits = Float16.fromFloat(scale);

        // Quantize each element
        byte[] quantized = new byte[BLOCK_SIZE];
        for (int i = 0; i < BLOCK_SIZE; i++) {
          float q = weights[blockOffset + i] / scale;
          int qi = Math.round(q);
          if (qi < -Q8_0_MAX) qi = -Q8_0_MAX;
          if (qi > Q8_0_MAX) qi = Q8_0_MAX;
          quantized[i] = (byte) qi;
        }

        // Write block to segment
        long segmentBlockOffset = ((long) row * blocksPerRow + block) * BLOCK_BYTES;
        segment.set(LE_SHORT, segmentBlockOffset, scaleBits);
        for (int i = 0; i < BLOCK_SIZE; i++) {
          segment.set(ValueLayout.JAVA_BYTE, segmentBlockOffset + 2 + i, quantized[i]);
        }
      }
    }

    return segment;
  }
}
