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
package com.integrallis.models.backend.purejava.mobilemoe;

import com.integrallis.vectors.core.VectorUtil;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Objects;

/** Zero-copy row-major MobileMoE symmetric INT4 matrix with little-endian FP16 group scales. */
final class MobileMoePackedInt4Matrix {

  private static final ValueLayout.OfShort LE_SHORT =
      ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  private final MemorySegment packed;
  private final MemorySegment scales;
  private final int rows;
  private final int columns;
  private final int groupSize;
  private final int groupsPerRow;

  private MobileMoePackedInt4Matrix(
      MemorySegment packed,
      MemorySegment scales,
      int rows,
      int columns,
      int groupSize,
      int groupsPerRow) {
    this.packed = packed.asReadOnly();
    this.scales = scales.asReadOnly();
    this.rows = rows;
    this.columns = columns;
    this.groupSize = groupSize;
    this.groupsPerRow = groupsPerRow;
  }

  static MobileMoePackedInt4Matrix of(
      MemorySegment packed, MemorySegment scales, int rows, int columns, int groupSize) {
    Objects.requireNonNull(packed, "packed");
    Objects.requireNonNull(scales, "scales");
    if (rows <= 0) {
      throw new IllegalArgumentException("rows must be positive: " + rows);
    }
    if (columns <= 0 || (columns & 1) != 0 || columns % groupSize != 0) {
      throw new IllegalArgumentException(
          "columns must be positive, even, and divisible by groupSize: " + columns);
    }
    if (groupSize <= 0 || (groupSize & 1) != 0) {
      throw new IllegalArgumentException("groupSize must be positive and even: " + groupSize);
    }
    long expectedPacked = Math.multiplyExact((long) rows, columns / 2L);
    if (packed.byteSize() != expectedPacked) {
      throw new IllegalArgumentException(
          "packed buffer requires " + expectedPacked + " bytes; got " + packed.byteSize());
    }
    int groupsPerRow = columns / groupSize;
    long expectedScales =
        Math.multiplyExact(Math.multiplyExact((long) rows, groupsPerRow), Short.BYTES);
    if (scales.byteSize() != expectedScales) {
      throw new IllegalArgumentException(
          "scale buffer requires " + expectedScales + " bytes; got " + scales.byteSize());
    }
    return new MobileMoePackedInt4Matrix(packed, scales, rows, columns, groupSize, groupsPerRow);
  }

  int rows() {
    return rows;
  }

  int columns() {
    return columns;
  }

  float value(int row, int column) {
    if (row < 0 || row >= rows || column < 0 || column >= columns) {
      throw new IndexOutOfBoundsException("matrix index (" + row + ", " + column + ")");
    }
    return quantized(row, column) * scale(row, column / groupSize);
  }

  void multiply(float[] input, float[] output) {
    Objects.requireNonNull(input, "input");
    Objects.requireNonNull(output, "output");
    if (input.length != columns) {
      throw new IllegalArgumentException(
          "input length must equal matrix columns " + columns + "; got " + input.length);
    }
    if (output.length != rows) {
      throw new IllegalArgumentException(
          "output length must equal matrix rows " + rows + "; got " + output.length);
    }
    VectorUtil.packedInt4GroupMatVec(input, packed, scales, rows, columns, groupSize, output);
  }

  void multiplyBatch(float[] input, int batchSize, float[] output) {
    Objects.requireNonNull(input, "input");
    Objects.requireNonNull(output, "output");
    if (input.length < Math.multiplyExact(batchSize, columns)) {
      throw new IllegalArgumentException(
          "input length must cover batchSize * columns; got " + input.length);
    }
    if (output.length < Math.multiplyExact(batchSize, rows)) {
      throw new IllegalArgumentException(
          "output length must cover batchSize * rows; got " + output.length);
    }
    VectorUtil.packedInt4GroupMatVecBatch(
        input, batchSize, packed, scales, rows, columns, groupSize, output);
  }

  private int quantized(int row, int column) {
    int bits = packedByte(row, column);
    return signedNibble((column & 1) == 0 ? bits & 0x0f : bits >>> 4);
  }

  private int packedByte(int row, int evenColumn) {
    long index = ((long) row * columns + evenColumn) / 2L;
    return Byte.toUnsignedInt(packed.get(ValueLayout.JAVA_BYTE, index));
  }

  private float scale(int row, int group) {
    long index = (long) row * groupsPerRow + group;
    return Float.float16ToFloat(scales.get(LE_SHORT, index * Short.BYTES));
  }

  private static int signedNibble(int nibble) {
    return nibble > 7 ? nibble - 16 : nibble;
  }
}
