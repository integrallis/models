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

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Arrays;
import java.util.Objects;

/** Zero-copy little-endian F32 row-major matrix used by MobileMoE routers. */
final class MobileMoeFloat32Matrix {

  private static final ValueLayout.OfFloat LE_FLOAT =
      ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  private final MemorySegment data;
  private final int rows;
  private final int columns;

  private MobileMoeFloat32Matrix(MemorySegment data, int rows, int columns) {
    this.data = data.asReadOnly();
    this.rows = rows;
    this.columns = columns;
  }

  static MobileMoeFloat32Matrix of(MemorySegment data, int rows, int columns) {
    Objects.requireNonNull(data, "data");
    if (rows <= 0 || columns <= 0) {
      throw new IllegalArgumentException("rows and columns must be positive");
    }
    long expected = Math.multiplyExact(Math.multiplyExact((long) rows, columns), Float.BYTES);
    if (data.byteSize() != expected) {
      throw new IllegalArgumentException(
          "F32 matrix requires " + expected + " bytes; got " + data.byteSize());
    }
    return new MobileMoeFloat32Matrix(data, rows, columns);
  }

  void multiply(float[] input, float[] output) {
    Objects.requireNonNull(input, "input");
    Objects.requireNonNull(output, "output");
    if (input.length != columns || output.length != rows) {
      throw new IllegalArgumentException("input/output lengths must equal " + columns + "/" + rows);
    }
    for (int row = 0; row < rows; row++) {
      float sum = 0.0f;
      long offset = (long) row * columns * Float.BYTES;
      for (int column = 0; column < columns; column++) {
        sum += data.get(LE_FLOAT, offset + (long) column * Float.BYTES) * input[column];
      }
      output[row] = sum;
    }
  }

  void multiplyBatch(float[] input, int batchSize, float[] output) {
    Objects.requireNonNull(input, "input");
    Objects.requireNonNull(output, "output");
    int inputEntries = Math.multiplyExact(batchSize, columns);
    int outputEntries = Math.multiplyExact(batchSize, rows);
    if (input.length < inputEntries || output.length < outputEntries) {
      throw new IllegalArgumentException(
          "input/output lengths must cover " + inputEntries + "/" + outputEntries);
    }
    Arrays.fill(output, 0, outputEntries, 0.0f);
    for (int row = 0; row < rows; row++) {
      long offset = (long) row * columns * Float.BYTES;
      for (int column = 0; column < columns; column++) {
        float weight = data.get(LE_FLOAT, offset + (long) column * Float.BYTES);
        for (int batch = 0; batch < batchSize; batch++) {
          output[batch * rows + row] += weight * input[batch * columns + column];
        }
      }
    }
  }
}
