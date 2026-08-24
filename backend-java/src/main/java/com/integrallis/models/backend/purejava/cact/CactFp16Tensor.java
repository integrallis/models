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
package com.integrallis.models.backend.purejava.cact;

import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Objects;

/** Read-only, zero-copy view of one little-endian FP16 `.cact` tensor. */
public final class CactFp16Tensor {

  private static final ValueLayout.OfShort LE_SHORT =
      ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  private final long[] shape;
  private final long[] strides;
  private final MemorySegment data;
  private final int elementCount;

  private CactFp16Tensor(long[] shape, MemorySegment data) {
    this.shape = shape;
    this.data = data;
    this.strides = new long[shape.length];
    long stride = 1;
    for (int dimension = shape.length - 1; dimension >= 0; dimension--) {
      strides[dimension] = stride;
      stride = Math.multiplyExact(stride, shape[dimension]);
    }
    this.elementCount = Math.toIntExact(stride);
  }

  /** Creates a typed view over an FP16 tensor's mapped payload. */
  public static CactFp16Tensor from(CactTensorData tensor) {
    Objects.requireNonNull(tensor, "tensor");
    if (tensor.info().type() != CactTensorType.FP16) {
      throw new IllegalArgumentException("tensor must have CACT FP16 type");
    }
    return new CactFp16Tensor(tensor.info().shape(), tensor.data());
  }

  /** Returns the tensor shape. */
  public long[] shape() {
    return shape.clone();
  }

  /** Returns one element addressed in row-major dimension order. */
  public float get(int... indices) {
    Objects.requireNonNull(indices, "indices");
    if (indices.length != shape.length) {
      throw new IllegalArgumentException(
          "expected " + shape.length + " indices; got " + indices.length);
    }
    long linear = 0;
    for (int dimension = 0; dimension < indices.length; dimension++) {
      int index = indices[dimension];
      if (index < 0 || index >= shape[dimension]) {
        throw new IndexOutOfBoundsException(
            "index " + index + " outside dimension " + dimension + " of " + shape[dimension]);
      }
      linear += Math.multiplyExact((long) index, strides[dimension]);
    }
    return read(linear);
  }

  /** Copies one row from a rank-two tensor. */
  public float[] readRow(int row) {
    if (shape.length != 2) {
      throw new IllegalStateException("readRow requires a rank-two FP16 tensor");
    }
    int columns = Math.toIntExact(shape[1]);
    float[] result = new float[columns];
    copyRow(row, result);
    return result;
  }

  /** Copies one row from a rank-two tensor into caller-owned storage. */
  public void copyRow(int row, float[] output) {
    Objects.requireNonNull(output, "output");
    if (shape.length != 2) {
      throw new IllegalStateException("copyRow requires a rank-two FP16 tensor");
    }
    int rows = Math.toIntExact(shape[0]);
    int columns = Math.toIntExact(shape[1]);
    if (row < 0 || row >= rows) {
      throw new IndexOutOfBoundsException("row " + row + " outside tensor with " + rows + " rows");
    }
    if (output.length < columns) {
      throw new IllegalArgumentException(
          "output requires at least " + columns + " elements; got " + output.length);
    }
    long start = Math.multiplyExact((long) row, columns);
    for (int column = 0; column < columns; column++) {
      output[column] = read(start + column);
    }
  }

  /** Decodes every element in row-major order. */
  public float[] readAll() {
    float[] result = new float[elementCount];
    for (int index = 0; index < elementCount; index++) {
      result[index] = read(index);
    }
    return result;
  }

  private float read(long index) {
    short bits = data.get(LE_SHORT, Math.multiplyExact(index, Short.BYTES));
    return Float.float16ToFloat(bits);
  }
}
