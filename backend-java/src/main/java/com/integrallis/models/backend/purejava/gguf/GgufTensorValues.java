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
package com.integrallis.models.backend.purejava.gguf;

import com.integrallis.models.backend.purejava.quant.F16Dequantizer;
import com.integrallis.models.backend.purejava.quant.Q4_0Dequantizer;
import com.integrallis.models.backend.purejava.quant.Q8_0Dequantizer;
import com.integrallis.vectors.core.VectorUtil;
import java.lang.foreign.MemorySegment;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Objects;

/** Type-aware access to mapped GGUF tensor values without materializing model matrices. */
public final class GgufTensorValues {

  private static final ValueLayout.OfFloat LE_FLOAT =
      ValueLayout.JAVA_FLOAT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);
  private static final ValueLayout.OfShort LE_SHORT =
      ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  private GgufTensorValues() {}

  /** Materializes one small tensor as F32 values. */
  public static float[] toFloatArray(GgufTensorData tensor) {
    Objects.requireNonNull(tensor, "tensor");
    int count = Math.toIntExact(tensor.info().elementCount());
    float[] values = new float[count];
    dequantize(tensor.dataSegment(), tensor.type(), 0, values, count);
    return values;
  }

  /** Dequantizes one matrix row into a caller-owned F32 buffer. */
  public static void dequantizeRow(
      MemorySegment segment, GgufTensorType type, int row, int columns, float[] output) {
    Objects.requireNonNull(segment, "segment");
    Objects.requireNonNull(type, "type");
    Objects.requireNonNull(output, "output");
    if (row < 0) {
      throw new IllegalArgumentException("row must be >= 0: " + row);
    }
    if (columns <= 0 || columns % type.blockSize() != 0) {
      throw new IllegalArgumentException(
          "columns must be a positive multiple of " + type.blockSize() + ": " + columns);
    }
    if (output.length != columns) {
      throw new IllegalArgumentException(
          "output length must match columns: " + output.length + " != " + columns);
    }
    long bytesPerRow = Math.multiplyExact((long) columns / type.blockSize(), type.typeSize());
    long sourceOffset = Math.multiplyExact(row, bytesPerRow);
    if (sourceOffset > segment.byteSize() - bytesPerRow) {
      throw new IllegalArgumentException("row exceeds tensor storage: " + row);
    }
    dequantize(segment, type, sourceOffset, output, columns);
  }

  private static void dequantize(
      MemorySegment source, GgufTensorType type, long sourceOffset, float[] output, int count) {
    switch (type) {
      case F32 -> {
        for (int index = 0; index < count; index++) {
          output[index] = source.get(LE_FLOAT, sourceOffset + (long) index * Float.BYTES);
        }
      }
      case F16 -> new F16Dequantizer().dequantize(source, sourceOffset, output, 0, count);
      case BF16 -> {
        for (int index = 0; index < count; index++) {
          short bits = source.get(LE_SHORT, sourceOffset + (long) index * Short.BYTES);
          output[index] = Float.intBitsToFloat(Short.toUnsignedInt(bits) << Short.SIZE);
        }
      }
      case Q4_0 -> new Q4_0Dequantizer().dequantize(source, sourceOffset, output, 0, count);
      case Q5_0 -> VectorUtil.ggufQ5_0Dequantize(source, sourceOffset, output, 0, count);
      case Q8_0 -> new Q8_0Dequantizer().dequantize(source, sourceOffset, output, 0, count);
      case Q4_K -> VectorUtil.ggufQ4_KDequantize(source, sourceOffset, output, 0, count);
      case Q5_K -> VectorUtil.ggufQ5_KDequantize(source, sourceOffset, output, 0, count);
      case Q6_K -> VectorUtil.ggufQ6_KDequantize(source, sourceOffset, output, 0, count);
      default -> throw new IllegalArgumentException("Unsupported GGUF tensor type: " + type);
    }
  }
}
