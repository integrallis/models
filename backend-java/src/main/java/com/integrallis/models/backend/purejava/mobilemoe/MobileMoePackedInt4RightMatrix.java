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
import java.lang.foreign.SegmentAllocator;
import java.lang.foreign.ValueLayout;
import java.nio.ByteOrder;
import java.util.Objects;

/** MobileMoE expert tensor stored as {@code [input, packed-output]} with output-group scales. */
final class MobileMoePackedInt4RightMatrix {

  private static final ValueLayout.OfShort LE_SHORT =
      ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  private final MemorySegment packed;
  private final MemorySegment scales;
  private final int inputs;
  private final int outputs;
  private final int groupSize;

  private MobileMoePackedInt4RightMatrix(
      MemorySegment packed, MemorySegment scales, int inputs, int outputs, int groupSize) {
    this.packed = packed.asReadOnly();
    this.scales = scales.asReadOnly();
    this.inputs = inputs;
    this.outputs = outputs;
    this.groupSize = groupSize;
  }

  static MobileMoePackedInt4RightMatrix of(
      MemorySegment packed, MemorySegment scales, int inputs, int outputs, int groupSize) {
    Objects.requireNonNull(packed, "packed");
    Objects.requireNonNull(scales, "scales");
    if (inputs <= 0) {
      throw new IllegalArgumentException("inputs must be positive: " + inputs);
    }
    if (groupSize <= 0 || (groupSize & 1) != 0) {
      throw new IllegalArgumentException("groupSize must be positive and even: " + groupSize);
    }
    if (outputs <= 0 || (outputs & 1) != 0 || outputs % groupSize != 0) {
      throw new IllegalArgumentException(
          "outputs must be positive, even, and divisible by groupSize: " + outputs);
    }
    long expectedPacked = Math.multiplyExact((long) inputs, outputs / 2L);
    if (packed.byteSize() != expectedPacked) {
      throw new IllegalArgumentException(
          "packed buffer requires " + expectedPacked + " bytes; got " + packed.byteSize());
    }
    int groups = outputs / groupSize;
    long expectedScales = Math.multiplyExact(Math.multiplyExact((long) inputs, groups), 2L);
    if (scales.byteSize() != expectedScales) {
      throw new IllegalArgumentException(
          "scale buffer requires " + expectedScales + " bytes; got " + scales.byteSize());
    }
    return new MobileMoePackedInt4RightMatrix(packed, scales, inputs, outputs, groupSize);
  }

  void multiply(float[] input, float[] output) {
    Objects.requireNonNull(input, "input");
    Objects.requireNonNull(output, "output");
    if (input.length != inputs || output.length != outputs) {
      throw new IllegalArgumentException(
          "input/output lengths must equal " + inputs + "/" + outputs);
    }
    VectorUtil.packedInt4GroupRightMatVec(
        input, packed, scales, inputs, outputs, groupSize, output);
  }

  void multiplyBatch(float[] input, int batchSize, float[] output) {
    Objects.requireNonNull(input, "input");
    Objects.requireNonNull(output, "output");
    if (input.length < Math.multiplyExact(batchSize, inputs)) {
      throw new IllegalArgumentException(
          "input length must cover batchSize * inputs; got " + input.length);
    }
    if (output.length < Math.multiplyExact(batchSize, outputs)) {
      throw new IllegalArgumentException(
          "output length must cover batchSize * outputs; got " + output.length);
    }
    VectorUtil.packedInt4GroupRightMatVecBatch(
        input, batchSize, packed, scales, inputs, outputs, groupSize, output);
  }

  MobileMoeQ8RightMatrix materializeQ8(SegmentAllocator allocator) {
    Objects.requireNonNull(allocator, "allocator");
    int blocksPerRow = inputs / 32;
    long byteCount = Math.multiplyExact(Math.multiplyExact((long) outputs, blocksPerRow), 34L);
    MemorySegment q8 = allocator.allocate(byteCount, 1);
    long offset = 0L;
    float[] block = new float[32];
    for (int output = 0; output < outputs; output++) {
      for (int inputBlock = 0; inputBlock < blocksPerRow; inputBlock++) {
        float maximum = 0.0f;
        for (int index = 0; index < block.length; index++) {
          float value = value(inputBlock * block.length + index, output);
          block[index] = value;
          maximum = Math.max(maximum, Math.abs(value));
        }
        float scale = maximum / 127.0f;
        q8.set(LE_SHORT, offset, Float.floatToFloat16(scale));
        offset += Short.BYTES;
        float inverse = maximum == 0.0f ? 0.0f : 127.0f / maximum;
        for (float value : block) {
          int quantized = Math.clamp(ggmlNearestInt(value * inverse), -127, 127);
          q8.set(ValueLayout.JAVA_BYTE, offset++, (byte) quantized);
        }
      }
    }
    return new MobileMoeQ8RightMatrix(q8, inputs, outputs);
  }

  private float value(int input, int output) {
    int packedValue =
        Byte.toUnsignedInt(
            packed.get(ValueLayout.JAVA_BYTE, ((long) input * outputs + output) / 2L));
    int nibble = (output & 1) == 0 ? packedValue & 0x0f : packedValue >>> 4;
    int quantized = nibble < 8 ? nibble : nibble - 16;
    int groupsPerInput = outputs / groupSize;
    long scaleIndex = (long) input * groupsPerInput + output / groupSize;
    float scale = Float.float16ToFloat(scales.get(LE_SHORT, scaleIndex * Short.BYTES));
    return quantized * scale;
  }

  private static int ggmlNearestInt(float value) {
    int bits = Float.floatToRawIntBits(value + 12_582_912.0f);
    return (bits & 0x007f_ffff) - 0x0040_0000;
  }
}
