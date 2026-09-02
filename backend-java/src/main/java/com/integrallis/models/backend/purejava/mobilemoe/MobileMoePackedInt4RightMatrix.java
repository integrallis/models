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

/** MobileMoE expert tensor stored as {@code [input, packed-output]} with output-group scales. */
final class MobileMoePackedInt4RightMatrix {

  private static final ValueLayout.OfShort LE_SHORT =
      ValueLayout.JAVA_SHORT_UNALIGNED.withOrder(ByteOrder.LITTLE_ENDIAN);

  private final MemorySegment packed;
  private final MemorySegment scales;
  private final int inputs;
  private final int outputs;
  private final int groupSize;
  private final int groupsPerInput;

  private MobileMoePackedInt4RightMatrix(
      MemorySegment packed,
      MemorySegment scales,
      int inputs,
      int outputs,
      int groupSize,
      int groupsPerInput) {
    this.packed = packed.asReadOnly();
    this.scales = scales.asReadOnly();
    this.inputs = inputs;
    this.outputs = outputs;
    this.groupSize = groupSize;
    this.groupsPerInput = groupsPerInput;
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
    return new MobileMoePackedInt4RightMatrix(packed, scales, inputs, outputs, groupSize, groups);
  }

  void multiply(float[] input, float[] output) {
    Objects.requireNonNull(input, "input");
    Objects.requireNonNull(output, "output");
    if (input.length != inputs || output.length != outputs) {
      throw new IllegalArgumentException(
          "input/output lengths must equal " + inputs + "/" + outputs);
    }
    Arrays.fill(output, 0.0f);
    for (int inputIndex = 0; inputIndex < inputs; inputIndex++) {
      float activation = input[inputIndex];
      for (int group = 0; group < groupsPerInput; group++) {
        float multiplier = activation * scale(inputIndex, group);
        int start = group * groupSize;
        int end = start + groupSize;
        for (int outputIndex = start; outputIndex < end; outputIndex += 2) {
          int bits = packedByte(inputIndex, outputIndex);
          output[outputIndex] += multiplier * signedNibble(bits & 0x0f);
          output[outputIndex + 1] += multiplier * signedNibble(bits >>> 4);
        }
      }
    }
  }

  private int packedByte(int input, int evenOutput) {
    long index = ((long) input * outputs + evenOutput) / 2L;
    return Byte.toUnsignedInt(packed.get(ValueLayout.JAVA_BYTE, index));
  }

  private float scale(int input, int group) {
    long index = (long) input * groupsPerInput + group;
    return Float.float16ToFloat(scales.get(LE_SHORT, index * Short.BYTES));
  }

  private static int signedNibble(int nibble) {
    return nibble > 7 ? nibble - 16 : nibble;
  }
}
