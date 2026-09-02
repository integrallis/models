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
import java.util.Objects;

/** MobileMoE expert tensor stored as {@code [input, packed-output]} with output-group scales. */
final class MobileMoePackedInt4RightMatrix {

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
}
