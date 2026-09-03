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

/** Row-major Q8_0 runtime layout for a transposed MobileMoE expert projection. */
final class MobileMoeQ8RightMatrix {

  private final MemorySegment weights;
  private final int inputs;
  private final int outputs;

  MobileMoeQ8RightMatrix(MemorySegment weights, int inputs, int outputs) {
    this.weights = Objects.requireNonNull(weights, "weights").asReadOnly();
    this.inputs = inputs;
    this.outputs = outputs;
    if (inputs <= 0 || inputs % 32 != 0 || outputs <= 0) {
      throw new IllegalArgumentException(
          "Q8_0 expert matrix dimensions must be positive and inputs divisible by 32: "
              + outputs
              + "x"
              + inputs);
    }
    long expectedBytes = Math.multiplyExact(Math.multiplyExact((long) outputs, inputs / 32L), 34L);
    if (weights.byteSize() != expectedBytes) {
      throw new IllegalArgumentException(
          "invalid Q8_0 expert matrix: "
              + outputs
              + "x"
              + inputs
              + " requires "
              + expectedBytes
              + " bytes; got "
              + weights.byteSize());
    }
  }

  void multiply(
      float[] input, float[] output, byte[] quantizedActivation, float[] activationScales) {
    multiplyBatch(input, 1, output, quantizedActivation, activationScales);
  }

  void multiplyBatch(
      float[] input,
      int batchSize,
      float[] output,
      byte[] quantizedActivation,
      float[] activationScales) {
    VectorUtil.ggufQ8_0Q8_0BatchedMatmul(
        input, weights, batchSize, outputs, inputs, output, quantizedActivation, activationScales);
  }

  static void multiplyDualBatch(
      float[] input,
      int batchSize,
      MobileMoeQ8RightMatrix first,
      float[] firstOutput,
      MobileMoeQ8RightMatrix second,
      float[] secondOutput,
      byte[] quantizedActivation,
      float[] activationScales) {
    requireSameInputs(first, second);
    VectorUtil.ggufQ8_0Q8_0DualBatchedMatmul(
        input,
        first.weights,
        first.outputs,
        firstOutput,
        second.weights,
        second.outputs,
        secondOutput,
        batchSize,
        first.inputs,
        quantizedActivation,
        activationScales);
  }

  static void multiplyTripleBatch(
      float[] input,
      int batchSize,
      MobileMoeQ8RightMatrix first,
      float[] firstOutput,
      MobileMoeQ8RightMatrix second,
      float[] secondOutput,
      MobileMoeQ8RightMatrix third,
      float[] thirdOutput,
      byte[] quantizedActivation,
      float[] activationScales) {
    requireSameInputs(first, second);
    requireSameInputs(first, third);
    VectorUtil.ggufQ8_0Q8_0TripleBatchedMatmul(
        input,
        first.weights,
        first.outputs,
        firstOutput,
        second.weights,
        second.outputs,
        secondOutput,
        third.weights,
        third.outputs,
        thirdOutput,
        batchSize,
        first.inputs,
        quantizedActivation,
        activationScales);
  }

  private static void requireSameInputs(
      MobileMoeQ8RightMatrix first, MobileMoeQ8RightMatrix other) {
    Objects.requireNonNull(first, "first");
    Objects.requireNonNull(other, "other");
    if (first.inputs != other.inputs) {
      throw new IllegalArgumentException(
          "Q8_0 grouped matrices must have equal input dimensions: "
              + first.inputs
              + " != "
              + other.inputs);
    }
  }
}
