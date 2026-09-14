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
package com.integrallis.models.backend.purejava.lora;

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.ops.TensorOps;
import java.lang.foreign.MemorySegment;
import java.util.Objects;

/** One mapped low-rank update: {@code output += scale * B * (A * input)}. */
final class LoraProjection {

  private final MemorySegment a;
  private final MemorySegment b;
  private final int inputDimension;
  private final int outputDimension;
  private final int rank;
  private final float scale;
  private final float[] inputScratch;
  private final float[] rankScratch;
  private final float[] outputScratch;

  LoraProjection(
      MemorySegment a,
      MemorySegment b,
      int inputDimension,
      int outputDimension,
      int rank,
      float scale) {
    this.a = Objects.requireNonNull(a, "a").asReadOnly();
    this.b = Objects.requireNonNull(b, "b").asReadOnly();
    positive("inputDimension", inputDimension);
    positive("outputDimension", outputDimension);
    positive("rank", rank);
    if (!Float.isFinite(scale)) {
      throw new IllegalArgumentException("scale must be finite: " + scale);
    }
    requireBytes("A", this.a, Math.multiplyExact(rank, inputDimension));
    requireBytes("B", this.b, Math.multiplyExact(outputDimension, rank));
    this.inputDimension = inputDimension;
    this.outputDimension = outputDimension;
    this.rank = rank;
    this.scale = scale;
    this.inputScratch = new float[inputDimension];
    this.rankScratch = new float[rank];
    this.outputScratch = new float[outputDimension];
  }

  void addTo(float[] output, int outputOffset, float[] input, int inputOffset) {
    Objects.requireNonNull(output, "output");
    Objects.requireNonNull(input, "input");
    Objects.checkFromIndexSize(outputOffset, outputDimension, output.length);
    Objects.checkFromIndexSize(inputOffset, inputDimension, input.length);

    float[] activeInput = input;
    if (inputOffset != 0 || input.length != inputDimension) {
      System.arraycopy(input, inputOffset, inputScratch, 0, inputDimension);
      activeInput = inputScratch;
    }
    TensorOps.ggufMatmul(rankScratch, activeInput, a, GgufTensorType.F32, rank, inputDimension);
    TensorOps.ggufMatmul(outputScratch, rankScratch, b, GgufTensorType.F32, outputDimension, rank);
    for (int index = 0; index < outputDimension; index++) {
      output[outputOffset + index] += scale * outputScratch[index];
    }
  }

  /**
   * Exercises the mapped F32 kernels with neutral input without changing later projection state.
   */
  void prewarm(int iterations) {
    if (iterations <= 0) {
      throw new IllegalArgumentException("iterations must be positive: " + iterations);
    }
    float[] input = new float[inputDimension];
    float[] output = new float[outputDimension];
    for (int iteration = 0; iteration < iterations; iteration++) {
      addTo(output, 0, input, 0);
    }
  }

  private static void requireBytes(String name, MemorySegment tensor, int elements) {
    long expected = Math.multiplyExact((long) elements, Float.BYTES);
    if (tensor.byteSize() != expected) {
      throw new IllegalArgumentException(
          name + " byte size must equal " + expected + ": " + tensor.byteSize());
    }
  }

  private static void positive(String name, int value) {
    if (value <= 0) throw new IllegalArgumentException(name + " must be > 0: " + value);
  }
}
