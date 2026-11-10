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
package com.integrallis.models.backend.purejava.spi;

/** Owned prepared F32 matrix multiplications shared by encoder architectures. */
public interface DenseProjectionKernels extends AutoCloseable {
  @FunctionalInterface
  interface Projection {
    /** Computes output[batch, outputs] = input[batch, inputs] * transpose(weights). */
    void project(float[] output, float[] input, int batch);
  }

  /** Copies immutable weights into provider-owned storage once. */
  Projection prepare(float[] weights, int outputs, int inputs);

  /** Copies immutable weights and bias for an affine projection. */
  default Projection prepareAffine(float[] weights, float[] bias, int outputs, int inputs) {
    if (bias.length != outputs) throw new IllegalArgumentException("Invalid projection bias");
    float[] ownedBias = bias.clone();
    Projection projection = prepare(weights, outputs, inputs);
    return (output, input, batch) -> {
      projection.project(output, input, batch);
      for (int i = 0; i < batch * outputs; i++) output[i] += ownedBias[i % outputs];
    };
  }

  /** Prepares a projection with fused accumulation in increasing input-index order. */
  default Projection prepareOrdered(float[] weights, int outputs, int inputs) {
    if (outputs < 1 || inputs < 1 || weights.length != Math.multiplyExact(outputs, inputs))
      throw new IllegalArgumentException("Invalid ordered projection shape");
    float[] owned = weights.clone();
    return (output, input, batch) -> {
      if (batch < 1
          || input.length < Math.multiplyExact(batch, inputs)
          || output.length < Math.multiplyExact(batch, outputs)
          || output == input)
        throw new IllegalArgumentException("Invalid ordered projection buffers");
      for (int row = 0; row < batch; row++)
        for (int column = 0; column < outputs; column++) {
          float sum = 0;
          for (int i = 0; i < inputs; i++)
            sum = Math.fma(input[row * inputs + i], owned[column * inputs + i], sum);
          output[row * outputs + column] = sum;
        }
    };
  }

  /** Performs ordinary multi-head attention when supported; false leaves output untouched. */
  default boolean attention(
      float[] output,
      float[] query,
      float[] key,
      float[] value,
      int tokens,
      int dimension,
      int heads) {
    return false;
  }

  /** Applies QuickGELU in place, or leaves the array untouched when unavailable. */
  default boolean quickGelu(float[] values) {
    return false;
  }

  /** Applies ERF GELU in place, or leaves the array untouched when unavailable. */
  default boolean geluErf(float[] values) {
    return false;
  }

  default String implementation() {
    return "java";
  }

  @Override
  default void close() {}
}
