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

  default String implementation() {
    return "java";
  }

  @Override
  default void close() {}
}
