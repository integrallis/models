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
package com.integrallis.models.backend.purejava.deberta;

/** Synchronous dense projection and disentangled attention over caller-owned row-major buffers. */
public interface DebertaV2Kernels
    extends com.integrallis.models.backend.purejava.spi.DenseProjectionKernels {
  /** A prepared immutable matrix whose storage is owned by the enclosing kernel provider. */
  @FunctionalInterface
  interface Projection
      extends com.integrallis.models.backend.purejava.spi.DenseProjectionKernels.Projection {
    /** Computes output[batch, outputs] = input[batch, inputs] * transpose(prepared weights). */
    void project(float[] output, float[] input, int batch);
  }

  /**
   * Copies immutable weights once; callers may subsequently modify or release their input array.
   */
  Projection prepare(float[] weights, int outputs, int inputs);

  /** Computes bidirectional attention with the supplied relative-position indices. */
  void attend(
      float[] output,
      float[] queries,
      float[] keys,
      float[] values,
      float[] relativeQueries,
      float[] relativeKeys,
      int[] relativeIndices,
      int sequenceLength,
      int dimension,
      int heads,
      int relativeRows);
}
