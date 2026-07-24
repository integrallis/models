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

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import java.lang.foreign.MemorySegment;
import java.util.Map;

/** Optional batched GGUF projection implementation injected into the Java transformer pipeline. */
public interface GgufBatchedMatrixKernel extends AutoCloseable {

  /** Stable implementation identifier used in diagnostics. */
  default String implementation() {
    return "custom";
  }

  /** Load-time planner recommendations needed to route eligible projections through this kernel. */
  default Map<String, String> planRecommendations() {
    return Map.of();
  }

  /** Returns whether this implementation supports the given GGUF weight format. */
  boolean supports(GgufTensorType type);

  /**
   * Returns whether this implementation should execute a particular projection shape.
   *
   * <p>Implementations may reject supported formats when boundary or scheduling overhead would make
   * a specific shape slower than the Java kernel.
   */
  default boolean isEligible(GgufTensorType type, int batchSize, int rows, int cols) {
    return supports(type);
  }

  /** Returns whether two projections can share one native activation preparation. */
  default boolean isDualEligible(
      GgufTensorType firstType,
      int firstRows,
      GgufTensorType secondType,
      int secondRows,
      int batchSize,
      int cols) {
    return false;
  }

  /** Computes two projections from the same batch-major input. */
  default void multiplyDual(
      float[] firstOutput,
      MemorySegment firstWeights,
      GgufTensorType firstType,
      int firstRows,
      float[] secondOutput,
      MemorySegment secondWeights,
      GgufTensorType secondType,
      int secondRows,
      float[] input,
      int batchSize,
      int cols) {
    throw new UnsupportedOperationException("injected kernel does not support dual projections");
  }

  /** Returns whether three projections can share one native activation preparation. */
  default boolean isTripleEligible(
      GgufTensorType firstType,
      int firstRows,
      GgufTensorType secondType,
      int secondRows,
      GgufTensorType thirdType,
      int thirdRows,
      int batchSize,
      int cols) {
    return false;
  }

  /** Computes three projections from the same batch-major input. */
  default void multiplyTriple(
      float[] firstOutput,
      MemorySegment firstWeights,
      GgufTensorType firstType,
      int firstRows,
      float[] secondOutput,
      MemorySegment secondWeights,
      GgufTensorType secondType,
      int secondRows,
      float[] thirdOutput,
      MemorySegment thirdWeights,
      GgufTensorType thirdType,
      int thirdRows,
      float[] input,
      int batchSize,
      int cols) {
    throw new UnsupportedOperationException("injected kernel does not support triple projections");
  }

  /** Computes batch-major matrix multiplication for a supported mapped GGUF tensor. */
  void multiply(
      float[] output,
      float[] input,
      MemorySegment weights,
      GgufTensorType type,
      int batchSize,
      int rows,
      int cols);

  @Override
  default void close() {}

  /** Returns the no-op provider used by the pure-Java backend. */
  static GgufBatchedMatrixKernel none() {
    return NoKernel.INSTANCE;
  }

  enum NoKernel implements GgufBatchedMatrixKernel {
    INSTANCE;

    @Override
    public String implementation() {
      return "vector-api";
    }

    @Override
    public boolean supports(GgufTensorType type) {
      return false;
    }

    @Override
    public void multiply(
        float[] output,
        float[] input,
        MemorySegment weights,
        GgufTensorType type,
        int batchSize,
        int rows,
        int cols) {
      throw new UnsupportedOperationException("no injected GGUF batched matrix kernel");
    }
  }
}
