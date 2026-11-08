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

/** Optional in-process implementation of batched causal attention with retained sequence state. */
public interface BatchedCausalAttentionKernel extends AutoCloseable {

  /** Whether ordinary RoPE angles use successive float multiplications in this numerical path. */
  default boolean usesFloatRotaryRecurrence() {
    return false;
  }

  /** Returns whether this prompt chunk can run without breaking retained cache continuity. */
  boolean isEligible(
      int layer,
      int startPosition,
      int batchSize,
      int numHeads,
      int numKvHeads,
      int keyLength,
      int valueLength,
      int maxSequenceLength,
      int slidingWindow);

  /** Stores the supplied K/V rows and computes batch-major grouped-query attention. */
  void attend(
      float[] output,
      float[] query,
      float[] key,
      float[] value,
      int layer,
      int startPosition,
      int batchSize,
      int numHeads,
      int numKvHeads,
      int keyLength,
      int valueLength,
      int maxSequenceLength,
      int slidingWindow);

  /** Computes the last retained prefill query without changing the original batch arithmetic. */
  default void attendFinalPrefill(
      float[] output,
      float[] query,
      float[] key,
      float[] value,
      int layer,
      int position,
      int originalBatchSize,
      int numHeads,
      int numKvHeads,
      int keyLength,
      int valueLength,
      int maxSequenceLength,
      int slidingWindow) {
    attend(
        output,
        query,
        key,
        value,
        layer,
        position,
        1,
        numHeads,
        numKvHeads,
        keyLength,
        valueLength,
        maxSequenceLength,
        slidingWindow);
  }

  /**
   * Retains K/V rows when prefill prunes unused queries from the final layer. Returning false must
   * leave provider state unchanged; the caller's ordinary KV cache remains authoritative.
   */
  default boolean append(
      float[] key,
      float[] value,
      int layer,
      int startPosition,
      int batchSize,
      int numHeads,
      int numKvHeads,
      int keyLength,
      int valueLength,
      int maxSequenceLength,
      int slidingWindow) {
    return false;
  }

  /** Notifies retained device state that speculative positions were discarded. */
  default void rewind(int checkpoint) {}

  /** Clears sequence continuity before a new conversation. */
  default void reset() {}

  @Override
  default void close() {}

  /** Returns the no-op implementation used by the ordinary Java backend. */
  static BatchedCausalAttentionKernel none() {
    return NoKernel.INSTANCE;
  }

  enum NoKernel implements BatchedCausalAttentionKernel {
    INSTANCE;

    @Override
    public boolean isEligible(
        int layer,
        int startPosition,
        int batchSize,
        int numHeads,
        int numKvHeads,
        int keyLength,
        int valueLength,
        int maxSequenceLength,
        int slidingWindow) {
      return false;
    }

    @Override
    public void attend(
        float[] output,
        float[] query,
        float[] key,
        float[] value,
        int layer,
        int startPosition,
        int batchSize,
        int numHeads,
        int numKvHeads,
        int keyLength,
        int valueLength,
        int maxSequenceLength,
        int slidingWindow) {
      throw new UnsupportedOperationException("no batched causal-attention kernel is configured");
    }
  }
}
