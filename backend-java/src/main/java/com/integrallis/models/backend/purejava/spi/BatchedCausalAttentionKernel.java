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

import java.util.Objects;

/** Optional in-process implementation of batched causal attention with retained sequence state. */
public interface BatchedCausalAttentionKernel extends AutoCloseable {

  /**
   * Which sequence the next eligibility check and attend call belong to, and the arithmetic the
   * Java path would apply to them.
   *
   * <p>The forward pass owns several divergent sequences at once: independent sessions, and
   * branches forked from one physically shared KV prefix. An implementation that retains any state
   * across calls — a device-resident mirror of the cache, for instance — cannot tell those apart
   * from the per-call attention arguments alone, because a fork and its source report the same
   * layer, position, and shape. {@code sequenceId} distinguishes them.
   *
   * @param sequenceId stable identity of the KV cache being extended; never zero
   * @param sharedPrefixLength positions this sequence reads from immutable shared storage
   * @param attentionScale the multiplier the Java path applies to every query/key score
   * @param fusedGroupedArithmetic whether the Java path scores through the fused grouped kernel
   *     rather than the head-by-head loop, which changes the reference numerics
   */
  record AttentionScope(
      long sequenceId,
      int sharedPrefixLength,
      float attentionScale,
      boolean fusedGroupedArithmetic) {

    public AttentionScope {
      if (sequenceId == 0L) {
        throw new IllegalArgumentException("sequenceId must be non-zero");
      }
      if (sharedPrefixLength < 0) {
        throw new IllegalArgumentException("sharedPrefixLength must be >= 0");
      }
      if (!Float.isFinite(attentionScale)) {
        throw new IllegalArgumentException("attentionScale must be finite: " + attentionScale);
      }
    }
  }

  /**
   * Binds the following eligibility check and attend call to one sequence.
   *
   * <p>Implementations that hold no cross-call state ignore this. Implementations that do must
   * refuse, or rebuild, when the scope names a sequence other than the one they retained.
   */
  default void selectScope(AttentionScope scope) {
    Objects.requireNonNull(scope, "scope");
  }

  /**
   * Returns the first position of {@code layer} this implementation has not yet mirrored for the
   * scope last selected, or {@code -1} when it retains no state and needs no mirroring.
   *
   * <p>A device-resident implementation returns the length of the window it already holds. The
   * caller then offers the missing rows through {@link #mirrorSpan} before attending, so a decode
   * step that follows a prefill computed on the Java path still sees the whole history.
   */
  default int mirroredPosition(int layer) {
    return -1;
  }

  /**
   * Offers one contiguous run of already-cached key and value rows so a retained mirror can be
   * brought up to date without recomputing them.
   *
   * <p>Rows are read-only and are not copied by the caller: row {@code r} of this run is the key at
   * {@code keys[keyOffset + r * keyRowStride]} and the value at {@code values[valueOffset + r *
   * valueRowStride]}, both of the full per-position cache width.
   */
  default void mirrorSpan(
      int layer,
      int firstPosition,
      int positionCount,
      float[] keys,
      int keyOffset,
      int keyRowStride,
      float[] values,
      int valueOffset,
      int valueRowStride) {
    throw new UnsupportedOperationException("kernel does not retain a mirrored KV window");
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

  /**
   * Why the most recent {@link #isEligible} call refused, for routing observability.
   *
   * <p>A kernel that is wired but never runs and a kernel that runs and does not help are
   * indistinguishable from end-to-end timings alone. This exists so the first case reports itself
   * rather than being read as the second.
   */
  default String lastRefusal() {
    return "no batched causal-attention kernel is configured";
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
