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
package com.integrallis.models.accelerator;

import com.integrallis.models.backend.purejava.spi.BatchedCausalAttentionKernel;
import java.util.LinkedHashMap;
import java.util.Map;

/** Experimental Java/Tornado batched attention provider with one retained KV cache per layer. */
public final class TornadoBatchedCausalAttentionKernel implements BatchedCausalAttentionKernel {
  private static final int DEFAULT_EXECUTION_BATCH_SIZE = 32;
  private static final int MINIMUM_BATCH_SIZE = 4;

  private final int executionBatchSize;
  private final Map<AttentionKey, TornadoCausalAttentionPlan> plans = new LinkedHashMap<>();
  private final Map<Integer, Integer> nextPositions = new LinkedHashMap<>();
  private long calls;
  private long totalNanos;
  private boolean closed;

  public TornadoBatchedCausalAttentionKernel() {
    this(DEFAULT_EXECUTION_BATCH_SIZE);
  }

  public TornadoBatchedCausalAttentionKernel(int executionBatchSize) {
    if (executionBatchSize < MINIMUM_BATCH_SIZE) {
      throw new IllegalArgumentException(
          "executionBatchSize must be at least " + MINIMUM_BATCH_SIZE);
    }
    this.executionBatchSize = executionBatchSize;
  }

  @Override
  public synchronized boolean isEligible(
      int layer,
      int startPosition,
      int batchSize,
      int numHeads,
      int numKvHeads,
      int keyLength,
      int valueLength,
      int maxSequenceLength,
      int slidingWindow) {
    return !closed
        && layer >= 0
        && startPosition >= 0
        && batchSize >= MINIMUM_BATCH_SIZE
        && batchSize <= executionBatchSize
        && numHeads > 0
        && numKvHeads > 0
        && numHeads % numKvHeads == 0
        && keyLength > 0
        && valueLength == keyLength
        && maxSequenceLength >= startPosition + batchSize
        && slidingWindow == 0
        && nextPositions.getOrDefault(layer, 0) == startPosition;
  }

  @Override
  public synchronized void attend(
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
    if (!isEligible(
        layer,
        startPosition,
        batchSize,
        numHeads,
        numKvHeads,
        keyLength,
        valueLength,
        maxSequenceLength,
        slidingWindow)) {
      throw new UnsupportedOperationException(
          "attention chunk is not eligible for the Tornado experiment");
    }
    AttentionKey planKey =
        new AttentionKey(
            layer, numHeads, numKvHeads, keyLength, valueLength, maxSequenceLength, slidingWindow);
    TornadoCausalAttentionPlan plan =
        plans.computeIfAbsent(
            planKey,
            ignored ->
                new TornadoCausalAttentionPlan(
                    "attention-layer-" + layer,
                    executionBatchSize,
                    numHeads,
                    numKvHeads,
                    keyLength,
                    valueLength,
                    maxSequenceLength,
                    slidingWindow));
    long started = System.nanoTime();
    plan.execute(query, key, value, output, batchSize, startPosition);
    totalNanos += System.nanoTime() - started;
    calls++;
    nextPositions.put(layer, startPosition + batchSize);
  }

  public synchronized int planCount() {
    return plans.size();
  }

  public synchronized long calls() {
    return calls;
  }

  public synchronized double totalMillis() {
    return totalNanos / 1_000_000.0;
  }

  @Override
  public synchronized void rewind(int checkpoint) {
    nextPositions.replaceAll((ignored, nextPosition) -> Math.min(nextPosition, checkpoint));
  }

  @Override
  public synchronized void reset() {
    nextPositions.clear();
  }

  @Override
  public synchronized void close() {
    if (closed) {
      return;
    }
    RuntimeException failure = null;
    for (TornadoCausalAttentionPlan plan : plans.values()) {
      try {
        plan.close();
      } catch (RuntimeException exception) {
        if (failure == null) {
          failure = exception;
        } else {
          failure.addSuppressed(exception);
        }
      }
    }
    plans.clear();
    nextPositions.clear();
    closed = true;
    if (failure != null) {
      throw failure;
    }
  }

  private record AttentionKey(
      int layer,
      int numHeads,
      int numKvHeads,
      int keyLength,
      int valueLength,
      int maxSequenceLength,
      int slidingWindow) {}
}
