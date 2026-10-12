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
package com.integrallis.models.backend.purejava.qwen35;

import com.integrallis.vectors.core.GgufStagePlan;
import java.util.Objects;

/** Exact row-order recurrence, partitioned across independent heads by the Vectors executor. */
final class GatedDeltaNetRecurrence {
  private GatedDeltaNetRecurrence() {}

  /** Advances a prefix from capacity-sized buffers, retaining caller-owned state and output. */
  static void forwardPrefixInPlace(
      float[] query,
      float[] key,
      float[] value,
      float[] logDecay,
      float[] beta,
      float[] state,
      float[] output,
      int tokenCount,
      int keyHeadCount,
      int valueHeadCount,
      int keyDimension,
      int valueDimension) {
    execute(
        query,
        key,
        value,
        logDecay,
        beta,
        state,
        0,
        output,
        tokenCount,
        keyHeadCount,
        valueHeadCount,
        keyDimension,
        valueDimension);
  }

  /** Advances one live branch into its own slot of a grouped decision's recurrent state. */
  static void forwardSlotInPlace(
      float[] query,
      float[] key,
      float[] value,
      float[] logDecay,
      float[] beta,
      float[] state,
      int stateBase,
      float[] output,
      int keyHeadCount,
      int valueHeadCount,
      int keyDimension,
      int valueDimension) {
    execute(
        query,
        key,
        value,
        logDecay,
        beta,
        state,
        stateBase,
        output,
        1,
        keyHeadCount,
        valueHeadCount,
        keyDimension,
        valueDimension);
  }

  private static void execute(
      float[] query,
      float[] key,
      float[] value,
      float[] logDecay,
      float[] beta,
      float[] state,
      int stateBase,
      float[] output,
      int tokenCount,
      int keyHeadCount,
      int valueHeadCount,
      int keyDimension,
      int valueDimension) {
    requirePositive(tokenCount, "tokenCount");
    requirePositive(keyHeadCount, "keyHeadCount");
    requirePositive(valueHeadCount, "valueHeadCount");
    requirePositive(keyDimension, "keyDimension");
    requirePositive(valueDimension, "valueDimension");
    if (valueHeadCount % keyHeadCount != 0)
      throw new IllegalArgumentException("valueHeadCount must be divisible by keyHeadCount");
    int querySize = Math.multiplyExact(Math.multiplyExact(tokenCount, keyHeadCount), keyDimension);
    int valueSize =
        Math.multiplyExact(Math.multiplyExact(tokenCount, valueHeadCount), valueDimension);
    int stateSize =
        Math.multiplyExact(Math.multiplyExact(valueHeadCount, keyDimension), valueDimension);
    int gates = Math.multiplyExact(tokenCount, valueHeadCount);
    requireCapacity(query, querySize, "query");
    requireCapacity(key, querySize, "key");
    requireCapacity(value, valueSize, "value");
    requireCapacity(logDecay, gates, "logDecay");
    requireCapacity(beta, gates, "beta");
    requireCapacity(output, valueSize, "output");
    Objects.requireNonNull(state, "state");
    if (stateBase < 0 || stateBase > state.length - stateSize)
      throw new IllegalArgumentException("recurrent state cannot hold the requested slot");
    if (state == output
        || state == query
        || state == key
        || state == value
        || state == logDecay
        || state == beta
        || output == query
        || output == key
        || output == value
        || output == logDecay
        || output == beta)
      throw new IllegalArgumentException(
          "recurrent state and output must not alias any other buffer");
    GgufStagePlan.of(
            GgufStagePlan.stage(
                valueHeadCount,
                (firstHead, endHead) ->
                    GatedDeltaNetKernel.apply(
                        query,
                        key,
                        value,
                        logDecay,
                        beta,
                        state,
                        output,
                        stateBase,
                        tokenCount,
                        keyHeadCount,
                        valueHeadCount,
                        keyDimension,
                        valueDimension,
                        firstHead,
                        endHead)))
        .execute();
  }

  private static void requirePositive(int value, String name) {
    if (value <= 0) throw new IllegalArgumentException(name + " must be > 0: " + value);
  }

  private static void requireCapacity(float[] values, int size, String name) {
    Objects.requireNonNull(values, name);
    if (values.length < size)
      throw new IllegalArgumentException(
          name + " length must be at least " + size + ": " + values.length);
  }
}
