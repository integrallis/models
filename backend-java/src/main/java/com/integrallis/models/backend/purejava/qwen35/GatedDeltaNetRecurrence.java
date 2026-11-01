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

import java.util.Arrays;
import java.util.Objects;

/** Retained-FP64 Gated DeltaNet state with unchanged FP32 normalization and decay. */
final class GatedDeltaNetRecurrence {

  private static final float NORMALIZATION_EPSILON = 1.0e-6f;

  record Result(float[] output, double[] finalState) {}

  private GatedDeltaNetRecurrence() {}

  /**
   * Applies the recurrent gated delta rule.
   *
   * <p>Query and key are laid out as {@code [token][head][keyDimension]}; value and output as
   * {@code [token][head][valueDimension]}; gates as {@code [token][head]}; and state as {@code
   * [head][keyDimension][valueDimension]}. When there are fewer key heads, value-head tensors use
   * the tiled order emitted by the GGUF Qwen3.5 conversion ({@code K0,K1,...,K0,K1,...}).
   *
   * <p>The input state is copied and never mutated. A {@code null} state starts a cold sequence.
   */
  static Result forward(
      float[] query,
      float[] key,
      float[] value,
      float[] logDecay,
      float[] beta,
      double[] initialState,
      int tokenCount,
      int headCount,
      int keyDimension,
      int valueDimension) {
    return execute(
        query,
        key,
        value,
        logDecay,
        beta,
        initialState,
        0,
        false,
        tokenCount,
        headCount,
        headCount,
        keyDimension,
        valueDimension,
        false,
        null,
        null,
        null,
        null,
        null,
        false);
  }

  /** Applies the recurrence while expanding grouped query/key heads to value heads in place. */
  static Result forward(
      float[] query,
      float[] key,
      float[] value,
      float[] logDecay,
      float[] beta,
      double[] initialState,
      int tokenCount,
      int keyHeadCount,
      int valueHeadCount,
      int keyDimension,
      int valueDimension) {
    return execute(
        query,
        key,
        value,
        logDecay,
        beta,
        initialState,
        0,
        false,
        tokenCount,
        keyHeadCount,
        valueHeadCount,
        keyDimension,
        valueDimension,
        false,
        null,
        null,
        null,
        null,
        null,
        false);
  }

  /** Applies the same recurrence while mutating and retaining a caller-owned state array. */
  static Result forwardInPlace(
      float[] query,
      float[] key,
      float[] value,
      float[] logDecay,
      float[] beta,
      double[] mutableState,
      int tokenCount,
      int headCount,
      int keyDimension,
      int valueDimension) {
    Objects.requireNonNull(mutableState, "mutableState");
    return execute(
        query,
        key,
        value,
        logDecay,
        beta,
        mutableState,
        0,
        false,
        tokenCount,
        headCount,
        headCount,
        keyDimension,
        valueDimension,
        true,
        null,
        null,
        null,
        null,
        null,
        false);
  }

  /** Applies grouped query/key recurrence while retaining a caller-owned state array. */
  static Result forwardInPlace(
      float[] query,
      float[] key,
      float[] value,
      float[] logDecay,
      float[] beta,
      double[] mutableState,
      int tokenCount,
      int keyHeadCount,
      int valueHeadCount,
      int keyDimension,
      int valueDimension) {
    Objects.requireNonNull(mutableState, "mutableState");
    return execute(
        query,
        key,
        value,
        logDecay,
        beta,
        mutableState,
        0,
        false,
        tokenCount,
        keyHeadCount,
        valueHeadCount,
        keyDimension,
        valueDimension,
        true,
        null,
        null,
        null,
        null,
        null,
        false);
  }

  /** Applies grouped recurrence into caller-owned state and workspace arrays. */
  static void forwardInPlace(
      float[] query,
      float[] key,
      float[] value,
      float[] logDecay,
      float[] beta,
      double[] mutableState,
      float[] output,
      float[] normalizedQuery,
      float[] normalizedKey,
      double[] memory,
      double[] delta,
      int tokenCount,
      int keyHeadCount,
      int valueHeadCount,
      int keyDimension,
      int valueDimension) {
    Objects.requireNonNull(mutableState, "mutableState");
    execute(
        query,
        key,
        value,
        logDecay,
        beta,
        mutableState,
        0,
        false,
        tokenCount,
        keyHeadCount,
        valueHeadCount,
        keyDimension,
        valueDimension,
        true,
        Objects.requireNonNull(output, "output"),
        Objects.requireNonNull(normalizedQuery, "normalizedQuery"),
        Objects.requireNonNull(normalizedKey, "normalizedKey"),
        Objects.requireNonNull(memory, "memory"),
        Objects.requireNonNull(delta, "delta"),
        false);
  }

  /** Applies a token prefix from capacity-sized caller-owned batch buffers. */
  static void forwardPrefixInPlace(
      float[] query,
      float[] key,
      float[] value,
      float[] logDecay,
      float[] beta,
      double[] mutableState,
      float[] output,
      float[] normalizedQuery,
      float[] normalizedKey,
      double[] memory,
      double[] delta,
      int tokenCount,
      int keyHeadCount,
      int valueHeadCount,
      int keyDimension,
      int valueDimension) {
    Objects.requireNonNull(mutableState, "mutableState");
    execute(
        query,
        key,
        value,
        logDecay,
        beta,
        mutableState,
        0,
        false,
        tokenCount,
        keyHeadCount,
        valueHeadCount,
        keyDimension,
        valueDimension,
        true,
        Objects.requireNonNull(output, "output"),
        Objects.requireNonNull(normalizedQuery, "normalizedQuery"),
        Objects.requireNonNull(normalizedKey, "normalizedKey"),
        Objects.requireNonNull(memory, "memory"),
        Objects.requireNonNull(delta, "delta"),
        true);
  }

  /**
   * Advances one branch of a grouped decision, whose state is one slot of a shared array.
   *
   * <p>The group holds every branch's recurrent state for a layer in one branch-major array and
   * this recurrence advances exactly the selected slot without touching other branches.
   */
  static void forwardSlotInPlace(
      float[] query,
      float[] key,
      float[] value,
      float[] logDecay,
      float[] beta,
      double[] groupState,
      int stateBase,
      float[] output,
      float[] normalizedQuery,
      float[] normalizedKey,
      double[] memory,
      double[] delta,
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
        Objects.requireNonNull(groupState, "groupState"),
        stateBase,
        true,
        1,
        keyHeadCount,
        valueHeadCount,
        keyDimension,
        valueDimension,
        true,
        Objects.requireNonNull(output, "output"),
        Objects.requireNonNull(normalizedQuery, "normalizedQuery"),
        Objects.requireNonNull(normalizedKey, "normalizedKey"),
        Objects.requireNonNull(memory, "memory"),
        Objects.requireNonNull(delta, "delta"),
        true);
  }

  private static Result execute(
      float[] query,
      float[] key,
      float[] value,
      float[] logDecay,
      float[] beta,
      double[] initialState,
      int stateBase,
      boolean windowedState,
      int tokenCount,
      int keyHeadCount,
      int valueHeadCount,
      int keyDimension,
      int valueDimension,
      boolean mutateState,
      float[] outputWorkspace,
      float[] normalizedQueryWorkspace,
      float[] normalizedKeyWorkspace,
      double[] memoryWorkspace,
      double[] deltaWorkspace,
      boolean capacitySizedBuffers) {
    Objects.requireNonNull(query, "query");
    Objects.requireNonNull(key, "key");
    Objects.requireNonNull(value, "value");
    Objects.requireNonNull(logDecay, "logDecay");
    Objects.requireNonNull(beta, "beta");
    requirePositive(tokenCount, "tokenCount");
    requirePositive(keyHeadCount, "keyHeadCount");
    requirePositive(valueHeadCount, "valueHeadCount");
    requirePositive(keyDimension, "keyDimension");
    requirePositive(valueDimension, "valueDimension");
    if (valueHeadCount % keyHeadCount != 0) {
      throw new IllegalArgumentException("valueHeadCount must be divisible by keyHeadCount");
    }

    int tokenKeyHeads = Math.multiplyExact(tokenCount, keyHeadCount);
    int tokenValueHeads = Math.multiplyExact(tokenCount, valueHeadCount);
    int querySize = Math.multiplyExact(tokenKeyHeads, keyDimension);
    int valueSize = Math.multiplyExact(tokenValueHeads, valueDimension);
    int stateSize =
        Math.multiplyExact(Math.multiplyExact(valueHeadCount, keyDimension), valueDimension);
    requireLength(query, querySize, "query", capacitySizedBuffers);
    requireLength(key, querySize, "key", capacitySizedBuffers);
    requireLength(value, valueSize, "value", capacitySizedBuffers);
    requireLength(logDecay, tokenValueHeads, "logDecay", capacitySizedBuffers);
    requireLength(beta, tokenValueHeads, "beta", capacitySizedBuffers);
    if (initialState != null) {
      if (!windowedState) {
        requireLength(initialState, stateSize, "initialState");
      } else if (stateBase < 0 || initialState.length - stateSize < stateBase) {
        throw new IllegalArgumentException(
            "initialState of "
                + initialState.length
                + " cannot hold "
                + stateSize
                + " doubles at "
                + stateBase);
      }
    }
    if (windowedState && !mutateState) {
      throw new IllegalArgumentException(
          "a state window is only meaningful when mutating in place");
    }

    double[] state =
        mutateState
            ? initialState
            : initialState == null ? new double[stateSize] : initialState.clone();
    float[] output = workspace(outputWorkspace, valueSize, "output", capacitySizedBuffers);
    float[] normalizedQuery = workspace(normalizedQueryWorkspace, keyDimension, "normalizedQuery");
    float[] normalizedKey = workspace(normalizedKeyWorkspace, keyDimension, "normalizedKey");
    double[] memory = workspace(memoryWorkspace, valueDimension, "memory");
    double[] delta = workspace(deltaWorkspace, valueDimension, "delta");
    float queryScale = (float) (1.0 / Math.sqrt(keyDimension));
    for (int token = 0; token < tokenCount; token++) {
      for (int head = 0; head < valueHeadCount; head++) {
        int tokenHead = token * valueHeadCount + head;
        int keyHead = head % keyHeadCount;
        int queryOffset = (token * keyHeadCount + keyHead) * keyDimension;
        normalize(query, queryOffset, normalizedQuery);
        normalize(key, queryOffset, normalizedKey);
        for (int column = 0; column < keyDimension; column++) {
          normalizedQuery[column] *= queryScale;
        }

        float decay = (float) Math.exp(logDecay[tokenHead]);
        int stateOffset = stateBase + head * keyDimension * valueDimension;
        for (int index = 0; index < keyDimension * valueDimension; index++) {
          state[stateOffset + index] *= decay;
        }

        int valueOffset = tokenHead * valueDimension;
        Arrays.fill(memory, 0.0);
        for (int row = 0; row < keyDimension; row++) {
          int rowOffset = stateOffset + row * valueDimension;
          double keyValue = normalizedKey[row];
          for (int column = 0; column < valueDimension; column++) {
            memory[column] = Math.fma(state[rowOffset + column], keyValue, memory[column]);
          }
        }
        for (int column = 0; column < valueDimension; column++) {
          delta[column] =
              ((double) value[valueOffset + column] - memory[column]) * (double) beta[tokenHead];
        }
        for (int row = 0; row < keyDimension; row++) {
          int rowOffset = stateOffset + row * valueDimension;
          double keyValue = normalizedKey[row];
          for (int column = 0; column < valueDimension; column++) {
            state[rowOffset + column] =
                Math.fma(delta[column], keyValue, state[rowOffset + column]);
          }
        }
        // Reuse the memory workspace: only the completed graph output is rounded to float.
        Arrays.fill(memory, 0.0);
        for (int row = 0; row < keyDimension; row++) {
          int rowOffset = stateOffset + row * valueDimension;
          double queryValue = normalizedQuery[row];
          for (int column = 0; column < valueDimension; column++) {
            memory[column] = Math.fma(state[rowOffset + column], queryValue, memory[column]);
          }
        }
        for (int column = 0; column < valueDimension; column++) {
          output[valueOffset + column] = (float) memory[column];
        }
      }
    }
    return new Result(output, state);
  }

  private static void normalize(float[] source, int offset, float[] destination) {
    float squaredNorm = 0.0f;
    for (int index = 0; index < destination.length; index++) {
      float value = source[offset + index];
      squaredNorm += value * value;
    }
    float inverseNorm = (float) (1.0 / Math.sqrt(squaredNorm + NORMALIZATION_EPSILON));
    for (int index = 0; index < destination.length; index++) {
      destination[index] = source[offset + index] * inverseNorm;
    }
  }

  private static void requirePositive(int value, String name) {
    if (value <= 0) {
      throw new IllegalArgumentException(name + " must be > 0: " + value);
    }
  }

  private static void requireLength(float[] values, int expected, String name) {
    if (values.length != expected) {
      throw new IllegalArgumentException(
          name + " length must be " + expected + ": " + values.length);
    }
  }

  private static void requireLength(
      float[] values, int expected, String name, boolean capacitySized) {
    if (!capacitySized) {
      requireLength(values, expected, name);
    } else if (values.length < expected) {
      throw new IllegalArgumentException(
          name + " length must be at least " + expected + ": " + values.length);
    }
  }

  private static float[] workspace(float[] values, int expected, String name) {
    if (values == null) {
      return new float[expected];
    }
    requireLength(values, expected, name);
    return values;
  }

  private static float[] workspace(
      float[] values, int expected, String name, boolean capacitySized) {
    if (values == null) {
      return new float[expected];
    }
    requireLength(values, expected, name, capacitySized);
    return values;
  }

  private static void requireLength(double[] values, int expected, String name) {
    if (values.length != expected) {
      throw new IllegalArgumentException(
          name + " length must be " + expected + ": " + values.length);
    }
  }

  private static double[] workspace(double[] values, int expected, String name) {
    if (values == null) return new double[expected];
    requireLength(values, expected, name);
    return values;
  }
}
