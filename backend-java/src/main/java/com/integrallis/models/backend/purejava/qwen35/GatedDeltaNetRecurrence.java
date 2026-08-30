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

import java.util.Objects;

/** Scalar float32 Gated DeltaNet recurrence used as the pure-Java compatibility baseline. */
final class GatedDeltaNetRecurrence {

  private static final float NORMALIZATION_EPSILON = 1.0e-6f;

  record Result(float[] output, float[] finalState) {}

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
      float[] initialState,
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
        tokenCount,
        headCount,
        headCount,
        keyDimension,
        valueDimension,
        false);
  }

  /** Applies the recurrence while expanding grouped query/key heads to value heads in place. */
  static Result forward(
      float[] query,
      float[] key,
      float[] value,
      float[] logDecay,
      float[] beta,
      float[] initialState,
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
        tokenCount,
        keyHeadCount,
        valueHeadCount,
        keyDimension,
        valueDimension,
        false);
  }

  /** Applies the same recurrence while mutating and retaining a caller-owned state array. */
  static Result forwardInPlace(
      float[] query,
      float[] key,
      float[] value,
      float[] logDecay,
      float[] beta,
      float[] mutableState,
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
        tokenCount,
        headCount,
        headCount,
        keyDimension,
        valueDimension,
        true);
  }

  /** Applies grouped query/key recurrence while retaining a caller-owned state array. */
  static Result forwardInPlace(
      float[] query,
      float[] key,
      float[] value,
      float[] logDecay,
      float[] beta,
      float[] mutableState,
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
        tokenCount,
        keyHeadCount,
        valueHeadCount,
        keyDimension,
        valueDimension,
        true);
  }

  private static Result execute(
      float[] query,
      float[] key,
      float[] value,
      float[] logDecay,
      float[] beta,
      float[] initialState,
      int tokenCount,
      int keyHeadCount,
      int valueHeadCount,
      int keyDimension,
      int valueDimension,
      boolean mutateState) {
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
    requireLength(query, querySize, "query");
    requireLength(key, querySize, "key");
    requireLength(value, valueSize, "value");
    requireLength(logDecay, tokenValueHeads, "logDecay");
    requireLength(beta, tokenValueHeads, "beta");
    if (initialState != null) {
      requireLength(initialState, stateSize, "initialState");
    }

    float[] state =
        mutateState
            ? initialState
            : initialState == null ? new float[stateSize] : initialState.clone();
    float[] output = new float[valueSize];
    float[] normalizedQuery = new float[keyDimension];
    float[] normalizedKey = new float[keyDimension];
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
        int stateOffset = head * keyDimension * valueDimension;
        for (int index = 0; index < keyDimension * valueDimension; index++) {
          state[stateOffset + index] *= decay;
        }

        int valueOffset = tokenHead * valueDimension;
        for (int column = 0; column < valueDimension; column++) {
          float memory = 0.0f;
          for (int row = 0; row < keyDimension; row++) {
            memory += state[stateOffset + row * valueDimension + column] * normalizedKey[row];
          }
          float delta = (value[valueOffset + column] - memory) * beta[tokenHead];
          for (int row = 0; row < keyDimension; row++) {
            state[stateOffset + row * valueDimension + column] += normalizedKey[row] * delta;
          }
        }

        for (int column = 0; column < valueDimension; column++) {
          float result = 0.0f;
          for (int row = 0; row < keyDimension; row++) {
            result += state[stateOffset + row * valueDimension + column] * normalizedQuery[row];
          }
          output[valueOffset + column] = result;
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
}
