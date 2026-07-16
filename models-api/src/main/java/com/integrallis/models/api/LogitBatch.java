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
package com.integrallis.models.api;

import java.util.Arrays;
import java.util.Objects;

/**
 * Logits for a contiguous token batch in token-major order.
 *
 * <p>The constructor takes ownership of the active prefix of {@code values}; callers must not
 * modify that array after construction unless the containing backend documents the result as
 * transient.
 */
public final class LogitBatch {

  private final int tokenCount;
  private final int vocabularySize;
  private final float[] values;

  public LogitBatch(int tokenCount, int vocabularySize, float[] values) {
    if (tokenCount <= 0) {
      throw new IllegalArgumentException("tokenCount must be > 0");
    }
    if (vocabularySize <= 0) {
      throw new IllegalArgumentException("vocabularySize must be > 0");
    }
    this.values = Objects.requireNonNull(values, "values");
    int expectedLength = Math.multiplyExact(tokenCount, vocabularySize);
    if (values.length < expectedLength) {
      throw new IllegalArgumentException(
          "values.length must be at least "
              + expectedLength
              + " for "
              + tokenCount
              + " tokens and vocabulary size "
              + vocabularySize
              + ": "
              + values.length);
    }
    this.tokenCount = tokenCount;
    this.vocabularySize = vocabularySize;
  }

  public int tokenCount() {
    return tokenCount;
  }

  public int vocabularySize() {
    return vocabularySize;
  }

  public float logit(int tokenIndex, int vocabularyIndex) {
    checkTokenIndex(tokenIndex);
    if (vocabularyIndex < 0 || vocabularyIndex >= vocabularySize) {
      throw new IndexOutOfBoundsException("vocabularyIndex out of range: " + vocabularyIndex);
    }
    return values[tokenIndex * vocabularySize + vocabularyIndex];
  }

  /** Returns the first vocabulary index containing the largest logit in a token row. */
  public int argmax(int tokenIndex) {
    checkTokenIndex(tokenIndex);
    int offset = tokenIndex * vocabularySize;
    int best = 0;
    float bestValue = values[offset];
    for (int index = 1; index < vocabularySize; index++) {
      float candidate = values[offset + index];
      if (candidate > bestValue) {
        best = index;
        bestValue = candidate;
      }
    }
    return best;
  }

  /** Returns a stable copy of one token's vocabulary logits. */
  public float[] copyRow(int tokenIndex) {
    checkTokenIndex(tokenIndex);
    int offset = tokenIndex * vocabularySize;
    return Arrays.copyOfRange(values, offset, offset + vocabularySize);
  }

  /** Returns a stable copy of all active rows. */
  public LogitBatch snapshot() {
    return new LogitBatch(
        tokenCount, vocabularySize, Arrays.copyOf(values, tokenCount * vocabularySize));
  }

  private void checkTokenIndex(int tokenIndex) {
    if (tokenIndex < 0 || tokenIndex >= tokenCount) {
      throw new IndexOutOfBoundsException("tokenIndex out of range: " + tokenIndex);
    }
  }
}
