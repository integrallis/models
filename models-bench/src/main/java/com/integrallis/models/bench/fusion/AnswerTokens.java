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
package com.integrallis.models.bench.fusion;

import com.integrallis.models.api.Tokenizer;

/** Maps a character span of decoded output back to the generated tokens that produced it. */
final class AnswerTokens {

  private AnswerTokens() {}

  /**
   * Returns {@code [first, endExclusive)} token indices covering {@code [start, end)} of {@code
   * tokenizer.decode(tokens)}, using prefix decoding so multi-byte merges are handled.
   */
  static int[] span(Tokenizer tokenizer, int[] tokens, int start, int end) {
    if (start < 0 || end <= start) {
      return new int[] {0, 0};
    }
    int first = firstPrefixLongerThan(tokenizer, tokens, start);
    int last = firstPrefixLongerThan(tokenizer, tokens, end - 1);
    return new int[] {Math.max(0, first - 1), Math.max(first, last)};
  }

  /** Smallest n in [1, tokens.length] with decode(tokens[0..n]).length() > target. */
  private static int firstPrefixLongerThan(Tokenizer tokenizer, int[] tokens, int target) {
    int low = 1;
    int high = tokens.length;
    while (low < high) {
      int mid = (low + high) >>> 1;
      if (prefixLength(tokenizer, tokens, mid) > target) {
        high = mid;
      } else {
        low = mid + 1;
      }
    }
    return low;
  }

  private static int prefixLength(Tokenizer tokenizer, int[] tokens, int count) {
    int[] prefix = new int[count];
    System.arraycopy(tokens, 0, prefix, 0, count);
    return tokenizer.decode(prefix).length();
  }

  /** Mean of {@code values[first..endExclusive)}, or NaN when the span is empty. */
  static double mean(double[] values, int[] span) {
    int first = span[0];
    int end = Math.min(span[1], values.length);
    if (end <= first) {
      return Double.NaN;
    }
    double sum = 0;
    for (int i = first; i < end; i++) {
      sum += values[i];
    }
    return sum / (end - first);
  }
}
