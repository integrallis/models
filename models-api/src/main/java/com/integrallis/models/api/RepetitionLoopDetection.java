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

/**
 * Configuration for the runtime's repetition-loop detector.
 *
 * <p>When enabled, generation stops with {@link StopReason#REPETITION_LOOP} as soon as the
 * generated tokens (the prompt is not considered) end in an exactly periodic run whose period is at
 * most {@code maxSpan} tokens, which repeats at least {@code minRepeats} times, and which covers at
 * least {@code minLoopTokens} tokens. Detection costs at most {@code maxSpan} integer comparisons
 * per generated token regardless of generation length.
 *
 * <p>Exact periodicity is deliberately narrow: a numbered list whose ordinals change is never a
 * loop, but legitimate exact repetition (a zero-filled array literal, a table rule) is one once it
 * crosses the thresholds. {@code minLoopTokens} keeps short spans such as repeated newlines from
 * firing early without making long-span loops wait longer. Detection is off by default.
 *
 * @param maxSpan longest repeating span, in tokens, that is checked; zero disables detection
 * @param minRepeats consecutive copies of the span required, at least 2 when enabled
 * @param minLoopTokens minimum total tokens the repeating run must cover, zero for no minimum
 */
public record RepetitionLoopDetection(int maxSpan, int minRepeats, int minLoopTokens) {

  private static final RepetitionLoopDetection DISABLED = new RepetitionLoopDetection(0, 0, 0);

  public RepetitionLoopDetection {
    if (maxSpan < 0 || minRepeats < 0 || minLoopTokens < 0) {
      throw new IllegalArgumentException(
          "repetition-loop detection values must not be negative, got maxSpan="
              + maxSpan
              + ", minRepeats="
              + minRepeats
              + ", minLoopTokens="
              + minLoopTokens);
    }
    if (maxSpan > 0 && minRepeats < 2) {
      throw new IllegalArgumentException(
          "repetition-loop minRepeats must be >= 2 when detection is enabled, got: " + minRepeats);
    }
  }

  /**
   * Returns the default configuration, which never detects a loop.
   *
   * @return disabled detection
   */
  public static RepetitionLoopDetection disabled() {
    return DISABLED;
  }

  /**
   * Returns whether detection runs.
   *
   * @return {@code true} when {@code maxSpan > 0}
   */
  public boolean enabled() {
    return maxSpan > 0;
  }
}
