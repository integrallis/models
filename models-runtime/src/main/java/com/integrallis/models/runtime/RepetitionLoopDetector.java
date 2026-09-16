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
package com.integrallis.models.runtime;

import com.integrallis.models.api.RepetitionLoopDetection;
import java.util.Objects;

/**
 * Streaming detector for exactly periodic tails in a generated token sequence.
 *
 * <p>For every candidate period {@code L} in {@code [1, maxSpan]} the detector keeps {@code
 * run[L]}, the number of consecutive most-recent positions {@code i} with {@code t[i] == t[i - L]}.
 * The sequence then ends in a period-{@code L} run of {@code run[L] + L} tokens, which is a loop
 * once that length reaches both {@code minRepeats * L} and {@code minLoopTokens}. Each token costs
 * at most {@code maxSpan} comparisons against a ring buffer of the last {@code maxSpan} tokens, so
 * work and memory are independent of how long generation runs.
 *
 * <p>Instances are single-use and not thread-safe; one belongs to one generation request.
 */
final class RepetitionLoopDetector {
  private final int maxSpan;
  private final int minRepeats;
  private final int minLoopTokens;
  private final int[] recent;
  private final int[] runs;
  private long seen;
  private long comparisons;
  private int detectedSpan;

  RepetitionLoopDetector(RepetitionLoopDetection config) {
    Objects.requireNonNull(config, "config");
    maxSpan = config.maxSpan();
    minRepeats = config.minRepeats();
    minLoopTokens = config.minLoopTokens();
    recent = new int[maxSpan];
    runs = new int[maxSpan + 1];
  }

  /**
   * Consumes the next generated token.
   *
   * @return {@code true} when this token completes a loop; {@link #detectedSpan()} then reports the
   *     shortest looping period
   */
  boolean accept(int token) {
    if (maxSpan == 0) {
      return false;
    }
    int writeIndex = (int) (seen % maxSpan);
    int comparable = (int) Math.min(maxSpan, seen);
    int found = 0;
    for (int span = 1; span <= comparable; span++) {
      int previous = recent[Math.floorMod(writeIndex - span, maxSpan)];
      comparisons++;
      if (previous == token) {
        int run = ++runs[span];
        if (found == 0 && isLoop(span, run)) {
          found = span;
        }
      } else {
        runs[span] = 0;
      }
    }
    recent[writeIndex] = token;
    seen++;
    if (found != 0) {
      detectedSpan = found;
      return true;
    }
    return false;
  }

  private boolean isLoop(int span, int run) {
    long periodicLength = (long) run + span;
    return periodicLength >= (long) minRepeats * span && periodicLength >= minLoopTokens;
  }

  /** Returns the shortest period of the most recently detected loop, or zero before detection. */
  int detectedSpan() {
    return detectedSpan;
  }

  /** Returns the number of token comparisons performed; exposed so cost can be verified. */
  long comparisons() {
    return comparisons;
  }
}
