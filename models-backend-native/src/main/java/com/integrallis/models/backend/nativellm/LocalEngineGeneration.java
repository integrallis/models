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
package com.integrallis.models.backend.nativellm;

import java.util.Objects;

/** Text and timing observations returned by one local-engine generation request. */
public record LocalEngineGeneration(
    String text,
    int inputTokens,
    int outputTokens,
    double firstTokenMillis,
    double totalMillis,
    double prefillTokensPerSecond,
    double loadMillis) {
  public LocalEngineGeneration {
    Objects.requireNonNull(text, "text");
    if (inputTokens < 0
        || outputTokens < 0
        || !nonNegativeFinite(firstTokenMillis)
        || !nonNegativeFinite(totalMillis)
        || !nonNegativeFinite(prefillTokensPerSecond)
        || !nonNegativeFinite(loadMillis)) {
      throw new IllegalArgumentException("generation measurements must be finite and non-negative");
    }
    if (firstTokenMillis > totalMillis) {
      throw new IllegalArgumentException("first-token latency cannot exceed total latency");
    }
  }

  private static boolean nonNegativeFinite(double value) {
    return Double.isFinite(value) && value >= 0;
  }
}
