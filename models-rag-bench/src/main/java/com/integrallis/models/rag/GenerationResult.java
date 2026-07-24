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
package com.integrallis.models.rag;

import java.util.Objects;

/** Text and server-observed timing for one generation. */
public record GenerationResult(
    String text,
    int inputTokens,
    int cacheReadInputTokens,
    int cacheWriteInputTokens,
    int outputTokens,
    double ttftMillis,
    double totalMillis,
    double prefillTokensPerSecond,
    double loadMillis,
    long peakRssBytes,
    double cpuMillis,
    Double estimatedApiCostUsd) {
  public GenerationResult(
      String text,
      int inputTokens,
      int outputTokens,
      double ttftMillis,
      double totalMillis,
      double prefillTokensPerSecond,
      double loadMillis) {
    this(
        text,
        inputTokens,
        0,
        0,
        outputTokens,
        ttftMillis,
        totalMillis,
        prefillTokensPerSecond,
        loadMillis,
        0,
        0,
        null);
  }

  public GenerationResult(
      String text,
      int inputTokens,
      int outputTokens,
      double ttftMillis,
      double totalMillis,
      double prefillTokensPerSecond,
      double loadMillis,
      long peakRssBytes,
      double cpuMillis) {
    this(
        text,
        inputTokens,
        0,
        0,
        outputTokens,
        ttftMillis,
        totalMillis,
        prefillTokensPerSecond,
        loadMillis,
        peakRssBytes,
        cpuMillis,
        null);
  }

  public GenerationResult {
    text = Objects.requireNonNull(text, "text");
    if (inputTokens < 0 || outputTokens < 1) {
      throw new IllegalArgumentException("token counts must be non-negative with outputTokens > 0");
    }
    if (cacheReadInputTokens < 0
        || cacheWriteInputTokens < 0
        || cacheReadInputTokens + cacheWriteInputTokens > inputTokens) {
      throw new IllegalArgumentException(
          "cache token counts must be non-negative and not exceed inputTokens");
    }
    requireNonNegative(ttftMillis, "ttftMillis");
    requireNonNegative(totalMillis, "totalMillis");
    requireNonNegative(prefillTokensPerSecond, "prefillTokensPerSecond");
    requireNonNegative(loadMillis, "loadMillis");
    if (peakRssBytes < 0) {
      throw new IllegalArgumentException("peakRssBytes must be non-negative");
    }
    requireNonNegative(cpuMillis, "cpuMillis");
    if (estimatedApiCostUsd != null) {
      requireNonNegative(estimatedApiCostUsd, "estimatedApiCostUsd");
    }
    if (totalMillis < ttftMillis) {
      throw new IllegalArgumentException("totalMillis must be >= ttftMillis");
    }
  }

  public double tpotMillis() {
    return outputTokens > 1 ? (totalMillis - ttftMillis) / (outputTokens - 1) : 0;
  }

  public double decodeTokensPerSecond() {
    double tpot = tpotMillis();
    return tpot > 0 ? 1_000.0 / tpot : 0;
  }

  private static void requireNonNegative(double value, String name) {
    if (!Double.isFinite(value) || value < 0) {
      throw new IllegalArgumentException(name + " must be finite and non-negative");
    }
  }
}
