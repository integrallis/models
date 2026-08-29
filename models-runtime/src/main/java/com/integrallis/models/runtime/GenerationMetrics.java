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

import com.integrallis.models.api.GenerationUsage;
import java.time.Duration;
import java.util.Objects;
import java.util.Optional;

/** Runtime-owned phase measurements for one text-generation request. */
public record GenerationMetrics(
    boolean available,
    boolean successful,
    Duration tokenization,
    Duration promptPreparation,
    Duration prefill,
    Optional<Duration> timeToFirstToken,
    Duration decode,
    Duration total,
    GenerationUsage usage,
    PromptCacheMetrics promptCache) {

  public GenerationMetrics {
    tokenization = requireDuration(tokenization, "tokenization");
    promptPreparation = requireDuration(promptPreparation, "promptPreparation");
    prefill = requireDuration(prefill, "prefill");
    timeToFirstToken = Objects.requireNonNull(timeToFirstToken, "timeToFirstToken");
    timeToFirstToken.ifPresent(value -> requireDuration(value, "timeToFirstToken"));
    decode = requireDuration(decode, "decode");
    total = requireDuration(total, "total");
    usage = Objects.requireNonNull(usage, "usage");
    promptCache = Objects.requireNonNull(promptCache, "promptCache");
    if (!available && successful) {
      throw new IllegalArgumentException("unavailable metrics cannot report success");
    }
  }

  /** Returns an empty snapshot for a runtime that has not completed a request. */
  public static GenerationMetrics unavailable() {
    return new GenerationMetrics(
        false,
        false,
        Duration.ZERO,
        Duration.ZERO,
        Duration.ZERO,
        Optional.empty(),
        Duration.ZERO,
        Duration.ZERO,
        new GenerationUsage(0, 0),
        PromptCacheMetrics.unavailable());
  }

  /**
   * Returns decode throughput using the intervals after the first generated token.
   *
   * <p>A single-token or empty completion has no measurable decode interval and returns zero.
   */
  public double decodeTokensPerSecond() {
    int intervals = Math.max(0, usage.completionTokens() - 1);
    if (intervals == 0) {
      return 0;
    }
    return intervals / Math.max(0.000_000_001, decode.toNanos() / 1_000_000_000.0);
  }

  private static Duration requireDuration(Duration value, String name) {
    Objects.requireNonNull(value, name);
    if (value.isNegative()) {
      throw new IllegalArgumentException(name + " must not be negative");
    }
    return value;
  }
}
