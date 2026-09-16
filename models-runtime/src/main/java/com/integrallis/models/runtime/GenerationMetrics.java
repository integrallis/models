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
import com.integrallis.models.api.StopReason;
import java.time.Duration;
import java.util.Objects;
import java.util.Optional;

/**
 * Runtime-owned phase measurements for one text-generation request.
 *
 * @param available whether a request has completed and been measured
 * @param successful whether that request completed without failure
 * @param tokenization prompt tokenization time
 * @param promptPreparation prompt-cache preparation time
 * @param prefill prompt prefill time
 * @param timeToFirstToken time from request start to the first generated token, when one was
 * @param decode time from the first generated token to completion
 * @param total request wall time
 * @param usage prompt and completion token counts
 * @param promptCache prompt-prefix cache measurements
 * @param stopReason why a successful generation ended; empty when unavailable or failed
 */
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
    PromptCacheMetrics promptCache,
    Optional<StopReason> stopReason) {

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
    stopReason = Objects.requireNonNull(stopReason, "stopReason");
    if (stopReason.isPresent() && !successful) {
      throw new IllegalArgumentException("only a successful generation has a stop reason");
    }
    if (!available && successful) {
      throw new IllegalArgumentException("unavailable metrics cannot report success");
    }
  }

  /**
   * Creates metrics without a stop reason, preserving the original canonical signature.
   *
   * @param available whether a request has completed and been measured
   * @param successful whether that request completed without failure
   * @param tokenization prompt tokenization time
   * @param promptPreparation prompt-cache preparation time
   * @param prefill prompt prefill time
   * @param timeToFirstToken time to the first generated token, when one was
   * @param decode decode time
   * @param total request wall time
   * @param usage prompt and completion token counts
   * @param promptCache prompt-prefix cache measurements
   */
  public GenerationMetrics(
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
    this(
        available,
        successful,
        tokenization,
        promptPreparation,
        prefill,
        timeToFirstToken,
        decode,
        total,
        usage,
        promptCache,
        Optional.empty());
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
