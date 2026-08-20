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

import java.util.List;
import java.util.Objects;

/**
 * Aggregate confidence signals for one generated sequence.
 *
 * <p>The summary intentionally uses conservative extrema alongside mean log probability: a single
 * low-probability token or high-entropy step is often more actionable than an average alone.
 */
public record GenerationConfidence(
    int tokenCount, float minProbability, float meanLogProbability, float maxEntropy) {

  public GenerationConfidence {
    if (tokenCount <= 0) {
      throw new IllegalArgumentException("tokenCount must be > 0: " + tokenCount);
    }
    if (!Float.isFinite(minProbability) || minProbability < 0.0f || minProbability > 1.0f) {
      throw new IllegalArgumentException(
          "minProbability must be finite and in [0, 1]: " + minProbability);
    }
    if (!Float.isFinite(meanLogProbability) || meanLogProbability > 0.0f) {
      throw new IllegalArgumentException(
          "meanLogProbability must be finite and <= 0: " + meanLogProbability);
    }
    if (!Float.isFinite(maxEntropy) || maxEntropy < 0.0f) {
      throw new IllegalArgumentException("maxEntropy must be finite and >= 0: " + maxEntropy);
    }
  }

  /** Convenience method used by tests and callers that need an explicit empty rejection. */
  public static GenerationConfidence empty() {
    return fromSignals(List.of());
  }

  /** Aggregates a non-empty varargs list of token confidence signals. */
  public static GenerationConfidence fromSignals(TokenConfidenceSignal... signals) {
    Objects.requireNonNull(signals, "signals");
    return fromSignals(List.of(signals));
  }

  /** Aggregates a non-empty list of token confidence signals. */
  public static GenerationConfidence fromSignals(List<TokenConfidenceSignal> signals) {
    Objects.requireNonNull(signals, "signals");
    if (signals.isEmpty()) {
      throw new IllegalArgumentException("signals must not be empty");
    }

    float minProbability = 1.0f;
    float maxEntropy = 0.0f;
    double logProbabilitySum = 0.0d;
    for (TokenConfidenceSignal signal : signals) {
      Objects.requireNonNull(signal, "signals must not contain null");
      minProbability = Math.min(minProbability, signal.chosenProbability());
      maxEntropy = Math.max(maxEntropy, signal.entropy());
      logProbabilitySum += Math.log(Math.max(signal.chosenProbability(), 1.0e-12f));
    }
    return new GenerationConfidence(
        signals.size(), minProbability, (float) (logProbabilitySum / signals.size()), maxEntropy);
  }
}
