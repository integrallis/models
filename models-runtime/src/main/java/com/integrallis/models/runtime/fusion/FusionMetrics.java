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
package com.integrallis.models.runtime.fusion;

/**
 * Running totals of a {@link FusedInferenceBackend}, reset with the backend.
 *
 * @param steps decode steps since the last reset (prefill counts as one)
 * @param consulted steps on which the auxiliaries voted
 * @param catchUpTokens tokens the auxiliaries consumed to catch up, summed over auxiliaries
 * @param leadNanos time in the lead
 * @param auxiliaryNanos time in the auxiliaries
 * @param fusionNanos time in the entropy and combination arithmetic
 */
public record FusionMetrics(
    long steps,
    long consulted,
    long catchUpTokens,
    long leadNanos,
    long auxiliaryNanos,
    long fusionNanos) {

  /** No activity yet. */
  public static FusionMetrics empty() {
    return new FusionMetrics(0, 0, 0, 0, 0, 0);
  }

  /** Fraction of steps on which the auxiliaries voted, or {@code NaN} before any step. */
  public double consultRate() {
    return steps == 0 ? Double.NaN : (double) consulted / steps;
  }

  FusionMetrics plus(FusionStep step) {
    long tokens = 0;
    for (int count : step.catchUpTokens()) {
      tokens += count;
    }
    return new FusionMetrics(
        steps + 1,
        consulted + (step.consulted() ? 1 : 0),
        catchUpTokens + tokens,
        leadNanos + step.leadNanos(),
        auxiliaryNanos + step.auxiliaryNanos(),
        fusionNanos + step.fusionNanos());
  }
}
