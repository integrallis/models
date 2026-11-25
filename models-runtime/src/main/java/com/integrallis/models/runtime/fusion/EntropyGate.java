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
 * Consults the auxiliaries when the lead's entropy reaches a threshold.
 *
 * <p>Below the threshold the lead is confident enough to emit alone; at or above it the auxiliaries
 * are asked. The published recipe uses 0.65 nats.
 *
 * @param thresholdNats entropy, in nats, at and above which the auxiliaries are consulted; must be
 *     finite and non-negative, or {@link Double#POSITIVE_INFINITY} to never consult
 */
public record EntropyGate(double thresholdNats) implements EnsembleGate {

  /** The threshold the published recipe uses. */
  public static final double PUBLISHED_THRESHOLD_NATS = 0.65;

  /** Validates the threshold. */
  public EntropyGate {
    if (Double.isNaN(thresholdNats) || thresholdNats < 0) {
      throw new IllegalArgumentException(
          "thresholdNats must be non-negative (or +Infinity): " + thresholdNats);
    }
  }

  @Override
  public boolean consult(double leadEntropyNats, int position) {
    return leadEntropyNats >= thresholdNats;
  }
}
