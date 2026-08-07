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
package com.integrallis.models.router;

import java.util.OptionalDouble;

/**
 * How to trade cost, quality, latency and reliability against each other.
 *
 * <p>Weights are relative and get normalized, so only their ratios matter. Constraints are hard: a
 * candidate that breaks one is removed before scoring rather than merely penalized, because a
 * context window that cannot hold the prompt is not a preference.
 *
 * <p>Most applications should use a preset. Routing is a tuning problem with no single right
 * answer, and hand-weighting five dimensions is a poor first experience, so {@link #BALANCED},
 * {@link #CHEAPEST}, {@link #FASTEST}, {@link #BEST_QUALITY}, {@link #LOCAL_FIRST} and {@link
 * #PRIVACY_STRICT} are pre-tuned. {@link #costQualityTradeoff(int)} collapses the whole space onto
 * one 0-to-10 dial for applications that want a single knob.
 *
 * @param costWeight relative weight on being cheap
 * @param qualityWeight relative weight on task quality
 * @param latencyWeight relative weight on responding quickly
 * @param availabilityWeight relative weight on observed reliability
 * @param localityWeight relative weight on staying in process
 * @param minimumQuality quality floor a candidate must clear
 * @param maximumCostPerMillionTokens blended price ceiling
 * @param maximumTimeToFirstTokenMillis time-to-first-token ceiling
 * @param localOnly whether hosted models are excluded outright
 */
public record RoutingPolicy(
    double costWeight,
    double qualityWeight,
    double latencyWeight,
    double availabilityWeight,
    double localityWeight,
    OptionalDouble minimumQuality,
    OptionalDouble maximumCostPerMillionTokens,
    OptionalDouble maximumTimeToFirstTokenMillis,
    boolean localOnly) {

  /** Sensible default: quality-leaning without ignoring cost or latency. */
  public static final RoutingPolicy BALANCED = weights(0.30, 0.40, 0.15, 0.10, 0.05);

  /** Minimize spend, which in a mixed fleet means preferring local models. */
  public static final RoutingPolicy CHEAPEST = weights(0.70, 0.15, 0.05, 0.05, 0.05);

  /** Minimize time to first token. */
  public static final RoutingPolicy FASTEST = weights(0.05, 0.15, 0.70, 0.05, 0.05);

  /**
   * Maximize task quality and let cost fall where it may.
   *
   * <p>Cost carries no weight at all. A token weight is worse than none: it leaves the preset
   * picking a cheaper model whenever two are within a hair of each other on quality, which is
   * exactly the surprise a preset with this name must not deliver.
   */
  public static final RoutingPolicy BEST_QUALITY = weights(0.00, 0.95, 0.00, 0.05, 0.00);

  /** Prefer in-process models, but escalate when a hosted model is clearly better. */
  public static final RoutingPolicy LOCAL_FIRST = weights(0.20, 0.30, 0.10, 0.05, 0.35);

  /** Never leave the process, whatever the quality cost. */
  public static final RoutingPolicy PRIVACY_STRICT =
      new RoutingPolicy(
          0.20,
          0.55,
          0.15,
          0.10,
          0.00,
          OptionalDouble.empty(),
          OptionalDouble.empty(),
          OptionalDouble.empty(),
          true);

  /** Validates weights and constraints. */
  public RoutingPolicy {
    requireWeight(costWeight, "costWeight");
    requireWeight(qualityWeight, "qualityWeight");
    requireWeight(latencyWeight, "latencyWeight");
    requireWeight(availabilityWeight, "availabilityWeight");
    requireWeight(localityWeight, "localityWeight");
    if (costWeight + qualityWeight + latencyWeight + availabilityWeight + localityWeight <= 0) {
      throw new IllegalArgumentException("at least one weight must be positive");
    }
  }

  /**
   * Collapses the trade-off onto one dial, in the spirit of a cost/quality slider.
   *
   * @param zeroToTen zero favours capability, ten favours economy
   * @return an interpolated policy
   */
  public static RoutingPolicy costQualityTradeoff(int zeroToTen) {
    if (zeroToTen < 0 || zeroToTen > 10) {
      throw new IllegalArgumentException("costQualityTradeoff must be within [0, 10]");
    }
    double economy = zeroToTen / 10.0;
    return weights(0.05 + 0.70 * economy, 0.85 - 0.70 * economy, 0.05, 0.05, 0.05 * economy);
  }

  /**
   * Returns a copy that also enforces a quality floor.
   *
   * @param minimum quality a candidate must reach for the classified task
   * @return a constrained policy
   */
  public RoutingPolicy withMinimumQuality(double minimum) {
    return new RoutingPolicy(
        costWeight,
        qualityWeight,
        latencyWeight,
        availabilityWeight,
        localityWeight,
        OptionalDouble.of(minimum),
        maximumCostPerMillionTokens,
        maximumTimeToFirstTokenMillis,
        localOnly);
  }

  /**
   * Returns a copy that also enforces a price ceiling.
   *
   * @param maximum blended price per million tokens
   * @return a constrained policy
   */
  public RoutingPolicy withMaximumCostPerMillionTokens(double maximum) {
    return new RoutingPolicy(
        costWeight,
        qualityWeight,
        latencyWeight,
        availabilityWeight,
        localityWeight,
        minimumQuality,
        OptionalDouble.of(maximum),
        maximumTimeToFirstTokenMillis,
        localOnly);
  }

  /**
   * Returns a copy that also enforces a latency ceiling.
   *
   * @param maximum time to first token in milliseconds
   * @return a constrained policy
   */
  public RoutingPolicy withMaximumTimeToFirstTokenMillis(double maximum) {
    return new RoutingPolicy(
        costWeight,
        qualityWeight,
        latencyWeight,
        availabilityWeight,
        localityWeight,
        minimumQuality,
        maximumCostPerMillionTokens,
        OptionalDouble.of(maximum),
        localOnly);
  }

  /**
   * Returns a copy that keeps every request in process.
   *
   * @return a local-only policy
   */
  public RoutingPolicy withLocalOnly() {
    return new RoutingPolicy(
        costWeight,
        qualityWeight,
        latencyWeight,
        availabilityWeight,
        localityWeight,
        minimumQuality,
        maximumCostPerMillionTokens,
        maximumTimeToFirstTokenMillis,
        true);
  }

  /**
   * Reports whether hosted models are excluded.
   *
   * @return true when routing may not leave the process
   */
  public boolean isLocalOnly() {
    return localOnly;
  }

  private static RoutingPolicy weights(
      double cost, double quality, double latency, double availability, double locality) {
    return new RoutingPolicy(
        cost,
        quality,
        latency,
        availability,
        locality,
        OptionalDouble.empty(),
        OptionalDouble.empty(),
        OptionalDouble.empty(),
        false);
  }

  private static void requireWeight(double value, String field) {
    if (!Double.isFinite(value) || value < 0) {
      throw new IllegalArgumentException(field + " must be finite and non-negative");
    }
  }
}
