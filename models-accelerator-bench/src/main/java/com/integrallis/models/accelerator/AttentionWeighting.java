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
package com.integrallis.models.accelerator;

import java.util.Objects;

/** Attention-sensitivity weighting with an effective-sample-size safety bound. */
public final class AttentionWeighting {

  private AttentionWeighting() {}

  /**
   * Normalizes sensitivities to mean one, then retains as much contrast as the regression can
   * support.
   *
   * <p>The returned weights are {@code 1 + alpha * (sensitivity - 1)}. The largest alpha in
   * {@code [0,1]} is selected such that the Kish effective sample size is at least {@code
   * min(observations, 2 * parameters)}.
   */
  public static double[] shrinkToEffectiveSampleSize(
      double[] sensitivities, int parameterCount) {
    Objects.requireNonNull(sensitivities, "sensitivities");
    if (sensitivities.length == 0) {
      throw new IllegalArgumentException("sensitivities must not be empty");
    }
    if (parameterCount < 1) {
      throw new IllegalArgumentException("parameterCount must be positive");
    }
    double sum = 0.0;
    for (int index = 0; index < sensitivities.length; index++) {
      double sensitivity = sensitivities[index];
      if (!Double.isFinite(sensitivity)) {
        throw new IllegalArgumentException(
            "sensitivities must be finite; sensitivities[" + index + "]=" + sensitivity);
      }
      if (sensitivity < 0.0) {
        throw new IllegalArgumentException(
            "sensitivities must be nonnegative; sensitivities[" + index + "]=" + sensitivity);
      }
      sum += sensitivity;
    }
    if (!(sum > 0.0) || !Double.isFinite(sum)) {
      throw new IllegalArgumentException("at least one positive sensitivity is required");
    }

    int observations = sensitivities.length;
    double scale = observations / sum;
    double[] normalized = new double[observations];
    double coefficientOfVariationSquared = 0.0;
    for (int index = 0; index < observations; index++) {
      normalized[index] = sensitivities[index] * scale;
      double centered = normalized[index] - 1.0;
      coefficientOfVariationSquared += centered * centered;
    }
    coefficientOfVariationSquared /= observations;

    double minimumEffectiveSamples = Math.min(observations, 2.0 * parameterCount);
    double alpha = 1.0;
    if (coefficientOfVariationSquared > 0.0) {
      double maximumSquared =
          (observations / minimumEffectiveSamples - 1.0) / coefficientOfVariationSquared;
      alpha = Math.min(1.0, Math.sqrt(Math.max(0.0, maximumSquared)));
    }
    double[] weights = new double[observations];
    for (int index = 0; index < observations; index++) {
      weights[index] = 1.0 + alpha * (normalized[index] - 1.0);
    }
    return weights;
  }

  /** Returns the Kish effective sample size of finite, nonnegative weights. */
  public static double effectiveSampleSize(double[] weights) {
    Objects.requireNonNull(weights, "weights");
    double sum = 0.0;
    double sumSquares = 0.0;
    for (int index = 0; index < weights.length; index++) {
      double weight = weights[index];
      if (!Double.isFinite(weight) || weight < 0.0) {
        throw new IllegalArgumentException("weights must be finite and nonnegative");
      }
      sum += weight;
      sumSquares += weight * weight;
    }
    if (!(sumSquares > 0.0)) {
      throw new IllegalArgumentException("at least one positive weight is required");
    }
    return sum * sum / sumSquares;
  }
}
