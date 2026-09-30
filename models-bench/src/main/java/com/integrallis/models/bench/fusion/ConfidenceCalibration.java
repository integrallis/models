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
package com.integrallis.models.bench.fusion;

import java.util.List;

/**
 * Per-member temperature calibration of answer-token confidence (gate G7).
 *
 * <p>Confidence {@code c} is the mean log-probability of the answer tokens. The calibrated
 * probability of being correct is {@code p = exp(c / τ)}; {@code τ} minimises the Bernoulli
 * negative log-likelihood over a log-spaced grid on [0.05, 20]. ECE uses 10 equal-width bins.
 */
public final class ConfidenceCalibration {

  static final int GRID_POINTS = 401;
  static final double MIN_TEMPERATURE = 0.05;
  static final double MAX_TEMPERATURE = 20.0;
  static final int ECE_BINS = 10;

  private ConfidenceCalibration() {}

  /** One development-split observation. */
  public record Sample(double meanLogProbability, boolean correct) {}

  /** A fitted temperature and its calibration error before (τ = 1) and after. */
  public record Fit(double temperature, double eceBefore, double eceAfter, int samples) {}

  static double probability(double meanLogProbability, double temperature) {
    return Math.min(1.0, Math.exp(meanLogProbability / temperature));
  }

  public static Fit fit(List<Sample> samples) {
    if (samples.isEmpty()) {
      throw new IllegalArgumentException("calibration requires at least one sample");
    }
    double bestTemperature = 1.0;
    double bestLoss = Double.POSITIVE_INFINITY;
    double logMin = Math.log(MIN_TEMPERATURE);
    double logMax = Math.log(MAX_TEMPERATURE);
    for (int i = 0; i < GRID_POINTS; i++) {
      double temperature = Math.exp(logMin + (logMax - logMin) * i / (GRID_POINTS - 1));
      double loss = 0;
      for (Sample sample : samples) {
        double p =
            Math.clamp(probability(sample.meanLogProbability(), temperature), 1e-6, 1 - 1e-6);
        loss -= sample.correct() ? Math.log(p) : Math.log(1 - p);
      }
      if (loss < bestLoss) {
        bestLoss = loss;
        bestTemperature = temperature;
      }
    }
    return new Fit(
        bestTemperature,
        expectedCalibrationError(samples, 1.0, ECE_BINS),
        expectedCalibrationError(samples, bestTemperature, ECE_BINS),
        samples.size());
  }

  public static double expectedCalibrationError(
      List<Sample> samples, double temperature, int bins) {
    double[] confidence = new double[bins];
    double[] accuracy = new double[bins];
    int[] counts = new int[bins];
    for (Sample sample : samples) {
      double p = probability(sample.meanLogProbability(), temperature);
      int bin = Math.min(bins - 1, (int) Math.floor(p * bins));
      confidence[bin] += p;
      accuracy[bin] += sample.correct() ? 1 : 0;
      counts[bin]++;
    }
    double ece = 0;
    for (int b = 0; b < bins; b++) {
      if (counts[b] > 0) {
        ece +=
            Math.abs(accuracy[b] / counts[b] - confidence[b] / counts[b])
                * counts[b]
                / samples.size();
      }
    }
    return ece;
  }
}
