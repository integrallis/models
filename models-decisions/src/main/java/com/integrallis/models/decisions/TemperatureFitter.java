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
package com.integrallis.models.decisions;

import java.util.Objects;

/**
 * Finds the temperature that makes a head's probabilities honest.
 *
 * <p>Temperature scaling is fitted on held-out evidence by minimizing negative log likelihood. It
 * cannot change which outcome wins, so it cannot buy accuracy; all it can do is move the stated
 * confidence toward the observed frequency, which is the only thing calibration claims.
 *
 * <p>The search is a fixed-iteration ternary search over log temperature, so it is deterministic.
 */
public final class TemperatureFitter {

  private static final double MIN_TEMPERATURE = 1.0 / 64.0;
  private static final double MAX_TEMPERATURE = 64.0;
  private static final int ITERATIONS = 200;

  private TemperatureFitter() {}

  /**
   * Returns the temperature minimizing negative log likelihood on this evidence.
   *
   * @param logits one raw score vector per observation
   * @param labels the observed outcome index for each observation
   */
  public static double fit(double[][] logits, int[] labels) {
    requirePairedEvidence(logits, labels);
    double low = Math.log(MIN_TEMPERATURE);
    double high = Math.log(MAX_TEMPERATURE);
    for (int step = 0; step < ITERATIONS; step++) {
      double firstThird = low + (high - low) / 3.0;
      double secondThird = high - (high - low) / 3.0;
      if (negativeLogLikelihood(logits, labels, Math.exp(firstThird))
          < negativeLogLikelihood(logits, labels, Math.exp(secondThird))) {
        high = secondThird;
      } else {
        low = firstThird;
      }
    }
    return Math.exp((low + high) / 2.0);
  }

  /**
   * Returns the mean negative log likelihood of the observed outcomes at one temperature.
   *
   * @param logits one raw score vector per observation
   * @param labels the observed outcome index for each observation
   * @param temperature the scaling to apply before measuring
   */
  public static double negativeLogLikelihood(double[][] logits, int[] labels, double temperature) {
    requirePairedEvidence(logits, labels);
    double total = 0.0;
    for (int row = 0; row < logits.length; row++) {
      double[] probabilities = Calibration.softmax(logits[row], temperature);
      int label = labels[row];
      if (label < 0 || label >= probabilities.length) {
        throw new IllegalArgumentException("label " + label + " is outside the score vector");
      }
      total -= Math.log(Math.max(probabilities[label], Double.MIN_NORMAL));
    }
    return total / logits.length;
  }

  private static void requirePairedEvidence(double[][] logits, int[] labels) {
    Objects.requireNonNull(logits, "logits");
    Objects.requireNonNull(labels, "labels");
    if (logits.length != labels.length) {
      throw new IllegalArgumentException(
          "each score vector needs its label, got " + logits.length + " and " + labels.length);
    }
    if (logits.length == 0) {
      throw new IllegalArgumentException("fitting needs at least one observation");
    }
  }
}
