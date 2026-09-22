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
 * Turning scores into probabilities, and measuring whether those probabilities mean anything.
 *
 * <p>Scaling and scoring are kept together deliberately: a temperature is only defensible once the
 * scoring rules say it improved calibration on held-out data. Both directions are pure functions of
 * their arguments and give bit-identical results for identical input.
 */
public final class Calibration {

  private Calibration() {}

  /**
   * Converts scores to a distribution, dividing by a temperature first.
   *
   * <p>A temperature above one flattens the distribution and below one sharpens it; neither moves
   * the winner. The largest score is subtracted before exponentiating, so large scores stay finite.
   *
   * @param logits the raw scores, one per outcome
   * @param temperature a positive finite scaling factor, one for no scaling
   * @return a fresh distribution summing to one
   */
  public static double[] softmax(double[] logits, double temperature) {
    Objects.requireNonNull(logits, "logits");
    if (logits.length == 0) {
      throw new IllegalArgumentException("logits must not be empty");
    }
    if (!Double.isFinite(temperature) || temperature <= 0.0) {
      throw new IllegalArgumentException("temperature must be positive and finite");
    }
    double[] scaled = new double[logits.length];
    double largest = Double.NEGATIVE_INFINITY;
    for (int index = 0; index < logits.length; index++) {
      double value = logits[index];
      if (!Double.isFinite(value)) {
        throw new IllegalArgumentException("logits must be finite");
      }
      scaled[index] = value / temperature;
      largest = Math.max(largest, scaled[index]);
    }
    double total = 0.0;
    for (int index = 0; index < scaled.length; index++) {
      scaled[index] = Math.exp(scaled[index] - largest);
      total += scaled[index];
    }
    for (int index = 0; index < scaled.length; index++) {
      scaled[index] /= total;
    }
    return scaled;
  }

  /**
   * Returns the expected calibration error: how far stated confidence sits from observed accuracy,
   * averaged over equal-width confidence bins and weighted by how many decisions landed in each.
   *
   * <p>Zero means every bin was right as often as it claimed. It does not mean the decisions were
   * accurate, only that their confidence was honest.
   *
   * @param confidences the stated confidence of each decision, each within zero and one
   * @param correct whether each decision turned out to be right
   * @param bins how many equal-width confidence bins to use
   */
  public static double expectedCalibrationError(double[] confidences, boolean[] correct, int bins) {
    requirePairedEvidence(confidences, correct);
    if (bins <= 0) {
      throw new IllegalArgumentException("bins must be positive");
    }
    int[] counts = new int[bins];
    int[] hits = new int[bins];
    double[] confidenceSums = new double[bins];
    for (int index = 0; index < confidences.length; index++) {
      double confidence = confidences[index];
      if (!Double.isFinite(confidence) || confidence < 0.0 || confidence > 1.0) {
        throw new IllegalArgumentException("confidence must lie within zero and one");
      }
      int bin = Math.min((int) (confidence * bins), bins - 1);
      counts[bin]++;
      confidenceSums[bin] += confidence;
      if (correct[index]) {
        hits[bin]++;
      }
    }
    double error = 0.0;
    for (int bin = 0; bin < bins; bin++) {
      if (counts[bin] == 0) {
        continue;
      }
      double accuracy = (double) hits[bin] / counts[bin];
      double meanConfidence = confidenceSums[bin] / counts[bin];
      error += ((double) counts[bin] / confidences.length) * Math.abs(accuracy - meanConfidence);
    }
    return error;
  }

  /**
   * Returns the Brier score: the mean squared distance between a stated probability and what
   * actually happened. Zero is a confident right answer; one is a confident wrong one.
   *
   * @param probabilities the probability each decision assigned to the outcome being true
   * @param outcomes whether the outcome was in fact true
   */
  public static double brierScore(double[] probabilities, boolean[] outcomes) {
    requirePairedEvidence(probabilities, outcomes);
    double total = 0.0;
    for (int index = 0; index < probabilities.length; index++) {
      double probability = probabilities[index];
      if (!Double.isFinite(probability) || probability < 0.0 || probability > 1.0) {
        throw new IllegalArgumentException("probability must lie within zero and one");
      }
      double observed = outcomes[index] ? 1.0 : 0.0;
      double residual = probability - observed;
      total += residual * residual;
    }
    return total / probabilities.length;
  }

  private static void requirePairedEvidence(double[] values, boolean[] outcomes) {
    Objects.requireNonNull(values, "values");
    Objects.requireNonNull(outcomes, "outcomes");
    if (values.length != outcomes.length) {
      throw new IllegalArgumentException(
          "each value needs its outcome, got " + values.length + " and " + outcomes.length);
    }
    if (values.length == 0) {
      throw new IllegalArgumentException("calibration needs at least one decision");
    }
  }
}
