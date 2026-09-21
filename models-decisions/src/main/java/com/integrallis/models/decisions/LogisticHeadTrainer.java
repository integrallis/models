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
 * Fits a {@link LinearDecisionHead} by deterministic multinomial logistic regression.
 *
 * <p>Full-batch gradient descent over a fixed iteration count, in a fixed order, with no shuffling,
 * no randomness and no early stopping. That is a deliberate choice rather than a simplification:
 * the pre-registration treats agreement and calibration error as exact functions of their inputs,
 * which only holds if fitting the same evidence twice returns the same weights twice.
 *
 * <p>Regularization applies to the weights and not the bias, so a class prior can still be learned
 * at full strength.
 */
public final class LogisticHeadTrainer {

  private final int iterations;
  private final double learningRate;
  private final double l2;

  /**
   * Creates a trainer.
   *
   * @param iterations the exact number of full-batch steps to take
   * @param learningRate the step size
   * @param l2 the weight-decay coefficient; zero disables it
   */
  public LogisticHeadTrainer(int iterations, double learningRate, double l2) {
    if (iterations <= 0) {
      throw new IllegalArgumentException("iterations must be positive");
    }
    if (!Double.isFinite(learningRate) || learningRate <= 0.0) {
      throw new IllegalArgumentException("learningRate must be positive and finite");
    }
    if (!Double.isFinite(l2) || l2 < 0.0) {
      throw new IllegalArgumentException("l2 must be non-negative and finite");
    }
    // Each step applies w -= learningRate * l2 * w, so the decay factor is (1 - learningRate * l2).
    // At or above 1 the weights flip sign and grow without bound, and the caller's first symptom is
    // a non-finite logit from deep inside the fit. Refuse it here, where the message can be useful.
    if (learningRate * l2 >= 1.0) {
      throw new IllegalArgumentException(
          "learningRate * l2 = "
              + (learningRate * l2)
              + " would diverge; the weight decay factor (1 - learningRate * l2) must stay below"
              + " one, so reduce l2 below "
              + (1.0 / learningRate));
    }
    this.iterations = iterations;
    this.learningRate = learningRate;
    this.l2 = l2;
  }

  /**
   * Fits a head over the given space.
   *
   * @param space the bounded answer space
   * @param features one hidden state per observation, all the same width
   * @param labels the observed outcome index for each observation
   */
  public LinearDecisionHead fit(AnswerSpace space, float[][] features, int[] labels) {
    Objects.requireNonNull(space, "space");
    Objects.requireNonNull(features, "features");
    Objects.requireNonNull(labels, "labels");
    if (features.length == 0) {
      throw new IllegalArgumentException("fitting needs at least one observation");
    }
    if (features.length != labels.length) {
      throw new IllegalArgumentException(
          "each observation needs its label, got " + features.length + " and " + labels.length);
    }
    int outcomes = space.size();
    int width = Objects.requireNonNull(features[0], "feature row").length;
    if (width == 0) {
      throw new IllegalArgumentException("a feature row must not be empty");
    }
    for (int row = 0; row < features.length; row++) {
      if (Objects.requireNonNull(features[row], "feature row").length != width) {
        throw new IllegalArgumentException("every feature row must be " + width + " wide");
      }
      if (labels[row] < 0 || labels[row] >= outcomes) {
        throw new IllegalArgumentException(
            "label " + labels[row] + " is outside the space's " + outcomes + " outcomes");
      }
    }

    double[][] weights = new double[outcomes][width];
    double[] bias = new double[outcomes];
    double[] scores = new double[outcomes];
    int observations = features.length;

    for (int step = 0; step < iterations; step++) {
      double[][] weightGradient = new double[outcomes][width];
      double[] biasGradient = new double[outcomes];

      for (int row = 0; row < observations; row++) {
        float[] feature = features[row];
        for (int outcome = 0; outcome < outcomes; outcome++) {
          double total = bias[outcome];
          double[] weightRow = weights[outcome];
          for (int index = 0; index < width; index++) {
            total += weightRow[index] * feature[index];
          }
          scores[outcome] = total;
        }
        double[] predicted = Calibration.softmax(scores, 1.0);
        for (int outcome = 0; outcome < outcomes; outcome++) {
          double error = predicted[outcome] - (labels[row] == outcome ? 1.0 : 0.0);
          double[] gradientRow = weightGradient[outcome];
          for (int index = 0; index < width; index++) {
            gradientRow[index] += error * feature[index];
          }
          biasGradient[outcome] += error;
        }
      }

      for (int outcome = 0; outcome < outcomes; outcome++) {
        double[] weightRow = weights[outcome];
        double[] gradientRow = weightGradient[outcome];
        for (int index = 0; index < width; index++) {
          weightRow[index] -=
              learningRate * (gradientRow[index] / observations + l2 * weightRow[index]);
        }
        bias[outcome] -= learningRate * (biasGradient[outcome] / observations);
      }
    }

    return new LinearDecisionHead(space, weights, bias);
  }
}
