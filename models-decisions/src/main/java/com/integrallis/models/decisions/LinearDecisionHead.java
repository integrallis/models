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
 * An affine head: one weight row per outcome, plus a bias.
 *
 * <p>Accumulation runs in a fixed order so repeated reads of one hidden state agree bit for bit. A
 * decision that moved between runs could not be calibrated, and calibration is the product.
 */
public final class LinearDecisionHead implements DecisionHead {

  private final AnswerSpace space;
  private final double[][] weights;
  private final double[] bias;
  private final int width;

  /**
   * Creates a head over the given space.
   *
   * @param space the answer space to score
   * @param weights one row per outcome, every row the width of the hidden state
   * @param bias one bias per outcome
   * @throws IllegalArgumentException if the shapes disagree with the space
   */
  public LinearDecisionHead(AnswerSpace space, double[][] weights, double[] bias) {
    this.space = Objects.requireNonNull(space, "space");
    Objects.requireNonNull(weights, "weights");
    Objects.requireNonNull(bias, "bias");
    if (weights.length != space.size()) {
      throw new IllegalArgumentException(
          "need one weight row per outcome: " + space.size() + ", got " + weights.length);
    }
    if (bias.length != space.size()) {
      throw new IllegalArgumentException(
          "need one bias per outcome: " + space.size() + ", got " + bias.length);
    }
    Objects.requireNonNull(weights[0], "weight row");
    this.width = weights[0].length;
    if (width == 0) {
      throw new IllegalArgumentException("a weight row must not be empty");
    }
    this.weights = new double[weights.length][];
    for (int outcome = 0; outcome < weights.length; outcome++) {
      double[] row = Objects.requireNonNull(weights[outcome], "weight row");
      if (row.length != width) {
        throw new IllegalArgumentException(
            "every weight row must be " + width + " wide, row " + outcome + " is " + row.length);
      }
      this.weights[outcome] = row.clone();
    }
    this.bias = bias.clone();
  }

  /** The internal weights, uncopied, for the artifact writer only. Never hand this to a caller. */
  double[][] weightsInternal() {
    return weights;
  }

  /** The internal bias, uncopied, for the artifact writer only. Never hand this to a caller. */
  double[] biasInternal() {
    return bias;
  }

  /** Returns the width of hidden state this head reads. */
  public int width() {
    return width;
  }

  @Override
  public AnswerSpace space() {
    return space;
  }

  @Override
  public double[] logits(float[] hiddenState) {
    Objects.requireNonNull(hiddenState, "hiddenState");
    if (hiddenState.length != width) {
      throw new IllegalArgumentException(
          "hidden state must be " + width + " wide, got " + hiddenState.length);
    }
    for (float value : hiddenState) {
      if (!Float.isFinite(value)) {
        throw new IllegalArgumentException("hidden state must be finite");
      }
    }
    double[] scores = new double[weights.length];
    for (int outcome = 0; outcome < weights.length; outcome++) {
      double[] row = weights[outcome];
      double total = bias[outcome];
      for (int index = 0; index < width; index++) {
        total += row[index] * hiddenState[index];
      }
      scores[outcome] = total;
    }
    return scores;
  }
}
