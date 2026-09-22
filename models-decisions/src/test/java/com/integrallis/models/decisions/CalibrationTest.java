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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Calibration is the claim that makes a probability worth reading, so every figure here is checked
 * against a case whose answer is known by hand rather than by running a model.
 */
@Tag("unit")
class CalibrationTest {

  @Test
  void temperatureOneIsAPlainSoftmax() {
    double[] scaled = Calibration.softmax(new double[] {1.0, 2.0, 3.0}, 1.0);

    double denominator = Math.exp(1.0) + Math.exp(2.0) + Math.exp(3.0);
    assertThat(scaled[0]).isCloseTo(Math.exp(1.0) / denominator, within(1e-12));
    assertThat(scaled[2]).isCloseTo(Math.exp(3.0) / denominator, within(1e-12));
  }

  @Test
  void aDistributionAlwaysSumsToOne() {
    for (double temperature : new double[] {0.1, 0.5, 1.0, 2.0, 50.0}) {
      double[] scaled = Calibration.softmax(new double[] {-3.0, 0.25, 7.0}, temperature);
      assertThat(sum(scaled)).isCloseTo(1.0, within(1e-12));
    }
  }

  @Test
  void raisingTheTemperatureFlattensAndLoweringItSharpens() {
    double[] logits = {1.0, 2.0, 3.0};

    double cold = Calibration.softmax(logits, 0.5)[2];
    double neutral = Calibration.softmax(logits, 1.0)[2];
    double hot = Calibration.softmax(logits, 10.0)[2];

    assertThat(cold).isGreaterThan(neutral);
    assertThat(neutral).isGreaterThan(hot);
    assertThat(hot).isCloseTo(1.0 / 3.0, within(0.05));
  }

  @Test
  void temperatureNeverMovesTheWinner() {
    double[] logits = {0.4, 2.2, 1.9, -5.0};

    for (double temperature : new double[] {0.05, 0.5, 1.0, 3.0, 100.0}) {
      assertThat(argmax(Calibration.softmax(logits, temperature))).isEqualTo(1);
    }
  }

  @Test
  void largeLogitsDoNotOverflowIntoNothing() {
    double[] scaled = Calibration.softmax(new double[] {1000.0, 1001.0}, 1.0);

    assertThat(scaled[0]).isFinite();
    assertThat(scaled[1]).isFinite();
    assertThat(sum(scaled)).isCloseTo(1.0, within(1e-12));
    assertThat(scaled[1]).isGreaterThan(scaled[0]);
  }

  @Test
  void aTemperatureMustBePositiveAndFinite() {
    assertThatThrownBy(() -> Calibration.softmax(new double[] {1.0, 2.0}, 0.0))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Calibration.softmax(new double[] {1.0, 2.0}, -1.0))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> Calibration.softmax(new double[] {1.0, 2.0}, Double.NaN))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void scalingTheSameLogitsTwiceGivesBitIdenticalProbabilities() {
    double[] logits = {0.4, 2.2, 1.9, -5.0};

    assertThat(Calibration.softmax(logits, 1.7)).isEqualTo(Calibration.softmax(logits, 1.7));
  }

  @Test
  void calibrationErrorIsTheGapBetweenConfidenceAndObservedAccuracy() {
    double[] confidences = {0.9, 0.9, 0.9, 0.9};
    boolean[] correct = {true, true, true, false};

    assertThat(Calibration.expectedCalibrationError(confidences, correct, 10))
        .isCloseTo(0.15, within(1e-12));
  }

  @Test
  void aPerfectlyCalibratedSetScoresZero() {
    double[] confidences = {0.5, 0.5, 1.0, 1.0};
    boolean[] correct = {true, false, true, true};

    assertThat(Calibration.expectedCalibrationError(confidences, correct, 10))
        .isCloseTo(0.0, within(1e-12));
  }

  @Test
  void eachBinIsWeightedByHowManyDecisionsLandedInIt() {
    double[] confidences = {0.95, 0.95, 0.95, 0.55};
    boolean[] correct = {true, true, true, true};

    assertThat(Calibration.expectedCalibrationError(confidences, correct, 10))
        .isCloseTo(0.75 * 0.05 + 0.25 * 0.45, within(1e-12));
  }

  @Test
  void brierScoresAConfidentRightAnswerAtZeroAndAConfidentWrongOneAtOne() {
    assertThat(Calibration.brierScore(new double[] {1.0}, new boolean[] {true}))
        .isCloseTo(0.0, within(1e-12));
    assertThat(Calibration.brierScore(new double[] {1.0}, new boolean[] {false}))
        .isCloseTo(1.0, within(1e-12));
    assertThat(Calibration.brierScore(new double[] {0.5}, new boolean[] {true}))
        .isCloseTo(0.25, within(1e-12));
  }

  @Test
  void calibrationMeasuresRejectRaggedOrEmptyEvidence() {
    assertThatThrownBy(
            () -> Calibration.expectedCalibrationError(new double[] {0.5}, new boolean[] {}, 10))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(
            () -> Calibration.expectedCalibrationError(new double[] {}, new boolean[] {}, 10))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(
            () -> Calibration.expectedCalibrationError(new double[] {0.5}, new boolean[] {true}, 0))
        .isInstanceOf(IllegalArgumentException.class);
  }

  private static double sum(double[] values) {
    double total = 0.0;
    for (double value : values) {
      total += value;
    }
    return total;
  }

  private static int argmax(double[] values) {
    int best = 0;
    for (int index = 1; index < values.length; index++) {
      if (values[index] > values[best]) {
        best = index;
      }
    }
    return best;
  }
}
