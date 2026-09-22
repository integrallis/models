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

import java.util.List;
import java.util.random.RandomGenerator;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Fitting a head and fitting its temperature, on problems whose right answer is known by
 * construction. Both must be deterministic: the pre-registration treats agreement and calibration
 * error as exact functions of the inputs rather than as draws, and that is only true if fitting
 * returns the same weights every time.
 */
@Tag("unit")
class HeadTrainingTest {

  private static final Noul ANSWERABLE = new Noul("the state answers the question");

  @Test
  void theTrainerSeparatesALinearlySeparableProblem() {
    float[][] features = new float[200][1];
    int[] labels = new int[200];
    for (int index = 0; index < 200; index++) {
      double x = (index - 100) / 25.0;
      features[index][0] = (float) x;
      labels[index] = x > 0 ? 1 : 0;
    }

    LinearDecisionHead head = trainer().fit(ANSWERABLE, features, labels);

    assertThat(head.decide(new float[] {3.0f}, 1.0).probabilityOfTrue()).isGreaterThan(0.9);
    assertThat(head.decide(new float[] {-3.0f}, 1.0).probabilityOfTrue()).isLessThan(0.1);
  }

  @Test
  void trainingTwiceOnTheSameDataGivesBitIdenticalWeights() {
    float[][] features = randomFeatures(64, 8, 11L);
    int[] labels = alternating(64);

    double[] first = trainer().fit(ANSWERABLE, features, labels).logits(new float[8]);
    double[] second = trainer().fit(ANSWERABLE, features, labels).logits(new float[8]);

    assertThat(first).isEqualTo(second);
  }

  @Test
  void theFittedHeadScoresTheSpaceItWasAskedFor() {
    Choice route = new Choice("route", List.of("chat", "tools", "refuse"));
    float[][] features = randomFeatures(60, 4, 5L);
    int[] labels = new int[60];
    for (int index = 0; index < 60; index++) {
      labels[index] = index % 3;
    }

    LinearDecisionHead head = trainer().fit(route, features, labels);

    assertThat(head.space()).isEqualTo(route);
    assertThat(head.logits(new float[4])).hasSize(3);
    assertThat(route.labels()).contains(head.decide(new float[4], 1.0).winner());
  }

  @Test
  void strongerRegularisationShrinksTheWeights() {
    float[][] features = randomFeatures(80, 6, 3L);
    int[] labels = alternating(80);

    double loose =
        magnitude(new LogisticHeadTrainer(300, 0.1, 0.0).fit(ANSWERABLE, features, labels));
    double tight =
        magnitude(new LogisticHeadTrainer(300, 0.1, 5.0).fit(ANSWERABLE, features, labels));

    assertThat(tight).isLessThan(loose);
  }

  @Test
  void aDecayThatWouldDivergeIsRefusedAtConstruction() {
    // The weight update subtracts learningRate * l2 * w each step, so the decay factor is
    // (1 - learningRate * l2). At or above 1 the weights flip sign and grow without bound, and the
    // first thing the caller sees is a non-finite logit from deep inside the fit. Refuse the
    // combination up front, where the message can say what is wrong.
    assertThatThrownBy(() -> new LogisticHeadTrainer(400, 0.1, 10.0))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("diverge");
    assertThatThrownBy(() -> new LogisticHeadTrainer(400, 0.1, 100.0))
        .isInstanceOf(IllegalArgumentException.class);
    // Just under the boundary is allowed.
    assertThat(new LogisticHeadTrainer(400, 0.1, 9.0)).isNotNull();
  }

  @Test
  void theTrainerRejectsEvidenceItCannotUse() {
    assertThatThrownBy(() -> trainer().fit(ANSWERABLE, new float[0][0], new int[0]))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(
            () -> trainer().fit(ANSWERABLE, new float[][] {{1.0f}, {1.0f}}, new int[] {0}))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(
            () -> trainer().fit(ANSWERABLE, new float[][] {{1.0f}, {1.0f}}, new int[] {0, 7}))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void aPerfectlyCalibratedSetLeavesTheTemperatureAlone() {
    double[][] logits = new double[400][];
    int[] labels = new int[400];
    RandomGenerator random = RandomGenerator.of("L64X128MixRandom");
    for (int index = 0; index < 400; index++) {
      double score = (index % 40) / 10.0 - 2.0;
      logits[index] = new double[] {0.0, score};
      double truth = 1.0 / (1.0 + Math.exp(-score));
      labels[index] = random.nextDouble() < truth ? 1 : 0;
    }

    assertThat(TemperatureFitter.fit(logits, labels)).isCloseTo(1.0, within(0.35));
  }

  @Test
  void anOverconfidentSetIsCooledAndAnUnderconfidentSetIsSharpened() {
    double[][] overconfident = new double[200][];
    double[][] underconfident = new double[200][];
    int[] labels = new int[200];
    for (int index = 0; index < 200; index++) {
      int label = index % 4 == 0 ? 0 : 1;
      labels[index] = label;
      overconfident[index] = new double[] {0.0, 12.0};
      underconfident[index] = new double[] {0.0, 0.02};
    }

    assertThat(TemperatureFitter.fit(overconfident, labels)).isGreaterThan(1.0);
    assertThat(TemperatureFitter.fit(underconfident, labels)).isLessThan(1.0);
  }

  @Test
  void theFittedTemperatureNeverScoresWorseThanNoScalingAtAll() {
    double[][] logits = new double[150][];
    int[] labels = new int[150];
    for (int index = 0; index < 150; index++) {
      labels[index] = index % 3 == 0 ? 0 : 1;
      logits[index] = new double[] {0.0, 6.0};
    }

    double fitted = TemperatureFitter.fit(logits, labels);

    assertThat(TemperatureFitter.negativeLogLikelihood(logits, labels, fitted))
        .isLessThanOrEqualTo(TemperatureFitter.negativeLogLikelihood(logits, labels, 1.0));
  }

  @Test
  void fittingTheSameEvidenceTwiceGivesTheIdenticalTemperature() {
    double[][] logits = new double[120][];
    int[] labels = new int[120];
    for (int index = 0; index < 120; index++) {
      labels[index] = index % 5 == 0 ? 0 : 1;
      logits[index] = new double[] {0.0, 3.5};
    }

    assertThat(TemperatureFitter.fit(logits, labels))
        .isEqualTo(TemperatureFitter.fit(logits, labels));
  }

  private static LogisticHeadTrainer trainer() {
    return new LogisticHeadTrainer(400, 0.1, 0.01);
  }

  private static double magnitude(LinearDecisionHead head) {
    double total = 0.0;
    for (double value : head.logits(new float[head.width()])) {
      total += Math.abs(value);
    }
    double[] probe = head.logits(unitVector(head.width()));
    for (double value : probe) {
      total += Math.abs(value);
    }
    return total;
  }

  private static float[] unitVector(int width) {
    float[] vector = new float[width];
    java.util.Arrays.fill(vector, 1.0f);
    return vector;
  }

  /** A fixed linear-congruential stream, so the fixtures are identical on every run and host. */
  private static float[][] randomFeatures(int rows, int width, long seed) {
    float[][] features = new float[rows][width];
    long state = seed;
    for (int row = 0; row < rows; row++) {
      for (int column = 0; column < width; column++) {
        state = state * 6364136223846793005L + 1442695040888963407L;
        features[row][column] = (float) (((state >>> 33) / (double) (1L << 31)) - 0.5);
      }
    }
    return features;
  }

  private static int[] alternating(int rows) {
    int[] labels = new int[rows];
    for (int index = 0; index < rows; index++) {
      labels[index] = index % 2;
    }
    return labels;
  }
}
