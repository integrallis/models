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
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The head is the whole of the decision: one read of a hidden state the base already computed, and
 * no decode loop anywhere. These tests hold it to exact arithmetic and to reproducibility, because
 * a decision that moves between runs cannot be calibrated.
 */
@Tag("unit")
class LinearDecisionHeadTest {

  private static final Choice ROUTE = new Choice("route", List.of("chat", "tools"));

  @Test
  void theHeadIsAnAffineReadOfTheHiddenState() {
    LinearDecisionHead head =
        new LinearDecisionHead(
            ROUTE, new double[][] {{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}}, new double[] {0.0, 0.0});

    double[] logits = head.logits(new float[] {0.25f, 0.75f, 9.0f});

    assertThat(logits[0]).isCloseTo(0.25, within(1e-12));
    assertThat(logits[1]).isCloseTo(0.75, within(1e-12));
  }

  @Test
  void theBiasIsAddedToEveryOutcome() {
    LinearDecisionHead head =
        new LinearDecisionHead(
            ROUTE, new double[][] {{1.0, 0.0}, {0.0, 1.0}}, new double[] {0.5, -0.5});

    double[] logits = head.logits(new float[] {1.0f, 1.0f});

    assertThat(logits[0]).isCloseTo(1.5, within(1e-12));
    assertThat(logits[1]).isCloseTo(0.5, within(1e-12));
  }

  @Test
  void theHeadMustHaveOneRowPerOutcomeOfItsSpace() {
    assertThatThrownBy(
            () ->
                new LinearDecisionHead(
                    new Choice("route", List.of("chat", "tools", "refuse")),
                    new double[][] {{1.0}, {0.0}},
                    new double[] {0.0, 0.0}))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void theHeadRejectsAHiddenStateOfTheWrongWidth() {
    LinearDecisionHead head =
        new LinearDecisionHead(
            ROUTE, new double[][] {{1.0, 0.0}, {0.0, 1.0}}, new double[] {0.0, 0.0});

    assertThatThrownBy(() -> head.logits(new float[] {1.0f, 1.0f, 1.0f}))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void theHeadRefusesANonFiniteHiddenStateRatherThanPropagatingIt() {
    LinearDecisionHead head =
        new LinearDecisionHead(
            ROUTE, new double[][] {{1.0, 0.0}, {0.0, 1.0}}, new double[] {0.0, 0.0});

    assertThatThrownBy(() -> head.logits(new float[] {Float.NaN, 1.0f}))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void everyRowMustBeTheSameWidth() {
    assertThatThrownBy(
            () ->
                new LinearDecisionHead(
                    ROUTE, new double[][] {{1.0, 0.0}, {0.0}}, new double[] {0.0, 0.0}))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void readingTheSameHiddenStateTwiceGivesBitIdenticalLogits() {
    LinearDecisionHead head =
        new LinearDecisionHead(
            ROUTE,
            new double[][] {{0.31, -0.22, 0.7}, {0.11, 0.93, -0.4}},
            new double[] {0.1, 0.2});
    float[] hidden = {0.25f, 0.75f, -0.5f};

    assertThat(head.logits(hidden)).isEqualTo(head.logits(hidden));
  }

  @Test
  void theHeadDecidesIntoItsOwnAnswerSpace() {
    LinearDecisionHead head =
        new LinearDecisionHead(
            ROUTE, new double[][] {{1.0, 0.0}, {0.0, 1.0}}, new double[] {0.0, 0.0});

    Verdict verdict = head.decide(new float[] {0.0f, 4.0f}, 1.0);

    assertThat(verdict.winner()).isEqualTo("tools");
    assertThat(ROUTE.labels()).contains(verdict.winner());
    assertThat(verdict.probabilityOf("chat") + verdict.probabilityOf("tools"))
        .isCloseTo(1.0, within(1e-12));
  }

  @Test
  void theHeadCopiesTheWeightsItWasGiven() {
    double[][] weights = {{1.0, 0.0}, {0.0, 1.0}};
    double[] bias = {0.0, 0.0};
    LinearDecisionHead head = new LinearDecisionHead(ROUTE, weights, bias);

    weights[0][0] = 99.0;
    bias[0] = 99.0;

    assertThat(head.logits(new float[] {1.0f, 0.0f})[0]).isCloseTo(1.0, within(1e-12));
  }
}
