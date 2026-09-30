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
 * A verdict is the only thing a decision may return. It is bound to the space that produced it, so
 * the winner is always a declared label and the distribution always covers exactly that space.
 */
@Tag("unit")
class VerdictTest {

  private static final Choice ROUTE = new Choice("route", List.of("chat", "tools", "refuse"));

  @Test
  void theWinnerIsAlwaysALabelOfItsOwnSpace() {
    Verdict verdict = new Verdict(ROUTE, new double[] {0.2, 0.7, 0.1});

    assertThat(verdict.winner()).isEqualTo("tools");
    assertThat(ROUTE.labels()).contains(verdict.winner());
  }

  @Test
  void aVerdictReportsTheProbabilityOfAnyLabelItCovers() {
    Verdict verdict = new Verdict(ROUTE, new double[] {0.2, 0.7, 0.1});

    assertThat(verdict.probabilityOf("chat")).isEqualTo(0.2);
    assertThat(verdict.probabilityOf("refuse")).isEqualTo(0.1);
    assertThatThrownBy(() -> verdict.probabilityOf("absent"))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void aNoulReadsAsTheProbabilityItsPropositionHolds() {
    Verdict verdict = new Verdict(new Noul("answerable"), new double[] {0.25, 0.75});

    assertThat(verdict.probabilityOfTrue()).isEqualTo(0.75);
    assertThat(verdict.winner()).isEqualTo("true");
  }

  @Test
  void onlyANoulCanBeReadAsATruthProbability() {
    Verdict verdict = new Verdict(ROUTE, new double[] {0.2, 0.7, 0.1});

    assertThatThrownBy(verdict::probabilityOfTrue).isInstanceOf(IllegalStateException.class);
  }

  @Test
  void confidenceIsOneWhenTheDistributionIsCertainAndZeroWhenItIsUniform() {
    assertThat(new Verdict(ROUTE, new double[] {0.0, 1.0, 0.0}).confidence())
        .isCloseTo(1.0, within(1e-12));

    double third = 1.0 / 3.0;
    assertThat(new Verdict(ROUTE, new double[] {third, third, third}).confidence())
        .isCloseTo(0.0, within(1e-9));
  }

  @Test
  void confidenceFallsAsTheDistributionFlattens() {
    double sharp = new Verdict(ROUTE, new double[] {0.05, 0.90, 0.05}).confidence();
    double blunt = new Verdict(ROUTE, new double[] {0.25, 0.50, 0.25}).confidence();

    assertThat(sharp).isGreaterThan(blunt);
  }

  @Test
  void aDistributionMustCoverExactlyItsSpace() {
    assertThatThrownBy(() -> new Verdict(ROUTE, new double[] {0.5, 0.5}))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void aDistributionMustSumToOne() {
    assertThatThrownBy(() -> new Verdict(ROUTE, new double[] {0.2, 0.2, 0.2}))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void aDistributionRejectsNegativeAndNonFiniteMass() {
    assertThatThrownBy(() -> new Verdict(ROUTE, new double[] {-0.1, 0.6, 0.5}))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> new Verdict(ROUTE, new double[] {Double.NaN, 0.5, 0.5}))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void aVerdictCopiesTheDistributionItWasGiven() {
    double[] mutable = {0.2, 0.7, 0.1};
    Verdict verdict = new Verdict(ROUTE, mutable);

    mutable[0] = 0.9;

    assertThat(verdict.probabilityOf("chat")).isEqualTo(0.2);
    assertThat(verdict.probabilities()[0]).isEqualTo(0.2);
  }

  @Test
  void tiesResolveToTheFirstDeclaredLabelSoAVerdictIsReproducible() {
    Verdict verdict = new Verdict(ROUTE, new double[] {0.5, 0.5, 0.0});

    assertThat(verdict.winner()).isEqualTo("chat");
  }
}
