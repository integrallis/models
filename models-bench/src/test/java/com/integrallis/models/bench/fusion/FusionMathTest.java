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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.Test;

class FusionMathTest {

  @Test
  void logSoftmaxIsShiftInvariantAndNormalized() {
    float[] logits = {1.0f, 2.0f, 3.0f};
    float[] shifted = {101.0f, 102.0f, 103.0f};
    double[] a = FusionMath.logSoftmax(logits);
    double[] b = FusionMath.logSoftmax(shifted);
    double mass = 0;
    for (int i = 0; i < a.length; i++) {
      assertThat(a[i]).isCloseTo(b[i], within(1e-9));
      mass += Math.exp(a[i]);
    }
    assertThat(mass).isCloseTo(1.0, within(1e-12));
    assertThat(a[2]).isCloseTo(3 - Math.log(Math.exp(1) + Math.exp(2) + Math.exp(3)), within(1e-9));
  }

  @Test
  void productOfExpertsIsWeightedSumOfLogProbabilities() {
    double[][] logProbs = {
      FusionMath.logSoftmax(new float[] {0, 1}), FusionMath.logSoftmax(new float[] {2, 0})
    };
    double[] fused =
        FusionMath.combine(
            FusionRule.POE, new float[][] {{0, 1}, {2, 0}}, logProbs, new double[] {0.25, 0.75});
    for (int t = 0; t < 2; t++) {
      assertThat(fused[t]).isCloseTo(0.25 * logProbs[0][t] + 0.75 * logProbs[1][t], within(1e-12));
    }
  }

  @Test
  void mixtureIsLogOfWeightedProbabilityMixture() {
    float[][] logits = {{0, 1, -1}, {2, 0, 0}};
    double[][] logProbs = {FusionMath.logSoftmax(logits[0]), FusionMath.logSoftmax(logits[1])};
    double[] fused =
        FusionMath.combine(FusionRule.MIXTURE, logits, logProbs, new double[] {0.4, 0.6});
    for (int t = 0; t < 3; t++) {
      double expected = Math.log(0.4 * Math.exp(logProbs[0][t]) + 0.6 * Math.exp(logProbs[1][t]));
      assertThat(fused[t]).isCloseTo(expected, within(1e-12));
    }
  }

  @Test
  void mixtureIgnoresZeroWeightMembersEvenWithExtremeLogits() {
    float[][] logits = {{0, 1}, {-1e30f, 1e30f}};
    double[][] logProbs = {FusionMath.logSoftmax(logits[0]), FusionMath.logSoftmax(logits[1])};
    double[] fused = FusionMath.combine(FusionRule.MIXTURE, logits, logProbs, new double[] {1, 0});
    assertThat(fused[0]).isCloseTo(logProbs[0][0], within(1e-12));
    assertThat(fused[1]).isCloseTo(logProbs[0][1], within(1e-12));
  }

  @Test
  void articleRuleSumsRawLogitsSoPerMemberOffsetsLeakIn() {
    float[][] logits = {{0, 1}, {100, 100.5f}};
    double[][] logProbs = {FusionMath.logSoftmax(logits[0]), FusionMath.logSoftmax(logits[1])};
    double[] fused =
        FusionMath.combine(FusionRule.ARTICLE, logits, logProbs, new double[] {0.5, 0.5});
    assertThat(fused[0]).isCloseTo(50.0, within(1e-9));
    assertThat(fused[1]).isCloseTo(0.5 + 50.25, within(1e-9));
  }

  @Test
  void entropyAndKlBehaveOnKnownDistributions() {
    double[] uniform = {Math.log(0.5), Math.log(0.5)};
    double[] skewed = {Math.log(0.9), Math.log(0.1)};
    assertThat(FusionMath.entropy(uniform)).isCloseTo(Math.log(2), within(1e-12));
    assertThat(FusionMath.klDivergence(uniform, uniform)).isCloseTo(0, within(1e-12));
    double expected = 0.5 * Math.log(0.5 / 0.9) + 0.5 * Math.log(0.5 / 0.1);
    assertThat(FusionMath.klDivergence(uniform, skewed)).isCloseTo(expected, within(1e-12));
  }

  @Test
  void argmaxPrefersLowestIdOnTies() {
    assertThat(FusionMath.argmax(new double[] {1, 3, 3})).isEqualTo(1);
    assertThat(FusionMath.argmax(new float[] {5, 5})).isEqualTo(0);
  }

  @Test
  void rejectsMismatchedVocabulariesAndWeights() {
    double[][] logProbs = {{0, 0}, {0, 0, 0}};
    assertThatThrownBy(
            () ->
                FusionMath.combine(
                    FusionRule.POE,
                    new float[][] {{0, 0}, {0, 0, 0}},
                    logProbs,
                    new double[] {0.5, 0.5}))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("vocabulary");
    assertThatThrownBy(() -> FusionMath.validateWeights(new double[] {0.5, 0.6}))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("simplex");
    assertThatThrownBy(() -> FusionMath.validateWeights(new double[] {1.5, -0.5}))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void simplexGridEnumeratesEveryPointOnceAtStepTenth() {
    var twoMembers = FusionMath.simplexGrid(2, 10);
    var threeMembers = FusionMath.simplexGrid(3, 10);
    assertThat(twoMembers).hasSize(11);
    assertThat(threeMembers).hasSize(66);
    for (double[] point : threeMembers) {
      double sum = 0;
      for (double w : point) {
        sum += w;
      }
      assertThat(sum).isCloseTo(1.0, within(1e-12));
    }
  }
}
