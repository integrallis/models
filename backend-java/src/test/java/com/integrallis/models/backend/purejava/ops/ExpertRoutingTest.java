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
package com.integrallis.models.backend.purejava.ops;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The two expert-routing modes, and the difference between them.
 *
 * <p>Both pick the same experts; they differ only in whether the chosen weights are renormalised.
 * llama.cpp selects between them on {@code expert_weights_norm}: Qwen3-MoE and Qwen3.5-MoE set it,
 * DeepSeek-V2 publishes no such key and so takes the unnormalised form. Using one where the other
 * belongs scales every routed contribution by a factor that varies per token -- plausible output,
 * quietly wrong -- so the distinction is pinned here rather than left to a decoder's own tests.
 */
@Tag("unit")
class ExpertRoutingTest {

  private static final float TOLERANCE = 1.0e-6f;

  @Test
  void theRenormalisedWeightsSumToOne() {
    float[] logits = {0.0f, 2.0f, 1.0f, -1.0f};
    int[] selected = new int[2];
    float[] weights = new float[2];

    ExpertRouting.selectExperts(logits, 4, 2, selected, weights);

    assertThat(selected).containsExactly(1, 2);
    assertThat(weights[0] + weights[1]).isEqualTo(1.0f, within(TOLERANCE));
    // softmax over {2, 1} alone: e/(e+1) and 1/(e+1).
    assertThat(weights[0]).isEqualTo(0.7310586f, within(1.0e-5f));
  }

  /**
   * The unnormalised weights are a softmax over <b>all</b> experts, so they sum to less than one.
   *
   * <p>{@code softmax(0, 2, 1, -1)} is {@code (0.0871443, 0.6439143, 0.2368828, 0.0320586)}; the
   * top two sum to 0.8807971, not 1. A renormalised reading would return 0.7310586 and 0.2689414.
   *
   * <p>Those numbers are computed, not estimated. The first version of this test carried
   * hand-arithmetic wrong in the third decimal place, and it failed against a correct
   * implementation -- which is the argument for computing an expected value rather than reasoning
   * one out.
   */
  @Test
  void theUnnormalisedWeightsAreAFullWidthSoftmaxAndSumToLessThanOne() {
    float[] logits = {0.0f, 2.0f, 1.0f, -1.0f};
    int[] selected = new int[2];
    float[] weights = new float[2];

    ExpertRouting.selectExpertsWithoutRenormalisation(logits, 4, 2, selected, weights);

    assertThat(selected).containsExactly(1, 2);
    assertThat(weights[0]).isEqualTo(0.6439143f, within(1.0e-5f));
    assertThat(weights[1]).isEqualTo(0.2368828f, within(1.0e-5f));
    assertThat(weights[0] + weights[1])
        .describedAs("the unchosen experts keep their mass")
        .isEqualTo(0.8807971f, within(1.0e-5f));
  }

  /** The two modes must disagree, or one of them is not doing what it says. */
  @Test
  void theTwoModesDisagreeOnTheWeightsWhileAgreeingOnTheExperts() {
    float[] first = {0.0f, 2.0f, 1.0f, -1.0f};
    float[] second = first.clone();
    int[] selectedFirst = new int[2];
    int[] selectedSecond = new int[2];
    float[] renormalised = new float[2];
    float[] raw = new float[2];

    ExpertRouting.selectExperts(first, 4, 2, selectedFirst, renormalised);
    ExpertRouting.selectExpertsWithoutRenormalisation(second, 4, 2, selectedSecond, raw);

    assertThat(selectedSecond).containsExactly(selectedFirst);
    assertThat(raw[0]).isNotEqualTo(renormalised[0]);
    // And the ratio between the two weights is preserved: only the total differs.
    assertThat(raw[0] / raw[1]).isEqualTo(renormalised[0] / renormalised[1], within(1.0e-4f));
  }

  /** Selecting every expert makes the unnormalised form a plain full-width softmax. */
  @Test
  void selectingEveryExpertSumsToOneEvenUnnormalised() {
    float[] logits = {0.0f, 2.0f, 1.0f, -1.0f};
    int[] selected = new int[4];
    float[] weights = new float[4];

    ExpertRouting.selectExpertsWithoutRenormalisation(logits, 4, 4, selected, weights);

    float total = 0.0f;
    for (float weight : weights) {
      total += weight;
    }
    assertThat(total).isEqualTo(1.0f, within(1.0e-5f));
    assertThat(selected).containsExactly(1, 2, 0, 3);
  }

  @Test
  void bothModesReturnTheExpertsInDescendingOrder() {
    float[] logits = {-3.0f, 5.0f, 0.5f, 4.0f, 1.5f};
    int[] selected = new int[3];
    float[] weights = new float[3];

    ExpertRouting.selectExperts(logits.clone(), 5, 3, selected, weights);
    assertThat(selected).containsExactly(1, 3, 4);
    assertThat(weights[0]).isGreaterThan(weights[1]);
    assertThat(weights[1]).isGreaterThan(weights[2]);

    ExpertRouting.selectExpertsWithoutRenormalisation(logits.clone(), 5, 3, selected, weights);
    assertThat(selected).containsExactly(1, 3, 4);
    assertThat(weights[0]).isGreaterThan(weights[1]);
    assertThat(weights[1]).isGreaterThan(weights[2]);
  }

  @Test
  void aTieKeepsTheLowerExpertIndexInBothModes() {
    int[] selected = new int[1];
    float[] weights = new float[1];

    ExpertRouting.selectExperts(new float[] {1.0f, 1.0f, 0.0f}, 3, 1, selected, weights);
    assertThat(selected[0]).isZero();

    ExpertRouting.selectExpertsWithoutRenormalisation(
        new float[] {1.0f, 1.0f, 0.0f}, 3, 1, selected, weights);
    assertThat(selected[0]).isZero();
  }
}
