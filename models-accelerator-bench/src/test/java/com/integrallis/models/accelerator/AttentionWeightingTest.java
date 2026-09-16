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
package com.integrallis.models.accelerator;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.data.Offset.offset;

import org.junit.jupiter.api.Test;

class AttentionWeightingTest {

  @Test
  void keepsFullAttentionContrastWhenEffectiveSampleSizeIsSufficient() {
    double[] weights = AttentionWeighting.shrinkToEffectiveSampleSize(new double[] {0, 1, 1, 2}, 1);

    assertThat(weights).containsExactly(0, 1, 1, 2);
    assertThat(AttentionWeighting.effectiveSampleSize(weights)).isEqualTo(8.0 / 3.0);
  }

  @Test
  void shrinksTowardUniformWhenAFullContrastWouldBeUnderdetermined() {
    double[] weights =
        AttentionWeighting.shrinkToEffectiveSampleSize(new double[] {0, 0, 0, 4}, 2);

    for (double weight : weights) {
      assertThat(weight).isPositive().isFinite();
    }
    assertThat(weights).hasSize(4);
    assertThat(java.util.Arrays.stream(weights).average().orElseThrow())
        .isCloseTo(1.0, offset(1.0e-12));
    assertThat(AttentionWeighting.effectiveSampleSize(weights)).isCloseTo(4.0, offset(1.0e-10));
  }

  @Test
  void rejectsNonFiniteOrNegativeSensitivities() {
    assertThatThrownBy(
            () -> AttentionWeighting.shrinkToEffectiveSampleSize(new double[] {1, -1}, 1))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("nonnegative");
    assertThatThrownBy(
            () ->
                AttentionWeighting.shrinkToEffectiveSampleSize(
                    new double[] {1, Double.NaN}, 1))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("finite");
  }
}
