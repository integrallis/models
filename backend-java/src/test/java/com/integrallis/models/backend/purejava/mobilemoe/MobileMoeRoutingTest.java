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
package com.integrallis.models.backend.purejava.mobilemoe;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class MobileMoeRoutingTest {

  @Test
  void expertBiasChangesSelectionButNotTheRoutingWeights() {
    float[] logits = {0.0f, (float) Math.log(3.0), (float) -Math.log(3.0), (float) Math.log(4.0)};
    float[] expertBias = {0.0f, 0.0f, 1.0f, 0.0f};
    int[] selected = new int[2];
    float[] weights = new float[2];

    MobileMoeRouting.select(logits, expertBias, 2, 2.5f, selected, weights);

    assertThat(selected).containsExactly(2, 3);
    assertThat(weights[0]).isCloseTo(0.5952381f, within(1.0e-6f));
    assertThat(weights[1]).isCloseTo(1.9047619f, within(1.0e-6f));
    assertThat(weights[0] + weights[1]).isCloseTo(2.5f, within(1.0e-6f));
  }

  @Test
  void stableSigmoidHandlesExtremeRouterLogits() {
    assertThat(MobileMoeRouting.sigmoid(1_000.0f)).isEqualTo(1.0f);
    assertThat(MobileMoeRouting.sigmoid(-1_000.0f)).isEqualTo(0.0f);
  }

  @Test
  void rejectsInvalidRoutingGeometry() {
    assertThatThrownBy(
            () ->
                MobileMoeRouting.select(
                    new float[] {0.0f}, new float[0], 1, 2.5f, new int[1], new float[1]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("expert bias");
    assertThatThrownBy(
            () ->
                MobileMoeRouting.select(
                    new float[] {0.0f}, new float[] {0.0f}, 2, 2.5f, new int[2], new float[2]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("topK");
  }
}
