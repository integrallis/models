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
package com.integrallis.models.backend.purejava.flint;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class FlintRoutingTest {
  @Test
  void biasSelectsExpertsButDoesNotWeightTheirContributions() {
    int[] selected = new int[2];
    float[] weights = new float[2];
    // Sigmoids are 0.5, 0.75, 0.25. Bias promotes the third expert above the second.
    FlintRouting.select(
        new float[] {0f, (float) Math.log(3), -(float) Math.log(3)},
        new float[] {0f, -1f, 1f},
        selected,
        weights,
        true,
        2f);
    assertThat(selected).containsExactly(2, 0);
    assertThat(weights[0]).isCloseTo(2f / 3f, within(1e-6f));
    assertThat(weights[1]).isCloseTo(4f / 3f, within(1e-6f));
  }

  @Test
  void unnormalizedWeightsAndTiesRetainTheirDefinedMeaning() {
    int[] selected = new int[2];
    float[] weights = new float[2];
    FlintRouting.select(new float[] {0f, 0f, 0f}, new float[3], selected, weights, false, 1f);
    assertThat(selected).containsExactly(0, 1);
    assertThat(weights).containsExactly(0.5f, 0.5f);
    FlintRouting.select(
        new float[] {-1000f, -1000f, -1000f}, new float[3], selected, weights, true, 1f);
    assertThat(weights).containsExactly(0f, 0f);
  }
}
