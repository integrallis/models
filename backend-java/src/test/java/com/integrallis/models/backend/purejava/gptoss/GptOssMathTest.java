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
package com.integrallis.models.backend.purejava.gptoss;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class GptOssMathTest {

  @Test
  void routingMatchesFreeTokenSoftmaxTopKRenormalization() {
    int[] selected = new int[4];
    float[] weights = new float[4];

    GptOssMath.selectExperts(
        new float[] {0.25f, -1.0f, 2.0f, 1.25f, 0.5f, -0.75f}, selected, weights);

    assertThat(selected).containsExactly(2, 3, 4, 0);
    assertThat(weights)
        .containsExactly(
            new float[] {0.534968f, 0.252701f, 0.1193675f, 0.0929635f}, within(0.0001f));
    assertThat(weights[0] + weights[1] + weights[2] + weights[3]).isCloseTo(1.0f, within(1.0e-6f));
  }

  @Test
  void routingBreaksEqualLogitsByExpertIndex() {
    int[] selected = new int[3];
    float[] weights = new float[3];

    GptOssMath.selectExperts(new float[] {1.0f, 2.0f, 2.0f, 2.0f}, selected, weights);

    assertThat(selected).containsExactly(1, 2, 3);
    assertThat(weights).containsExactly(new float[] {1.0f / 3, 1.0f / 3, 1.0f / 3});
  }

  @Test
  void activationMatchesFreeTokenInterleavingAndClampEquation() {
    float[] output = new float[5];

    GptOssMath.swigluOai(
        new float[] {-9.0f, -9.0f, -1.0f, -2.0f, 0.0f, 0.5f, 2.0f, 4.0f, 9.0f, 9.0f},
        output,
        1.702f,
        7.0f);

    assertThat(output)
        .containsExactly(
            new float[] {1.2019068e-5f, 0.15420423f, 0.0f, 9.678293f, 55.999626f}, within(0.0001f));
  }

  @Test
  void rejectsInvalidRoutingAndActivationBuffers() {
    assertThatThrownBy(
            () -> GptOssMath.selectExperts(new float[] {1.0f, Float.NaN}, new int[1], new float[1]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("routerLogits[1]");
    assertThatThrownBy(() -> GptOssMath.selectExperts(new float[] {1.0f}, new int[0], new float[0]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("top-k");
    assertThatThrownBy(() -> GptOssMath.swigluOai(new float[3], new float[2], 1.702f, 7.0f))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("twice");
    assertThatThrownBy(() -> GptOssMath.swigluOai(new float[4], new float[2], 1.702f, Float.NaN))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("limit");
  }
}
