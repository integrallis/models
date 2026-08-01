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
package com.integrallis.models.backend.purejava.gemma4;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.data.Offset.offset;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class Gemma4MathTest {

  @Test
  void normalizesAnUnweightedHeadInPlaceWithoutTouchingAdjacentValues() {
    float[] values = {99.0f, 3.0f, 4.0f, 98.0f};

    Gemma4Math.normalizeWithoutWeight(values, 1, values, 1, 2, 1.0e-6f);

    assertThat(values[0]).isEqualTo(99.0f);
    assertThat(values[1]).isCloseTo(0.8485281f, offset(1.0e-6f));
    assertThat(values[2]).isCloseTo(1.1313708f, offset(1.0e-6f));
    assertThat(values[3]).isEqualTo(98.0f);
  }

  @Test
  void routerInputUsesUnweightedRmsNormAndTheLearnedDimensionScale() {
    float[] output = new float[2];

    Gemma4Math.normalizeRouterInput(
        output, new float[] {3.0f, 4.0f}, new float[] {2.0f, 0.5f}, 1.0e-6f);

    assertThat(output[0]).isCloseTo(1.1999999f, offset(1.0e-6f));
    assertThat(output[1]).isCloseTo(0.4f, offset(1.0e-6f));
  }

  @Test
  void routerSelectsStableTopKThenAppliesSoftmaxAndPerExpertScales() {
    int[] selected = new int[3];
    float[] weights = new float[3];

    Gemma4Math.selectExperts(
        new float[] {0.5f, 2.0f, -1.0f, 2.0f, 1.5f},
        new float[] {1.0f, 0.5f, 2.0f, 1.5f, 1.0f},
        selected,
        weights);

    assertThat(selected).containsExactly(1, 3, 4);
    assertThat(weights[0]).isCloseTo(0.19182587f, offset(1.0e-6f));
    assertThat(weights[1]).isCloseTo(0.5754776f, offset(1.0e-6f));
    assertThat(weights[2]).isCloseTo(0.23269653f, offset(1.0e-6f));
  }

  @Test
  void finalLogitSoftcapMatchesTheGemma4Definition() {
    float[] logits = {-60.0f, -30.0f, 0.0f, 30.0f, 60.0f};

    Gemma4Math.softcap(logits, 30.0f);

    float[] expected = {-28.920828f, -22.847824f, 0.0f, 22.847824f, 28.920828f};
    for (int index = 0; index < logits.length; index++) {
      assertThat(logits[index]).isCloseTo(expected[index], offset(3.0e-6f));
    }
  }

  @Test
  void routerRejectsNonFiniteLogitsAndMismatchedOutputBuffers() {
    assertThatThrownBy(
            () ->
                Gemma4Math.selectExperts(
                    new float[] {1.0f, Float.NaN},
                    new float[] {1.0f, 1.0f},
                    new int[1],
                    new float[1]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("routerLogits[1]");
    assertThatThrownBy(
            () ->
                Gemma4Math.selectExperts(
                    new float[] {1.0f, 2.0f}, new float[] {1.0f, 1.0f}, new int[1], new float[2]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("same top-k length");
  }
}
