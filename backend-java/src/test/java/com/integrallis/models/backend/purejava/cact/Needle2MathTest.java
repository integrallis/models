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
package com.integrallis.models.backend.purejava.cact;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class Needle2MathTest {

  private static final float TOLERANCE = 2.0e-6f;

  @Test
  void zeroCenteredRmsNormMatchesNeedleReference() {
    float[] input = {0.5f, -1.0f, 2.0f, -0.25f};
    float[] scale = {-0.1f, 0.2f, 0.0f, 0.05f};
    float[] output = new float[4];

    Needle2Math.zeroCenteredRmsNorm(input, scale, output);

    assertThat(output)
        .containsExactly(
            new float[] {0.39047465f, -1.0412658f, 1.735443f, -0.22777687f}, within(TOLERANCE));
  }

  @Test
  void hadamardMlpMatchesNeedleReference() {
    float[] output = new float[4];

    Needle2Math.hadamardMlp(
        new float[] {0.5f, -1.0f, 2.0f, -0.25f},
        new float[] {1.0f, 0.5f, 1.5f, -1.0f},
        new float[] {0.75f, -0.5f, 1.25f, 0.25f},
        new float[] {0.1f, 0.2f, -0.3f, 0.4f},
        4,
        output);

    assertThat(output)
        .containsExactly(
            new float[] {0.017193085f, 0.10664165f, -0.15148069f, 0.26851755f}, within(TOLERANCE));
  }

  @Test
  void sinkhornAndRopeMatchNeedleReference() {
    float[] matrix = {0.2f, -0.4f, 1.1f, 0.3f};
    Needle2Math.sinkhorn(matrix, 2, 20);
    assertThat(matrix)
        .containsExactly(
            new float[] {0.4750208f, 0.5249792f, 0.5249792f, 0.47502083f}, within(TOLERANCE));

    float[] vector = {0.5f, -1.0f, 2.0f, -0.25f};
    Needle2Math.applyRope(vector, 0, 4, 3, 10000.0f);
    assertThat(vector)
        .containsExactly(
            new float[] {-0.7772362f, -0.9920512f, -1.909425f, -0.279883f}, within(TOLERANCE));
  }
}
