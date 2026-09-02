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

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import org.junit.jupiter.api.Test;

class RotaryTableLlama3Test {

  @Test
  void matchesTheMobileMoeLlama3FrequencyRulesAndAdjacentPairLayout() {
    RotaryTable rope = RotaryTable.llama3(8, 10_000.0f, 8.0f, 1_000, 1.0f, 1.0f);
    rope.prepare(3);
    float[] actual = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    rope.apply(actual, 0, false);

    float[] frequencies = {1.0f, 0.1f, 0.01f, 0.001f / 8.0f};
    float[] expected = new float[8];
    for (int pair = 0; pair < frequencies.length; pair++) {
      expected[pair * 2] = (float) Math.cos(3.0 * frequencies[pair]);
      expected[pair * 2 + 1] = (float) Math.sin(3.0 * frequencies[pair]);
    }
    assertArrayEquals(expected, actual, 1.0e-6f);
  }

  @Test
  void interpolatesTheMiddleFrequencyBand() {
    RotaryTable rope = RotaryTable.llama3(8, 10_000.0f, 4.0f, 8_192, 1.0f, 4.0f);
    rope.prepare(1);
    float[] actual = {1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f};
    rope.apply(actual, 0, false);

    double original = 0.001;
    double wavelength = 2.0 * Math.PI / original;
    double smooth = (8_192.0 / wavelength - 1.0) / 3.0;
    double interpolated = (1.0 - smooth) * original / 4.0 + smooth * original;
    assertArrayEquals(
        new float[] {
          (float) Math.cos(1.0),
          (float) Math.sin(1.0),
          (float) Math.cos(0.1),
          (float) Math.sin(0.1),
          (float) Math.cos(0.01),
          (float) Math.sin(0.01),
          (float) Math.cos(interpolated),
          (float) Math.sin(interpolated)
        },
        actual,
        1.0e-6f);
  }
}
