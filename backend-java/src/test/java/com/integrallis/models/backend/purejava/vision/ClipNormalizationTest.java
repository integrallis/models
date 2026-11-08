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
package com.integrallis.models.backend.purejava.vision;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;

import java.util.Arrays;
import org.junit.jupiter.api.Test;

class ClipNormalizationTest {
  @Test
  void aLargeCommonOffsetDoesNotDistortTheNormalizedContrast() {
    int size = 1024;
    float[] original = new float[size];
    float[] translated = new float[size];
    float[] weight = new float[size];
    float[] bias = new float[size];
    Arrays.fill(weight, 1);
    for (int i = 0; i < size; i++) {
      original[i] = i < size / 2 ? -0.5f : 0.5f;
      translated[i] = original[i] + 16384f;
    }
    float[] expected = new float[size];
    float[] actual = new float[size];
    ClipVisionEncoder.normalizeRow(expected, original, 0, weight, bias, size, 1e-5f);
    ClipVisionEncoder.normalizeRow(actual, translated, 0, weight, bias, size, 1e-5f);
    assertArrayEquals(expected, actual, 1e-6f);
    ClipVisionEncoder.normalizeRow(translated, translated, 0, weight, bias, size, 1e-5f);
    assertArrayEquals(actual, translated);
  }
}
