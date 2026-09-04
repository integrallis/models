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
package com.integrallis.models.backend.purejava.audio;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class Radix2FftTest {

  @Test
  void reconstructsAConstantSignalFromItsRealSpectrum() {
    float[] real = {8.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    float[] imaginary = new float[real.length];

    assertThat(Radix2Fft.inverseReal(real, imaginary, 8))
        .containsExactly(
            new float[] {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f}, within(1.0e-6f));
  }

  @Test
  void reconstructsAUnitImpulseFromAFlatSpectrum() {
    float[] real = {1.0f, 1.0f, 1.0f, 1.0f, 1.0f};
    float[] imaginary = new float[real.length];

    assertThat(Radix2Fft.inverseReal(real, imaginary, 8))
        .containsExactly(
            new float[] {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f}, within(1.0e-6f));
  }

  @Test
  void rejectsNonPowerOfTwoTransformsAndWrongBinCounts() {
    assertThatThrownBy(() -> Radix2Fft.inverseReal(new float[4], new float[4], 6))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("power of two");
    assertThatThrownBy(() -> Radix2Fft.inverseReal(new float[4], new float[3], 8))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("bins");
  }
}
