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
package com.integrallis.models.audio;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;
import static org.assertj.core.api.Assertions.within;

import java.util.Random;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The audio DFT.
 *
 * <p>Whisper's {@code n_fft} is 400, which is not a power of two, so the radix-2 path is
 * <b>unreachable from the frontend</b> and would otherwise ship untested. These tests exercise it
 * directly and, more usefully, pin the two paths against each other: the direct transform is the
 * readable reference, so any faster implementation has something to be wrong against.
 */
@Tag("unit")
class DftTest {

  @Test
  void theRadixTwoAndDirectPathsAgree() {
    // The only check that matters for the fast path: on a length both paths accept, they must
    // produce the same spectrum. 256 is a power of two so radix2 runs; 255 forces the direct one.
    Random random = new Random(20261009L);
    float[] realFast = new float[256];
    float[] imaginaryFast = new float[256];
    for (int i = 0; i < realFast.length; i++) {
      realFast[i] = random.nextFloat() * 2.0f - 1.0f;
    }
    float[] realSlow = realFast.clone();
    float[] imaginarySlow = imaginaryFast.clone();

    Dft.transformInPlace(realFast, imaginaryFast);
    directReference(realSlow, imaginarySlow);

    for (int k = 0; k < realFast.length; k++) {
      assertThat(realFast[k]).as("real bin %s", k).isCloseTo(realSlow[k], within(1.0e-3f));
      assertThat(imaginaryFast[k])
          .as("imaginary bin %s", k)
          .isCloseTo(imaginarySlow[k], within(1.0e-3f));
    }
  }

  @Test
  void aConstantSignalPutsAllItsEnergyInBinZero() {
    // DC in, DC out: bin 0 is the sum, every other bin is zero. True for both paths, so it is run
    // on a power-of-two length and on whisper's own 400.
    for (int length : new int[] {64, 400}) {
      float[] real = new float[length];
      float[] imaginary = new float[length];
      java.util.Arrays.fill(real, 0.5f);
      Dft.transformInPlace(real, imaginary);
      assertThat(real[0]).as("length %s bin 0", length).isCloseTo(0.5f * length, within(1.0e-2f));
      for (int k = 1; k < length; k++) {
        assertThat(Math.hypot(real[k], imaginary[k]))
            .as("length %s bin %s", length, k)
            .isLessThan(1.0e-2);
      }
    }
  }

  @Test
  void aSingleCycleSineLandsInBinOne() {
    // One full cycle across the window is bin 1 by construction, which catches a transform that is
    // right in magnitude but wrong in frequency ordering.
    int length = 400;
    float[] real = new float[length];
    float[] imaginary = new float[length];
    for (int i = 0; i < length; i++) {
      real[i] = (float) Math.sin(2.0 * Math.PI * i / length);
    }
    Dft.transformInPlace(real, imaginary);
    double loudest = 0.0;
    int at = -1;
    for (int k = 1; k < length / 2; k++) {
      double magnitude = Math.hypot(real[k], imaginary[k]);
      if (magnitude > loudest) {
        loudest = magnitude;
        at = k;
      }
    }
    assertThat(at).isEqualTo(1);
  }

  @Test
  void lengthsBelowTwoAreLeftAlone() {
    float[] real = {3.0f};
    float[] imaginary = {0.0f};
    Dft.transformInPlace(real, imaginary);
    assertThat(real[0]).isEqualTo(3.0f);
  }

  @Test
  void mismatchedLengthsAreRefused() {
    assertThatThrownBy(() -> Dft.transformInPlace(new float[8], new float[4]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("same length");
  }

  /** The textbook transform, written here so the test does not lean on the code under test. */
  private static void directReference(float[] real, float[] imaginary) {
    int n = real.length;
    double[] inputReal = new double[n];
    double[] inputImaginary = new double[n];
    for (int i = 0; i < n; i++) {
      inputReal[i] = real[i];
      inputImaginary[i] = imaginary[i];
    }
    for (int k = 0; k < n; k++) {
      double sumReal = 0.0;
      double sumImaginary = 0.0;
      for (int t = 0; t < n; t++) {
        double angle = -2.0 * Math.PI * k * t / n;
        sumReal += inputReal[t] * Math.cos(angle) - inputImaginary[t] * Math.sin(angle);
        sumImaginary += inputReal[t] * Math.sin(angle) + inputImaginary[t] * Math.cos(angle);
      }
      real[k] = (float) sumReal;
      imaginary[k] = (float) sumImaginary;
    }
  }
}
