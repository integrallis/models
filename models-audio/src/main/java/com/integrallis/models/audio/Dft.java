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

/**
 * In-place complex DFT for the audio frontends.
 *
 * <p>Radix-2 Cooley-Tukey when the length is a power of two, and a direct O(n^2) transform
 * otherwise. Whisper's {@code n_fft} is 400, which is <b>not</b> a power of two -- 2^4 * 5^2 -- so
 * the direct path is the one it actually takes. That is deliberate rather than lazy: 400 points is
 * 160k multiply-adds per frame, the frame count for a 30 s window is 3000, and correctness here is
 * worth more than a mixed-radix implementation that has to be right for every factorisation. If
 * this shows up in a profile, replace it with Bluestein and keep this as the reference the
 * replacement is tested against.
 */
final class Dft {

  private Dft() {}

  /** Transforms {@code real}/{@code imaginary} in place; both must be the same length. */
  static void transformInPlace(float[] real, float[] imaginary) {
    int n = real.length;
    if (imaginary.length != n) {
      throw new IllegalArgumentException("real and imaginary must have the same length");
    }
    if (n <= 1) {
      return;
    }
    if ((n & (n - 1)) == 0) {
      radix2(real, imaginary);
    } else {
      direct(real, imaginary);
    }
  }

  private static void radix2(float[] real, float[] imaginary) {
    int n = real.length;
    for (int i = 1, j = 0; i < n; i++) {
      int bit = n >> 1;
      for (; (j & bit) != 0; bit >>= 1) {
        j ^= bit;
      }
      j ^= bit;
      if (i < j) {
        float tr = real[i];
        real[i] = real[j];
        real[j] = tr;
        float ti = imaginary[i];
        imaginary[i] = imaginary[j];
        imaginary[j] = ti;
      }
    }
    for (int length = 2; length <= n; length <<= 1) {
      double angle = -2.0 * Math.PI / length;
      float stepReal = (float) Math.cos(angle);
      float stepImaginary = (float) Math.sin(angle);
      for (int start = 0; start < n; start += length) {
        float wReal = 1.0f;
        float wImaginary = 0.0f;
        for (int k = 0; k < length / 2; k++) {
          int even = start + k;
          int odd = even + length / 2;
          float oddReal = real[odd] * wReal - imaginary[odd] * wImaginary;
          float oddImaginary = real[odd] * wImaginary + imaginary[odd] * wReal;
          real[odd] = real[even] - oddReal;
          imaginary[odd] = imaginary[even] - oddImaginary;
          real[even] += oddReal;
          imaginary[even] += oddImaginary;
          float nextReal = wReal * stepReal - wImaginary * stepImaginary;
          wImaginary = wReal * stepImaginary + wImaginary * stepReal;
          wReal = nextReal;
        }
      }
    }
  }

  private static void direct(float[] real, float[] imaginary) {
    int n = real.length;
    double[] inputReal = new double[n];
    double[] inputImaginary = new double[n];
    for (int i = 0; i < n; i++) {
      inputReal[i] = real[i];
      inputImaginary[i] = imaginary[i];
    }
    // Only the one-sided spectrum is consumed, but the whole transform is written so this stays a
    // usable reference rather than a special case that cannot be tested against itself.
    for (int k = 0; k < n; k++) {
      double sumReal = 0.0;
      double sumImaginary = 0.0;
      for (int t = 0; t < n; t++) {
        double angle = -2.0 * Math.PI * k * t / n;
        double cos = Math.cos(angle);
        double sin = Math.sin(angle);
        sumReal += inputReal[t] * cos - inputImaginary[t] * sin;
        sumImaginary += inputReal[t] * sin + inputImaginary[t] * cos;
      }
      real[k] = (float) sumReal;
      imaginary[k] = (float) sumImaginary;
    }
  }
}
