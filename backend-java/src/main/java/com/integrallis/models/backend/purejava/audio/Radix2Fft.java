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

import java.util.Objects;

/** In-place radix-2 FFT operations used by Java-native audio decoders. */
public final class Radix2Fft {

  private Radix2Fft() {}

  /** Reconstructs a real signal from its non-negative-frequency complex bins. */
  public static float[] inverseReal(float[] binReal, float[] binImaginary, int size) {
    Objects.requireNonNull(binReal, "binReal");
    Objects.requireNonNull(binImaginary, "binImaginary");
    if (size <= 0 || (size & (size - 1)) != 0) {
      throw new IllegalArgumentException("FFT size must be a positive power of two: " + size);
    }
    int expectedBins = size / 2 + 1;
    if (binReal.length != expectedBins || binImaginary.length != expectedBins) {
      throw new IllegalArgumentException(
          "real and imaginary bins must both contain " + expectedBins + " values");
    }

    double[] real = new double[size];
    double[] imaginary = new double[size];
    for (int bin = 0; bin < expectedBins; bin++) {
      real[bin] = binReal[bin];
      imaginary[bin] = binImaginary[bin];
    }
    for (int bin = 1; bin < size / 2; bin++) {
      real[size - bin] = real[bin];
      imaginary[size - bin] = -imaginary[bin];
    }

    transform(real, imaginary, true);
    float[] samples = new float[size];
    for (int i = 0; i < size; i++) {
      samples[i] = (float) real[i];
    }
    return samples;
  }

  private static void transform(double[] real, double[] imaginary, boolean inverse) {
    int size = real.length;
    for (int i = 1, reversed = 0; i < size; i++) {
      int bit = size >>> 1;
      while ((reversed & bit) != 0) {
        reversed ^= bit;
        bit >>>= 1;
      }
      reversed ^= bit;
      if (i < reversed) {
        swap(real, i, reversed);
        swap(imaginary, i, reversed);
      }
    }

    for (int width = 2; width <= size; width <<= 1) {
      double angle = (inverse ? 2.0 : -2.0) * Math.PI / width;
      double stepReal = Math.cos(angle);
      double stepImaginary = Math.sin(angle);
      int half = width >>> 1;
      for (int offset = 0; offset < size; offset += width) {
        double twiddleReal = 1.0;
        double twiddleImaginary = 0.0;
        for (int j = 0; j < half; j++) {
          int even = offset + j;
          int odd = even + half;
          double oddReal = real[odd] * twiddleReal - imaginary[odd] * twiddleImaginary;
          double oddImaginary = real[odd] * twiddleImaginary + imaginary[odd] * twiddleReal;
          real[odd] = real[even] - oddReal;
          imaginary[odd] = imaginary[even] - oddImaginary;
          real[even] += oddReal;
          imaginary[even] += oddImaginary;
          double nextReal = twiddleReal * stepReal - twiddleImaginary * stepImaginary;
          twiddleImaginary = twiddleReal * stepImaginary + twiddleImaginary * stepReal;
          twiddleReal = nextReal;
        }
      }
    }

    if (inverse) {
      for (int i = 0; i < size; i++) {
        real[i] /= size;
        imaginary[i] /= size;
      }
    }
  }

  private static void swap(double[] values, int left, int right) {
    double value = values[left];
    values[left] = values[right];
    values[right] = value;
  }
}
