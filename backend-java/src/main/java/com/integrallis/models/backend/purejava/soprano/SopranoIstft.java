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
package com.integrallis.models.backend.purejava.soprano;

import com.integrallis.models.backend.purejava.audio.Radix2Fft;
import java.util.Objects;

/** Soprano's non-iterative inverse-STFT waveform reconstruction. */
final class SopranoIstft {

  private static final float MAX_MAGNITUDE = 100.0f;
  private static final float MIN_ENVELOPE = 1.0e-11f;

  private SopranoIstft() {}

  static float[] decode(float[] head, int frames, int fftSize, int hopLength, float[] window) {
    Objects.requireNonNull(head, "head");
    Objects.requireNonNull(window, "window");
    if (frames <= 0) {
      throw new IllegalArgumentException("frames must be positive: " + frames);
    }
    if (fftSize <= 0 || (fftSize & (fftSize - 1)) != 0) {
      throw new IllegalArgumentException("FFT size must be a positive power of two: " + fftSize);
    }
    if (hopLength <= 0) {
      throw new IllegalArgumentException("hopLength must be positive: " + hopLength);
    }
    int headWidth = fftSize + 2;
    if (head.length != Math.multiplyExact(frames, headWidth)) {
      throw new IllegalArgumentException(
          "Soprano decoder head must contain " + frames + " x " + headWidth + " values");
    }
    if (window.length != fftSize) {
      throw new IllegalArgumentException(
          "Soprano ISTFT window must contain " + fftSize + " values");
    }

    int bins = fftSize / 2 + 1;
    int unfoldedSize = Math.addExact(Math.multiplyExact(frames - 1, hopLength), fftSize);
    float[] folded = new float[unfoldedSize];
    float[] envelope = new float[unfoldedSize];
    float[] real = new float[bins];
    float[] imaginary = new float[bins];
    for (int frame = 0; frame < frames; frame++) {
      int row = frame * headWidth;
      for (int bin = 0; bin < bins; bin++) {
        float magnitude = Math.min((float) Math.exp(head[row + bin]), MAX_MAGNITUDE);
        if (bin == 0 || bin == bins - 1) {
          magnitude = 0.0f;
        }
        float phase = head[row + bins + bin];
        real[bin] = magnitude * (float) Math.cos(phase);
        imaginary[bin] = magnitude * (float) Math.sin(phase);
      }
      float[] samples = Radix2Fft.inverseReal(real, imaginary, fftSize);
      int start = frame * hopLength;
      for (int index = 0; index < fftSize; index++) {
        float weight = window[index];
        folded[start + index] += samples[index] * weight;
        envelope[start + index] += weight * weight;
      }
    }

    int padding = fftSize / 2;
    int outputSize = unfoldedSize - 2 * padding;
    if (outputSize <= 0) {
      throw new IllegalArgumentException("Soprano ISTFT produced no samples after center trim");
    }
    float[] output = new float[outputSize];
    for (int index = 0; index < outputSize; index++) {
      int source = index + padding;
      if (envelope[source] > MIN_ENVELOPE) {
        output[index] = folded[source] / envelope[source];
      }
    }
    return output;
  }
}
