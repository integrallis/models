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

import java.io.IOException;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * {@link WhisperMelFrontend} against the artifact's own tensors.
 *
 * <p>The whisper GGUF converter ships {@code frontend.window} and {@code frontend.mel_filterbank}
 * inside the model, so the published artifact states exactly what its frontend must produce. These
 * fixtures are those tensors, extracted verbatim from {@code
 * handy-computer/whisper-tiny-gguf/whisper-tiny-Q5_K_M.gguf} -- see the resource README. That makes
 * this an oracle comparison rather than a fixture written to match our own output, which is the
 * difference between checking the implementation and checking it against itself.
 */
@Tag("unit")
class WhisperMelFrontendOracleTest {

  @Test
  void theComputedWindowMatchesTheOneInsideTheArtifact() throws IOException {
    float[] expected = readFloats("/whisper/window-400.f32", 400);
    float[] actual = WhisperMelFrontend.hannPeriodic(400);

    double worst = 0.0;
    for (int i = 0; i < expected.length; i++) {
      worst = Math.max(worst, Math.abs(expected[i] - actual[i]));
    }
    assertThat(worst)
        .as("periodic Hann must match the artifact's own window to float precision")
        .isLessThan(1.0e-6);

    // The window is also the test that settles periodic against symmetric. audio.cpp's Default
    // family divides by winLength - 1; that form differs from this artifact by about 6e-3, which
    // is five orders of magnitude worse than the agreement above. The header said hann_periodic
    // and the artifact's own tensor confirms it.
    double symmetricWorst = 0.0;
    for (int i = 0; i < expected.length; i++) {
      double symmetric = 0.5 - 0.5 * Math.cos(2.0 * Math.PI * i / 399);
      symmetricWorst = Math.max(symmetricWorst, Math.abs(expected[i] - symmetric));
    }
    assertThat(symmetricWorst)
        .as("a symmetric window is measurably the wrong answer here")
        .isGreaterThan(1.0e-4);
  }

  @Test
  void theComputedFilterbankMatchesTheOneInsideTheArtifact() throws IOException {
    int melBins = 80;
    int frequencyBins = 201;
    // GGUF dims are fastest-varying first, so dims [201, 80] is mel-major with the bins
    // contiguous -- the same layout filterbank() returns.
    float[] expected = readFloats("/whisper/mel-filterbank-80x201.f32", melBins * frequencyBins);
    float[][] actual = WhisperMelFrontend.whisperDefault().filterbank();
    assertThat(actual).hasDimensions(melBins, frequencyBins);

    double worst = 0.0;
    double peak = 0.0;
    for (int mel = 0; mel < melBins; mel++) {
      for (int bin = 0; bin < frequencyBins; bin++) {
        double reference = expected[mel * frequencyBins + bin];
        peak = Math.max(peak, Math.abs(reference));
        worst = Math.max(worst, Math.abs(reference - actual[mel][bin]));
      }
    }
    assertThat(peak).as("the fixture carries real weight").isGreaterThan(0.0);
    assertThat(worst / peak)
        .as("Slaney filterbank must match the artifact's own to float precision")
        .isLessThan(1.0e-6);
  }

  @Test
  void skippingSlaneyNormalizationWouldBeMeasurablyWrong() throws IOException {
    // Guards the detail most likely to be dropped as cosmetic. Without the 2/(hz[m+2]-hz[m])
    // scaling every filter peaks at 1 instead of ~0.026, so the error is tens of times the
    // artifact's own peak -- not a rounding difference.
    int melBins = 80;
    int frequencyBins = 201;
    float[] expected = readFloats("/whisper/mel-filterbank-80x201.f32", melBins * frequencyBins);
    float[][] normalized = WhisperMelFrontend.whisperDefault().filterbank();

    double peak = 0.0;
    for (float value : expected) {
      peak = Math.max(peak, Math.abs(value));
    }
    double unnormalizedPeak = 0.0;
    for (float[] row : normalized) {
      for (float value : row) {
        unnormalizedPeak = Math.max(unnormalizedPeak, value);
      }
    }
    // Both are the normalized peak; the point is that it is far below 1, which is what an
    // unnormalized triangular filter would reach.
    assertThat(peak).isLessThan(0.1);
    assertThat(unnormalizedPeak).isLessThan(0.1);
    assertThat(Math.abs(peak - unnormalizedPeak) / peak).isLessThan(1.0e-5);
  }

  private static float[] readFloats(String resource, int count) throws IOException {
    try (InputStream stream = WhisperMelFrontendOracleTest.class.getResourceAsStream(resource)) {
      if (stream == null) {
        throw new IOException("missing test resource: " + resource);
      }
      byte[] bytes = stream.readAllBytes();
      if (bytes.length != count * Float.BYTES) {
        throw new IOException(
            resource + " is " + bytes.length + " bytes, expected " + count * Float.BYTES);
      }
      ByteBuffer buffer = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN);
      float[] values = new float[count];
      buffer.asFloatBuffer().get(values);
      return values;
    }
  }
}
