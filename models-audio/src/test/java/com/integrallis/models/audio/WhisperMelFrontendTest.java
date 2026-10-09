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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Whisper's log-mel frontend.
 *
 * <p>Each test pins one of the details that produces a plausible-looking matrix when wrong, because
 * that is the whole failure mode here: an encoder fed a subtly wrong spectrogram transcribes badly
 * rather than erroring.
 */
@Tag("unit")
class WhisperMelFrontendTest {

  @Test
  void theHannWindowIsPeriodicNotSymmetric() {
    // The header says hann_periodic, and torch's hann_window is periodic by default, which is what
    // whisper was trained with. audio.cpp's Default family divides by N-1 and is symmetric, so
    // transcribing that one would have been wrong -- this test is the guard against doing so.
    float[] periodic = WhisperMelFrontend.hannPeriodic(400);
    assertThat(periodic[0]).isEqualTo(0.0f);
    assertThat(periodic[200])
        .as("periodic Hann peaks at exactly 1 on the midpoint of an even length")
        .isCloseTo(1.0f, within(1.0e-7f));
    // A symmetric window would divide by 399 and miss the peak at index 200.
    double symmetricMid = 0.5 - 0.5 * Math.cos(2.0 * Math.PI * 200 / 399);
    assertThat(symmetricMid).isNotCloseTo(1.0, within(1.0e-6));
  }

  @Test
  void theMelScaleIsSlaneyWithItsBreakpointAtOneKilohertz() {
    // Slaney: linear at 200/3 Hz per mel below 1 kHz, logarithmic above. HTK uses a single
    // formula and would give a different filterbank entirely.
    assertThat(WhisperMelFrontend.hzToMel(0.0)).isEqualTo(0.0);
    assertThat(WhisperMelFrontend.hzToMel(1000.0))
        .as("1000 / (200/3) = 15 exactly, which is the breakpoint")
        .isCloseTo(15.0, within(1.0e-9));
    assertThat(WhisperMelFrontend.hzToMel(500.0)).isCloseTo(7.5, within(1.0e-9));
    // Above the break it is logarithmic, so doubling the frequency adds a constant.
    double a = WhisperMelFrontend.hzToMel(2000.0) - WhisperMelFrontend.hzToMel(1000.0);
    double b = WhisperMelFrontend.hzToMel(4000.0) - WhisperMelFrontend.hzToMel(2000.0);
    assertThat(a).isCloseTo(b, within(1.0e-6));
  }

  @Test
  void melAndHzRoundTrip() {
    for (double hz : new double[] {0.0, 100.0, 999.0, 1000.0, 1001.0, 4000.0, 8000.0}) {
      assertThat(WhisperMelFrontend.melToHz(WhisperMelFrontend.hzToMel(hz)))
          .as("round trip at %s Hz", hz)
          .isCloseTo(hz, within(1.0e-3));
    }
  }

  @Test
  void theFilterbankHasTheShapeTheHeaderDeclares() {
    WhisperMelFrontend frontend = WhisperMelFrontend.whisperDefault();
    assertThat(frontend.frequencyBins()).as("n_fft/2 + 1").isEqualTo(201);
    float[][] bank = frontend.filterbank();
    assertThat(bank).hasDimensions(80, 201);
    for (int mel = 0; mel < 80; mel++) {
      for (int bin = 0; bin < 201; bin++) {
        assertThat(bank[mel][bin])
            .as("filter %s bin %s is non-negative", mel, bin)
            .isGreaterThanOrEqualTo(0.0f);
      }
      assertThat(sum(bank[mel])).as("filter %s carries weight", mel).isGreaterThan(0.0f);
    }
  }

  @Test
  void slaneyNormalizationMakesLowFiltersTallerThanHighOnes() {
    // The normalization divides by the filter's Hz width. Mel filters widen with frequency, so a
    // correctly normalized bank has decreasing peaks. Omitting the normalization leaves every peak
    // at 1 and lets the wide high-frequency filters quietly dominate the spectrum.
    float[][] bank = WhisperMelFrontend.whisperDefault().filterbank();
    float lowPeak = max(bank[2]);
    float highPeak = max(bank[70]);
    assertThat(lowPeak).isGreaterThan(highPeak);
  }

  @Test
  void aToneLandsInTheMelBinThatCoversItsFrequency() {
    // End to end on a signal whose answer is known: a 1 kHz sine should put its energy in the
    // filters whose triangle spans 1 kHz, and nowhere near the top of the bank.
    WhisperMelFrontend frontend = WhisperMelFrontend.whisperDefault();
    int sampleRate = 16_000;
    float[] samples = new float[sampleRate / 2];
    for (int i = 0; i < samples.length; i++) {
      samples[i] = (float) Math.sin(2.0 * Math.PI * 1000.0 * i / sampleRate);
    }
    float[][] mel = frontend.logMel(samples);
    assertThat(mel).hasDimensions(80, frontend.frameCount(samples.length));

    int loudest = 0;
    float best = Float.NEGATIVE_INFINITY;
    int middleFrame = mel[0].length / 2;
    for (int m = 0; m < 80; m++) {
      if (mel[m][middleFrame] > best) {
        best = mel[m][middleFrame];
        loudest = m;
      }
    }
    // 1 kHz is the Slaney breakpoint, mel 15 of a 0..8000 Hz bank over 80 filters.
    assertThat(loudest)
        .as("a 1 kHz tone should peak in a low-mid filter, got filter %s", loudest)
        .isBetween(10, 30);
    assertThat(mel[79][middleFrame]).as("nothing near 8 kHz").isLessThan(mel[loudest][middleFrame]);
  }

  @Test
  void theFinalStftFrameIsDropped() {
    // frames = stftFrames - 1, per the reference. A model fed one frame too many does not fail,
    // it misaligns, so this is pinned rather than inferred.
    WhisperMelFrontend frontend = WhisperMelFrontend.whisperDefault();
    assertThat(frontend.frameCount(16_000)).isEqualTo(100);
    assertThat(frontend.frameCount(480_000))
        .as("a 30 s window is the 3000 frames the header's nb_max_frames declares")
        .isEqualTo(3000);
  }

  @Test
  void normalizationFloorsRelativeToTheLoudestValueInTheWholeClip() {
    // The floor is max - 8 over the entire matrix, not per frame, so every value lands in a known
    // band and silence does not blow up.
    WhisperMelFrontend frontend = WhisperMelFrontend.whisperDefault();
    float[] samples = new float[8_000];
    for (int i = 0; i < samples.length; i++) {
      samples[i] = (float) (0.5 * Math.sin(2.0 * Math.PI * 440.0 * i / 16_000));
    }
    float[][] mel = frontend.logMel(samples);
    float min = Float.POSITIVE_INFINITY;
    float max = Float.NEGATIVE_INFINITY;
    for (float[] row : mel) {
      for (float value : row) {
        min = Math.min(min, value);
        max = Math.max(max, value);
      }
    }
    // (max(v, max-8) + 4) / 4 spans exactly 2 units between the floor and the peak.
    assertThat(max - min).isCloseTo(2.0f, within(1.0e-5f));
  }

  @Test
  void silenceProducesTheFloorEverywhereRatherThanNaN() {
    // log10(0) is -infinity; the 1e-10 clamp is what keeps an all-zero buffer finite.
    float[][] mel = WhisperMelFrontend.whisperDefault().logMel(new float[16_000]);
    for (float[] row : mel) {
      for (float value : row) {
        assertThat(Float.isFinite(value)).isTrue();
      }
    }
  }

  @Test
  void refusesAConfigurationItCannotHonour() {
    assertThatThrownBy(() -> new WhisperMelFrontend(16_000, 400, 160, 512, 80, 0, 8_000))
        .as("a window longer than the transform cannot be placed")
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("cannot exceed n_fft");
    assertThatThrownBy(() -> new WhisperMelFrontend(16_000, 400, 160, 400, 80, 8_000, 0))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("f_max must exceed f_min");
    assertThatThrownBy(() -> WhisperMelFrontend.whisperDefault().logMel(new float[0]))
        .isInstanceOf(IllegalArgumentException.class);
  }

  private static float sum(float[] values) {
    float total = 0.0f;
    for (float value : values) {
      total += value;
    }
    return total;
  }

  private static float max(float[] values) {
    float best = Float.NEGATIVE_INFINITY;
    for (float value : values) {
      best = Math.max(best, value);
    }
    return best;
  }
}
