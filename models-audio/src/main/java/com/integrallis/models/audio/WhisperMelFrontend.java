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
 * Whisper's log-mel frontend: PCM in, the matrix the encoder consumes out.
 *
 * <p>This carries no model weights. Every constant is declared by the artifact's own GGUF header
 * under {@code stt.frontend.*}, which is unusually complete for whisper and is why this can be
 * written without guessing: {@code type mel}, {@code normalize whisper_logmel}, {@code num_mels
 * 80}, {@code n_fft 400}, {@code win_length 400}, {@code hop_length 160}, {@code window
 * hann_periodic}, {@code pad_mode reflect}, {@code center true}, {@code sample_rate 16000}, {@code
 * mel_norm slaney}, {@code f_min 0}, {@code f_max 8000}.
 *
 * <p>The arithmetic is transcribed from {@code WhisperLogMelExtractor::compute} in {@code
 * audio.cpp} ({@code src/framework/audio/dsp.cpp}), with four details that are invisible when wrong
 * because they all still produce a plausible-looking matrix:
 *
 * <ul>
 *   <li><b>Magnitude, not power.</b> The reference feeds the mel filterbank the STFT magnitude.
 *       Squaring first changes every value and still looks like a spectrogram.
 *   <li><b>The final frame is dropped</b> -- {@code frames = stftFrames - 1}. A model fed 3001
 *       frames where it expects 3000 does not fail, it misaligns.
 *   <li><b>The mel scale is Slaney, not HTK</b>: linear below 1 kHz at {@code 200/3} Hz per mel,
 *       logarithmic above with {@code ln(6.4)/27} per step. HTK's single formula is a different
 *       filterbank entirely.
 *   <li><b>The Hann window is periodic</b> ({@code denominator = winLength}), per the header's
 *       {@code hann_periodic}. Note this <em>differs</em> from audio.cpp's {@code Default} family,
 *       which divides by {@code winLength - 1} and is therefore symmetric; transcribing that one
 *       would have been wrong for whisper. The artifact's header is the authority.
 * </ul>
 *
 * <p>The normalization is global over the clip: {@code log10}, then a floor at {@code max - 8},
 * then {@code (value + 4) / 4}. Because the floor depends on the loudest frame, this is a property
 * of the whole window rather than of each frame, and chunking the audio differently changes it.
 */
public final class WhisperMelFrontend {

  /** Hz per mel below the log region, and the breakpoint, from the Slaney scale. */
  private static final double MEL_HZ_PER_UNIT = 200.0 / 3.0;

  private static final double MEL_LOG_BREAK_HZ = 1000.0;
  private static final double MEL_LOG_BREAK = MEL_LOG_BREAK_HZ / MEL_HZ_PER_UNIT;

  /** {@code ln(6.4) / 27}, the reference's constant rather than a recomputed one. */
  private static final double MEL_LOG_STEP = 0.06875177742094912;

  private final int sampleRate;
  private final int fftSize;
  private final int hopLength;
  private final int windowLength;
  private final int melBins;
  private final double minFrequency;
  private final double maxFrequency;
  private final float[] window;
  private final float[][] filterbank;

  /** The configuration whisper-tiny through whisper-large declare; all of them share it. */
  public static WhisperMelFrontend whisperDefault() {
    return new WhisperMelFrontend(16_000, 400, 160, 400, 80, 0.0, 8_000.0);
  }

  /**
   * @param sampleRate {@code stt.frontend.sample_rate}
   * @param fftSize {@code stt.frontend.n_fft}
   * @param hopLength {@code stt.frontend.hop_length}
   * @param windowLength {@code stt.frontend.win_length}
   * @param melBins {@code stt.frontend.num_mels}
   * @param minFrequency {@code stt.frontend.f_min}
   * @param maxFrequency {@code stt.frontend.f_max}
   */
  public WhisperMelFrontend(
      int sampleRate,
      int fftSize,
      int hopLength,
      int windowLength,
      int melBins,
      double minFrequency,
      double maxFrequency) {
    if (sampleRate <= 0 || fftSize <= 1 || hopLength <= 0 || windowLength <= 1 || melBins <= 0) {
      throw new IllegalArgumentException("whisper mel frontend requires positive dimensions");
    }
    if (windowLength > fftSize) {
      throw new IllegalArgumentException(
          "winLength " + windowLength + " cannot exceed n_fft " + fftSize);
    }
    if (!(maxFrequency > minFrequency)) {
      throw new IllegalArgumentException("f_max must exceed f_min");
    }
    this.sampleRate = sampleRate;
    this.fftSize = fftSize;
    this.hopLength = hopLength;
    this.windowLength = windowLength;
    this.melBins = melBins;
    this.minFrequency = minFrequency;
    this.maxFrequency = maxFrequency;
    this.window = hannPeriodic(windowLength);
    this.filterbank = slaneyFilterbank();
  }

  /** Bins per STFT frame, {@code n_fft / 2 + 1}. */
  public int frequencyBins() {
    return fftSize / 2 + 1;
  }

  /** The mel filterbank, {@code melBins} rows of {@link #frequencyBins()}. */
  public float[][] filterbank() {
    return filterbank;
  }

  /** The analysis window, periodic Hann of {@code winLength}. */
  public float[] window() {
    return window.clone();
  }

  /**
   * Periodic Hann: {@code 0.5 - 0.5 cos(2 pi i / N)}.
   *
   * <p>Periodic, not symmetric. The header says {@code hann_periodic} and torch's {@code
   * hann_window} is periodic by default, which is what whisper was trained with.
   */
  static float[] hannPeriodic(int length) {
    float[] values = new float[length];
    if (length == 1) {
      values[0] = 1.0f;
      return values;
    }
    for (int i = 0; i < length; i++) {
      values[i] = (float) (0.5 - 0.5 * Math.cos(2.0 * Math.PI * i / length));
    }
    return values;
  }

  /** Slaney mel scale: linear to 1 kHz, logarithmic above it. */
  static double hzToMel(double hz) {
    if (hz >= MEL_LOG_BREAK_HZ) {
      return MEL_LOG_BREAK + Math.log(hz / MEL_LOG_BREAK_HZ) / MEL_LOG_STEP;
    }
    return hz / MEL_HZ_PER_UNIT;
  }

  /** Inverse of {@link #hzToMel}. */
  static double melToHz(double mel) {
    if (mel >= MEL_LOG_BREAK) {
      return MEL_LOG_BREAK_HZ * Math.exp(MEL_LOG_STEP * (mel - MEL_LOG_BREAK));
    }
    return mel * MEL_HZ_PER_UNIT;
  }

  private float[][] slaneyFilterbank() {
    int bins = frequencyBins();
    double melMin = hzToMel(minFrequency);
    double melMax = hzToMel(maxFrequency);
    double[] hzPoints = new double[melBins + 2];
    for (int i = 0; i < melBins + 2; i++) {
      hzPoints[i] = melToHz(melMin + (melMax - melMin) * i / (melBins + 1));
    }
    double[] fftFrequencies = new double[bins];
    for (int i = 0; i < bins; i++) {
      fftFrequencies[i] = sampleRate * 0.5 * i / (bins - 1);
    }
    float[][] bank = new float[melBins][bins];
    for (int mel = 0; mel < melBins; mel++) {
      double left = hzPoints[mel];
      double center = hzPoints[mel + 1];
      double right = hzPoints[mel + 2];
      double lowerWidth = Math.max(center - left, 1.0e-12);
      double upperWidth = Math.max(right - center, 1.0e-12);
      for (int i = 0; i < bins; i++) {
        double frequency = fftFrequencies[i];
        double lower = (frequency - left) / lowerWidth;
        double upper = (right - frequency) / upperWidth;
        bank[mel][i] = (float) Math.max(0.0, Math.min(lower, upper));
      }
      // Slaney normalization: each filter integrates to the same area, so a flat spectrum does not
      // tilt across the bank. Omitting it makes high-frequency bins quietly dominate.
      double enorm = 2.0 / Math.max(hzPoints[mel + 2] - hzPoints[mel], 1.0e-12);
      for (int i = 0; i < bins; i++) {
        bank[mel][i] *= (float) enorm;
      }
    }
    return bank;
  }

  /** How many frames {@code samples} produces, after the reference drops the final one. */
  public int frameCount(int samples) {
    if (samples <= 0) {
      throw new IllegalArgumentException("samples must be positive");
    }
    return stftFrames(samples) - 1;
  }

  private int stftFrames(int samples) {
    // center = true: the signal is reflect-padded by n_fft/2 on each side, so a frame is centred
    // on sample 0.
    return samples / hopLength + 1;
  }

  /**
   * Computes the log-mel matrix, mel-major: {@code melBins} rows of {@link #frameCount(int)}.
   *
   * @param samples mono PCM in [-1, 1] at {@link #sampleRate}
   * @return the matrix whisper's encoder consumes
   */
  public float[][] logMel(float[] samples) {
    if (samples == null || samples.length == 0) {
      throw new IllegalArgumentException("samples must not be empty");
    }
    int frames = stftFrames(samples.length);
    if (frames <= 1) {
      throw new IllegalArgumentException(
          "need more than one STFT frame; got " + frames + " from " + samples.length + " samples");
    }
    int bins = frequencyBins();
    int keptFrames = frames - 1;
    float[][] mel = new float[melBins][keptFrames];

    float[] real = new float[fftSize];
    float[] imaginary = new float[fftSize];
    float[] magnitude = new float[bins];
    int half = fftSize / 2;

    for (int frame = 0; frame < keptFrames; frame++) {
      int centre = frame * hopLength;
      java.util.Arrays.fill(real, 0.0f);
      java.util.Arrays.fill(imaginary, 0.0f);
      for (int i = 0; i < windowLength; i++) {
        int index = centre + i - half;
        real[i] = reflect(samples, index) * window[i];
      }
      Dft.transformInPlace(real, imaginary);
      for (int bin = 0; bin < bins; bin++) {
        magnitude[bin] = (float) Math.hypot(real[bin], imaginary[bin]);
      }
      for (int m = 0; m < melBins; m++) {
        float[] row = filterbank[m];
        double sum = 0.0;
        for (int bin = 0; bin < bins; bin++) {
          sum += row[bin] * magnitude[bin];
        }
        mel[m][frame] = (float) sum;
      }
    }

    // log10, then a floor relative to the loudest value in the whole clip, then scale. The floor
    // is global on purpose: it is what makes the output depend on the window as a whole.
    float maxLog = Float.NEGATIVE_INFINITY;
    for (int m = 0; m < melBins; m++) {
      for (int frame = 0; frame < keptFrames; frame++) {
        float value = (float) Math.log10(Math.max(mel[m][frame], 1.0e-10f));
        mel[m][frame] = value;
        maxLog = Math.max(maxLog, value);
      }
    }
    float floor = maxLog - 8.0f;
    for (int m = 0; m < melBins; m++) {
      for (int frame = 0; frame < keptFrames; frame++) {
        mel[m][frame] = (Math.max(mel[m][frame], floor) + 4.0f) / 4.0f;
      }
    }
    return mel;
  }

  /** Reflect padding, which is what {@code pad_mode reflect} with {@code center true} means. */
  private static float reflect(float[] samples, int index) {
    int length = samples.length;
    if (length == 1) {
      return samples[0];
    }
    int period = 2 * (length - 1);
    int wrapped = Math.floorMod(index, period);
    return samples[wrapped < length ? wrapped : period - wrapped];
  }
}
