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
package com.integrallis.models.api;

import java.time.Duration;
import java.util.Arrays;
import java.util.Objects;

/** Immutable normalized floating-point PCM audio with interleaved channels. */
public final class PcmAudio {

  private final int sampleRate;
  private final int channels;
  private final float[] samples;

  /** Creates PCM audio and takes an immutable snapshot of the supplied samples. */
  public PcmAudio(int sampleRate, int channels, float[] samples) {
    if (sampleRate <= 0) {
      throw new IllegalArgumentException("sampleRate must be positive: " + sampleRate);
    }
    if (channels <= 0) {
      throw new IllegalArgumentException("channels must be positive: " + channels);
    }
    Objects.requireNonNull(samples, "samples");
    if (samples.length % channels != 0) {
      throw new IllegalArgumentException(
          "interleaved sample count must be divisible by channels: "
              + samples.length
              + " and "
              + channels);
    }
    for (float sample : samples) {
      if (!Float.isFinite(sample)) {
        throw new IllegalArgumentException("PCM samples must be finite");
      }
    }
    this.sampleRate = sampleRate;
    this.channels = channels;
    this.samples = samples.clone();
  }

  /** Samples per second, per channel. */
  public int sampleRate() {
    return sampleRate;
  }

  /** Number of interleaved channels. */
  public int channels() {
    return channels;
  }

  /** Returns a caller-owned copy of the normalized samples. */
  public float[] samples() {
    return samples.clone();
  }

  /** Number of sample frames, where one frame contains one sample for every channel. */
  public int frameCount() {
    return samples.length / channels;
  }

  /** Audio duration rounded down to nanosecond precision. */
  public Duration duration() {
    long nanos = (long) ((double) frameCount() * 1_000_000_000.0 / sampleRate);
    return Duration.ofNanos(nanos);
  }

  @Override
  public boolean equals(Object other) {
    if (this == other) {
      return true;
    }
    return other instanceof PcmAudio audio
        && sampleRate == audio.sampleRate
        && channels == audio.channels
        && Arrays.equals(samples, audio.samples);
  }

  @Override
  public int hashCode() {
    int result = Objects.hash(sampleRate, channels);
    return 31 * result + Arrays.hashCode(samples);
  }

  @Override
  public String toString() {
    return "PcmAudio[sampleRate="
        + sampleRate
        + ", channels="
        + channels
        + ", frameCount="
        + frameCount()
        + "]";
  }
}
