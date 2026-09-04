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

import java.util.Objects;

/** Small frame-major operations used by Soprano's ConvNeXt vocoder. */
final class SopranoVocoderMath {

  private SopranoVocoderMath() {}

  static float[] interpolateAligned(float[] input, int frames, int channels, int upscale) {
    Objects.requireNonNull(input, "input");
    if (frames <= 0 || channels <= 0 || upscale <= 0) {
      throw new IllegalArgumentException("frames, channels, and upscale must be positive");
    }
    if (input.length != Math.multiplyExact(frames, channels)) {
      throw new IllegalArgumentException("input shape does not match frames and channels");
    }
    if (frames == 1) {
      return input.clone();
    }
    int outputFrames = Math.addExact(Math.multiplyExact(upscale, frames - 1), 1);
    float[] output = new float[Math.multiplyExact(outputFrames, channels)];
    for (int frame = 0; frame < outputFrames; frame++) {
      float source = (float) frame * (frames - 1) / (outputFrames - 1);
      int left = (int) source;
      int right = Math.min(left + 1, frames - 1);
      float fraction = source - left;
      for (int channel = 0; channel < channels; channel++) {
        float from = input[left * channels + channel];
        float to = input[right * channels + channel];
        output[frame * channels + channel] = from + fraction * (to - from);
      }
    }
    return output;
  }

  static float[] depthwiseConv1d(
      float[] input, int frames, int channels, float[] weights, float[] bias, int kernelSize) {
    Objects.requireNonNull(input, "input");
    Objects.requireNonNull(weights, "weights");
    Objects.requireNonNull(bias, "bias");
    if (frames <= 0 || channels <= 0 || kernelSize <= 0 || (kernelSize & 1) == 0) {
      throw new IllegalArgumentException(
          "frames, channels, and an odd kernelSize must be positive");
    }
    if (input.length != Math.multiplyExact(frames, channels)
        || weights.length != Math.multiplyExact(channels, kernelSize)
        || bias.length != channels) {
      throw new IllegalArgumentException("depthwise convolution tensor shapes do not match");
    }
    int padding = kernelSize / 2;
    float[] output = new float[input.length];
    for (int frame = 0; frame < frames; frame++) {
      for (int channel = 0; channel < channels; channel++) {
        float value = bias[channel];
        for (int kernel = 0; kernel < kernelSize; kernel++) {
          int sourceFrame = frame + kernel - padding;
          if (sourceFrame >= 0 && sourceFrame < frames) {
            value +=
                input[sourceFrame * channels + channel] * weights[channel * kernelSize + kernel];
          }
        }
        output[frame * channels + channel] = value;
      }
    }
    return output;
  }
}
