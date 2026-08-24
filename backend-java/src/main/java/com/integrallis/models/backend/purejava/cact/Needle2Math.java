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
package com.integrallis.models.backend.purejava.cact;

import java.util.Arrays;
import java.util.Objects;

/** Scalar reference math for the Needle 2 forward pass. */
final class Needle2Math {

  private static final float RMS_EPSILON = 1.0e-6f;

  private Needle2Math() {}

  static void zeroCenteredRmsNorm(float[] input, float[] scale, float[] output) {
    Objects.requireNonNull(input, "input");
    Objects.requireNonNull(scale, "scale");
    Objects.requireNonNull(output, "output");
    if (input.length != scale.length || output.length < input.length || input.length == 0) {
      throw new IllegalArgumentException("RMS norm arrays must have the same non-zero width");
    }
    float squared = 0.0f;
    for (float value : input) {
      squared += value * value;
    }
    float inverseRms = (float) (1.0 / Math.sqrt(squared / input.length + RMS_EPSILON));
    for (int index = 0; index < input.length; index++) {
      output[index] = (1.0f + scale[index]) * input[index] * inverseRms;
    }
  }

  static void hadamardMlp(
      float[] input, float[] d1, float[] d2, float[] d3, int modelWidth, float[] output) {
    Objects.requireNonNull(input, "input");
    Objects.requireNonNull(d1, "d1");
    Objects.requireNonNull(d2, "d2");
    Objects.requireNonNull(d3, "d3");
    Objects.requireNonNull(output, "output");
    int width = d1.length;
    if (modelWidth <= 0
        || input.length != modelWidth
        || output.length < modelWidth
        || d2.length != width
        || d3.length != width
        || width < modelWidth
        || Integer.bitCount(width) != 1) {
      throw new IllegalArgumentException("Hadamard MLP geometry is inconsistent");
    }
    float[] work = Arrays.copyOf(input, width);
    for (int index = 0; index < width; index++) {
      work[index] *= d1[index];
    }
    normalizedHadamard(work);
    for (int index = 0; index < width; index++) {
      float gated = d2[index] * work[index];
      work[index] = gated * sigmoid(gated);
    }
    normalizedHadamard(work);
    for (int index = 0; index < modelWidth; index++) {
      output[index] = d3[index] * work[index];
    }
  }

  static void sinkhorn(float[] matrix, int size, int iterations) {
    Objects.requireNonNull(matrix, "matrix");
    if (size <= 0 || matrix.length != Math.multiplyExact(size, size) || iterations < 0) {
      throw new IllegalArgumentException("Sinkhorn geometry is inconsistent");
    }
    for (int iteration = 0; iteration < iterations; iteration++) {
      for (int row = 0; row < size; row++) {
        int offset = row * size;
        float normalizer = logSumExp(matrix, offset, 1, size);
        for (int column = 0; column < size; column++) {
          matrix[offset + column] -= normalizer;
        }
      }
      for (int column = 0; column < size; column++) {
        float normalizer = logSumExp(matrix, column, size, size);
        for (int row = 0; row < size; row++) {
          matrix[row * size + column] -= normalizer;
        }
      }
    }
    for (int index = 0; index < matrix.length; index++) {
      matrix[index] = (float) Math.exp(matrix[index]);
    }
  }

  static void applyRope(float[] vector, int offset, int headWidth, int position, float theta) {
    Objects.requireNonNull(vector, "vector");
    if (offset < 0
        || headWidth <= 0
        || (headWidth & 1) != 0
        || offset > vector.length - headWidth
        || position < 0
        || !Float.isFinite(theta)
        || theta <= 0.0f) {
      throw new IllegalArgumentException("RoPE geometry is inconsistent");
    }
    int half = headWidth / 2;
    for (int pair = 0; pair < half; pair++) {
      double frequency = Math.pow(theta, -(2.0 * pair) / headWidth);
      double angle = position * frequency;
      float cosine = (float) Math.cos(angle);
      float sine = (float) Math.sin(angle);
      int first = offset + pair;
      int second = first + half;
      float left = vector[first];
      float right = vector[second];
      vector[first] = left * cosine - right * sine;
      vector[second] = right * cosine + left * sine;
    }
  }

  static float sigmoid(float value) {
    return (float) (1.0 / (1.0 + Math.exp(-value)));
  }

  private static float logSumExp(float[] values, int offset, int stride, int count) {
    float maximum = Float.NEGATIVE_INFINITY;
    for (int index = 0; index < count; index++) {
      maximum = Math.max(maximum, values[offset + index * stride]);
    }
    float sum = 0.0f;
    for (int index = 0; index < count; index++) {
      sum += (float) Math.exp(values[offset + index * stride] - maximum);
    }
    return maximum + (float) Math.log(sum);
  }

  private static void normalizedHadamard(float[] values) {
    for (int stride = 1; stride < values.length; stride *= 2) {
      for (int base = 0; base < values.length; base += stride * 2) {
        for (int index = 0; index < stride; index++) {
          float left = values[base + index];
          float right = values[base + stride + index];
          values[base + index] = left + right;
          values[base + stride + index] = left - right;
        }
      }
    }
    float scale = (float) (1.0 / Math.sqrt(values.length));
    for (int index = 0; index < values.length; index++) {
      values[index] *= scale;
    }
  }
}
