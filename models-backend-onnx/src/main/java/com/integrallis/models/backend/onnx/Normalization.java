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
package com.integrallis.models.backend.onnx;

/** {@code LayerNormalization} and {@code InstanceNormalization}. */
final class Normalization {

  private Normalization() {}

  /**
   * Opset-17 {@code LayerNormalization} over the trailing dimensions from {@code axis}; returns
   * {@code Y}, {@code Mean} and {@code InvStdDev} in that order.
   */
  static Tensor[] layerNorm(
      Tensor x, Tensor scale, Tensor bias, long axisAttribute, float epsilon) {
    int axis = x.normalizeAxis((int) axisAttribute);
    long[] shape = x.shape();
    int outer = 1;
    for (int i = 0; i < axis; i++) {
      outer *= (int) shape[i];
    }
    int inner = x.size() / Math.max(1, outer);
    float[] xs = x.toFloat().floats();
    float[] out = new float[xs.length];
    float[] means = new float[outer];
    float[] invStd = new float[outer];
    int[] scaleMap =
        scale.size() == inner ? null : Broadcast.indexMap(scale.shape(), trailing(shape, axis));
    int[] biasMap =
        bias == null || bias.size() == inner
            ? null
            : Broadcast.indexMap(bias.shape(), trailing(shape, axis));
    for (int o = 0; o < outer; o++) {
      int base = o * inner;
      double sum = 0;
      for (int i = 0; i < inner; i++) {
        sum += xs[base + i];
      }
      double mean = sum / inner;
      double variance = 0;
      for (int i = 0; i < inner; i++) {
        double d = xs[base + i] - mean;
        variance += d * d;
      }
      variance /= inner;
      double inv = 1.0 / Math.sqrt(variance + epsilon);
      means[o] = (float) mean;
      invStd[o] = (float) inv;
      for (int i = 0; i < inner; i++) {
        float s = scale.floatAt(scaleMap == null ? i : scaleMap[i]);
        float b = bias == null ? 0f : bias.floatAt(biasMap == null ? i : biasMap[i]);
        out[base + i] = (float) ((xs[base + i] - mean) * inv) * s + b;
      }
    }
    long[] statShape = new long[shape.length];
    System.arraycopy(shape, 0, statShape, 0, axis);
    java.util.Arrays.fill(statShape, axis, shape.length, 1);
    return new Tensor[] {
      Tensor.ofFloats(shape, out),
      Tensor.ofFloats(statShape, means),
      Tensor.ofFloats(statShape, invStd)
    };
  }

  private static long[] trailing(long[] shape, int axis) {
    return java.util.Arrays.copyOfRange(shape, axis, shape.length);
  }

  /** {@code InstanceNormalization} over the spatial dimensions of each {@code (n, c)} plane. */
  static Tensor instanceNorm(Tensor x, Tensor scale, Tensor bias, float epsilon) {
    int n = (int) x.dim(0);
    int c = (int) x.dim(1);
    int spatial = x.size() / Math.max(1, n * c);
    float[] xs = x.toFloat().floats();
    float[] out = new float[xs.length];
    for (int b = 0; b < n; b++) {
      for (int ch = 0; ch < c; ch++) {
        int base = (b * c + ch) * spatial;
        double sum = 0;
        for (int i = 0; i < spatial; i++) {
          sum += xs[base + i];
        }
        double mean = sum / spatial;
        double variance = 0;
        for (int i = 0; i < spatial; i++) {
          double d = xs[base + i] - mean;
          variance += d * d;
        }
        variance /= spatial;
        double inv = 1.0 / Math.sqrt(variance + epsilon);
        float s = scale.floatAt(ch);
        float o = bias == null ? 0f : bias.floatAt(ch);
        for (int i = 0; i < spatial; i++) {
          out[base + i] = (float) ((xs[base + i] - mean) * inv) * s + o;
        }
      }
    }
    return Tensor.ofFloats(x.shape(), out);
  }
}
