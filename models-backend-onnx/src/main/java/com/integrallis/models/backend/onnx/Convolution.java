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

/**
 * One-dimensional {@code Conv} and {@code ConvTranspose} over {@code [N, C, L]} inputs.
 *
 * <p>Only the 1-D case is implemented, because that is all the graphs this module serves contain; a
 * 2-D kernel is refused rather than silently mishandled.
 */
final class Convolution {

  private Convolution() {}

  static Tensor conv(
      Tensor x, Tensor w, Tensor bias, long[] dilations, long group, long[] pads, long[] strides) {
    require1d(x, w, "Conv");
    int n = (int) x.dim(0);
    int c = (int) x.dim(1);
    int length = (int) x.dim(2);
    int m = (int) w.dim(0);
    int cPerGroup = (int) w.dim(1);
    int k = (int) w.dim(2);
    int g = (int) group;
    int dilation = dilations.length == 0 ? 1 : (int) dilations[0];
    int stride = strides.length == 0 ? 1 : (int) strides[0];
    int padBegin = pads.length == 0 ? 0 : (int) pads[0];
    int padEnd = pads.length < 2 ? padBegin : (int) pads[1];
    if (c != cPerGroup * g || m % g != 0) {
      throw new IllegalArgumentException(
          "Conv channel mismatch: input "
              + c
              + ", weight "
              + cPerGroup
              + " x "
              + g
              + " groups, output "
              + m);
    }
    int span = dilation * (k - 1) + 1;
    int outLength = (length + padBegin + padEnd - span) / stride + 1;
    if (outLength <= 0) {
      throw new IllegalArgumentException("Conv output length " + outLength);
    }
    float[] xs = x.toFloat().floats();
    float[] ws = w.toFloat().floats();
    float[] out = new float[n * m * outLength];
    int mPerGroup = m / g;
    for (int batch = 0; batch < n; batch++) {
      for (int om = 0; om < m; om++) {
        int groupIndex = om / mPerGroup;
        float[] row = new float[outLength];
        for (int ic = 0; ic < cPerGroup; ic++) {
          int channel = groupIndex * cPerGroup + ic;
          int xBase = (batch * c + channel) * length;
          int wBase = (om * cPerGroup + ic) * k;
          for (int tap = 0; tap < k; tap++) {
            float weight = ws[wBase + tap];
            int offset = tap * dilation - padBegin;
            // t runs over outputs whose tap lands inside the input.
            int tStart = Math.max(0, ceilDiv(-offset, stride));
            int tEnd = Math.min(outLength - 1, (length - 1 - offset) / stride);
            if (stride == 1) {
              int xi = xBase + tStart + offset;
              for (int t = tStart; t <= tEnd; t++, xi++) {
                row[t] += weight * xs[xi];
              }
            } else {
              for (int t = tStart; t <= tEnd; t++) {
                row[t] += weight * xs[xBase + t * stride + offset];
              }
            }
          }
        }
        float b = bias == null ? 0f : bias.floatAt(om);
        int oBase = (batch * m + om) * outLength;
        for (int t = 0; t < outLength; t++) {
          out[oBase + t] = row[t] + b;
        }
      }
    }
    return Tensor.ofFloats(new long[] {n, m, outLength}, out);
  }

  static Tensor convTranspose(
      Tensor x,
      Tensor w,
      Tensor bias,
      long[] dilations,
      long group,
      long[] outputPadding,
      long[] pads,
      long[] strides) {
    require1d(x, w, "ConvTranspose");
    int n = (int) x.dim(0);
    int c = (int) x.dim(1);
    int length = (int) x.dim(2);
    int g = (int) group;
    int mPerGroup = (int) w.dim(1);
    int k = (int) w.dim(2);
    int m = mPerGroup * g;
    int dilation = dilations.length == 0 ? 1 : (int) dilations[0];
    int stride = strides.length == 0 ? 1 : (int) strides[0];
    int padBegin = pads.length == 0 ? 0 : (int) pads[0];
    int padEnd = pads.length < 2 ? padBegin : (int) pads[1];
    int extra = outputPadding.length == 0 ? 0 : (int) outputPadding[0];
    if (c != w.dim(0) || c % g != 0) {
      throw new IllegalArgumentException(
          "ConvTranspose channel mismatch: input " + c + ", weight " + w.dim(0));
    }
    int cPerGroup = c / g;
    int outLength = (length - 1) * stride - padBegin - padEnd + dilation * (k - 1) + extra + 1;
    if (outLength <= 0) {
      throw new IllegalArgumentException("ConvTranspose output length " + outLength);
    }
    float[] xs = x.toFloat().floats();
    float[] ws = w.toFloat().floats();
    float[] out = new float[n * m * outLength];
    for (int batch = 0; batch < n; batch++) {
      for (int om = 0; om < m; om++) {
        int groupIndex = om / mPerGroup;
        int mInGroup = om % mPerGroup;
        int oBase = (batch * m + om) * outLength;
        for (int ic = 0; ic < cPerGroup; ic++) {
          int channel = groupIndex * cPerGroup + ic;
          int xBase = (batch * c + channel) * length;
          int wBase = (channel * mPerGroup + mInGroup) * k;
          for (int tap = 0; tap < k; tap++) {
            float weight = ws[wBase + tap];
            int offset = tap * dilation - padBegin;
            for (int t = 0; t < length; t++) {
              int o = t * stride + offset;
              if (o >= 0 && o < outLength) {
                out[oBase + o] += weight * xs[xBase + t];
              }
            }
          }
        }
        if (bias != null) {
          float b = bias.floatAt(om);
          for (int t = 0; t < outLength; t++) {
            out[oBase + t] += b;
          }
        }
      }
    }
    return Tensor.ofFloats(new long[] {n, m, outLength}, out);
  }

  private static void require1d(Tensor x, Tensor w, String op) {
    if (x.rank() != 3 || w.rank() != 3) {
      throw new UnsupportedOperationException(
          op
              + " supports 1-D signals only: got input rank "
              + x.rank()
              + ", weight rank "
              + w.rank());
    }
  }

  private static int ceilDiv(int a, int b) {
    return Math.floorDiv(a + b - 1, b);
  }
}
