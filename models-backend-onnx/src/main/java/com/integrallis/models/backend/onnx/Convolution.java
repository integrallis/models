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
    // A wide kernel at a large stride (the head's STFT: 600 taps every 150 samples) is a dot
    // product of each frame with each filter; a narrow kernel at stride 1 is a sum of shifted,
    // scaled copies of the input. Both reduce to contiguous SIMD loops.
    boolean frameDot = dilation == 1 && stride >= 8 && k >= 32;
    if (frameDot) {
      frameDotConv(
          xs, ws, bias, out, n, c, length, m, cPerGroup, k, g, stride, padBegin, outLength);
      return Tensor.ofFloats(new long[] {n, m, outLength}, out);
    }
    if (stride == 1 && dilation == 1 && (long) cPerGroup * k * outLength <= 8_000_000L) {
      im2colConv(xs, ws, bias, out, n, c, length, m, cPerGroup, k, g, padBegin, outLength);
      return Tensor.ofFloats(new long[] {n, m, outLength}, out);
    }
    float[] row = new float[outLength];
    for (int batch = 0; batch < n; batch++) {
      for (int om = 0; om < m; om++) {
        int groupIndex = om / mPerGroup;
        java.util.Arrays.fill(row, 0f);
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
            if (tEnd < tStart) {
              continue;
            }
            if (stride == 1) {
              Simd.axpy(row, tStart, weight, xs, xBase + tStart + offset, tEnd - tStart + 1);
            } else {
              for (int t = tStart; t <= tEnd; t++) {
                row[t] = Math.fma(weight, xs[xBase + t * stride + offset], row[t]);
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
    if (dilation == 1 && stride >= 8 && k >= 32 && g == 1) {
      // The inverse of the STFT-as-convolution: every output frame is one row of X^T . W, and the
      // frames overlap-add into the signal at their stride.
      float[] frames = new float[length * m * k];
      float[] xt = new float[length * c];
      for (int batch = 0; batch < n; batch++) {
        for (int ch = 0; ch < c; ch++) {
          int xBase = (batch * c + ch) * length;
          for (int t = 0; t < length; t++) {
            xt[t * c + ch] = xs[xBase + t];
          }
        }
        java.util.Arrays.fill(frames, 0f);
        // W is [c][m][k]; as a [c][m*k] matrix it maps a frame's channel vector to m*k outputs.
        Simd.gemm(xt, 0, c, ws, 0, m * k, frames, 0, m * k, length, c, m * k);
        for (int t = 0; t < length; t++) {
          int o0 = t * stride - padBegin;
          int tapStart = Math.max(0, -o0);
          int tapEnd = Math.min(k, outLength - o0);
          for (int om = 0; om < m; om++) {
            if (tapEnd > tapStart) {
              Simd.axpy(
                  out,
                  (batch * m + om) * outLength + o0 + tapStart,
                  1f,
                  frames,
                  (t * m + om) * k + tapStart,
                  tapEnd - tapStart);
            }
          }
        }
        if (bias != null) {
          for (int om = 0; om < m; om++) {
            float b = bias.floatAt(om);
            int oBase = (batch * m + om) * outLength;
            for (int t = 0; t < outLength; t++) {
              out[oBase + t] += b;
            }
          }
        }
      }
      return Tensor.ofFloats(new long[] {n, m, outLength}, out);
    }
    for (int batch = 0; batch < n; batch++) {
      for (int om = 0; om < m; om++) {
        int groupIndex = om / mPerGroup;
        int mInGroup = om % mPerGroup;
        int oBase = (batch * m + om) * outLength;
        for (int ic = 0; ic < cPerGroup; ic++) {
          int channel = groupIndex * cPerGroup + ic;
          int xBase = (batch * c + channel) * length;
          int wBase = (channel * mPerGroup + mInGroup) * k;
          if (dilation == 1) {
            // Each input sample scatters the whole filter, scaled, onto a contiguous output window.
            for (int t = 0; t < length; t++) {
              float xv = xs[xBase + t];
              int o0 = t * stride - padBegin;
              int tapStart = Math.max(0, -o0);
              int tapEnd = Math.min(k, outLength - o0);
              if (tapEnd > tapStart) {
                Simd.axpy(out, oBase + o0 + tapStart, xv, ws, wBase + tapStart, tapEnd - tapStart);
              }
            }
            continue;
          }
          for (int tap = 0; tap < k; tap++) {
            float weight = ws[wBase + tap];
            int offset = tap * dilation - padBegin;
            for (int t = 0; t < length; t++) {
              int o = t * stride + offset;
              if (o >= 0 && o < outLength) {
                out[oBase + o] = Math.fma(weight, xs[xBase + t], out[oBase + o]);
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

  /**
   * A wide kernel at a large stride is a dot product of each input frame with each filter (the
   * head's STFT). Four filters share each pass over a frame; frames that overhang the input at
   * either end take the bounds-checked scalar path.
   */
  private static void frameDotConv(
      float[] xs,
      float[] ws,
      Tensor bias,
      float[] out,
      int n,
      int c,
      int length,
      int m,
      int cPerGroup,
      int k,
      int g,
      int stride,
      int padBegin,
      int outLength) {
    int mPerGroup = m / g;
    float[] four = new float[4];
    for (int batch = 0; batch < n; batch++) {
      for (int groupIndex = 0; groupIndex < g; groupIndex++) {
        int omStart = groupIndex * mPerGroup;
        int omEnd = omStart + mPerGroup;
        for (int om = omStart; om < omEnd; om += 4) {
          int block = Math.min(4, omEnd - om);
          for (int ic = 0; ic < cPerGroup; ic++) {
            int channel = groupIndex * cPerGroup + ic;
            int xBase = (batch * c + channel) * length;
            int w0 = (om * cPerGroup + ic) * k;
            int w1 = w0 + cPerGroup * k;
            int w2 = w1 + cPerGroup * k;
            int w3 = w2 + cPerGroup * k;
            for (int t = 0; t < outLength; t++) {
              int s = t * stride - padBegin;
              if (block == 4 && s >= 0 && s + k <= length) {
                Simd.dot4(ws, w0, w1, w2, w3, xs, xBase + s, k, four, 0);
                for (int r = 0; r < 4; r++) {
                  out[(batch * m + om + r) * outLength + t] += four[r];
                }
                continue;
              }
              for (int r = 0; r < block; r++) {
                int wBase = w0 + r * cPerGroup * k;
                float acc = 0f;
                if (s >= 0 && s + k <= length) {
                  acc = Simd.dot(ws, wBase, xs, xBase + s, k);
                } else {
                  for (int tap = 0; tap < k; tap++) {
                    int xi = s + tap;
                    if (xi >= 0 && xi < length) {
                      acc = Math.fma(ws[wBase + tap], xs[xBase + xi], acc);
                    }
                  }
                }
                out[(batch * m + om + r) * outLength + t] += acc;
              }
            }
          }
        }
      }
      if (bias != null) {
        for (int om = 0; om < m; om++) {
          float b = bias.floatAt(om);
          int oBase = (batch * m + om) * outLength;
          for (int t = 0; t < outLength; t++) {
            out[oBase + t] += b;
          }
        }
      }
    }
  }

  /**
   * A narrow kernel at stride 1 is a matrix product once the input is unfolded: the column matrix
   * holds, for every (channel, tap), the input shifted by that tap with zeros in the padding, and
   * the filters multiply it as one GEMM.
   */
  private static void im2colConv(
      float[] xs,
      float[] ws,
      Tensor bias,
      float[] out,
      int n,
      int c,
      int length,
      int m,
      int cPerGroup,
      int k,
      int g,
      int padBegin,
      int outLength) {
    int mPerGroup = m / g;
    int rows = cPerGroup * k;
    float[] columns = new float[rows * outLength];
    for (int batch = 0; batch < n; batch++) {
      for (int groupIndex = 0; groupIndex < g; groupIndex++) {
        for (int ic = 0; ic < cPerGroup; ic++) {
          int channel = groupIndex * cPerGroup + ic;
          int xBase = (batch * c + channel) * length;
          for (int tap = 0; tap < k; tap++) {
            int row = (ic * k + tap) * outLength;
            int offset = tap - padBegin;
            int tStart = Math.max(0, -offset);
            int tEnd = Math.min(outLength, length - offset);
            java.util.Arrays.fill(columns, row, row + outLength, 0f);
            if (tEnd > tStart) {
              System.arraycopy(xs, xBase + tStart + offset, columns, row + tStart, tEnd - tStart);
            }
          }
        }
        int omStart = groupIndex * mPerGroup;
        Simd.gemm(
            ws,
            omStart * rows,
            rows,
            columns,
            0,
            outLength,
            out,
            (batch * m + omStart) * outLength,
            outLength,
            mPerGroup,
            rows,
            outLength);
      }
      if (bias != null) {
        for (int om = 0; om < m; om++) {
          float b = bias.floatAt(om);
          int oBase = (batch * m + om) * outLength;
          for (int t = 0; t < outLength; t++) {
            out[oBase + t] += b;
          }
        }
      }
    }
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
