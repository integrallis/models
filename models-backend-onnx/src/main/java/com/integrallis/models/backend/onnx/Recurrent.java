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
 * ONNX {@code LSTM} with the default activations (sigmoid, tanh, tanh), layout 0, gate order {@code
 * i, o, f, c}, forward, reverse or bidirectional, without peepholes.
 */
final class Recurrent {

  private Recurrent() {}

  /** Returns {@code Y [seq, dirs, batch, hidden]}, {@code Y_h} and {@code Y_c}. */
  static Tensor[] lstm(
      Tensor x,
      Tensor w,
      Tensor r,
      Tensor b,
      Tensor sequenceLens,
      Tensor initialH,
      Tensor initialC,
      String direction,
      int hidden) {
    int seq = (int) x.dim(0);
    int batch = (int) x.dim(1);
    int input = (int) x.dim(2);
    int dirs = direction.equals("bidirectional") ? 2 : 1;
    if (w.dim(0) != dirs || w.dim(1) != 4L * hidden || w.dim(2) != input) {
      throw new IllegalArgumentException(
          "LSTM weight shape "
              + w
              + " does not match "
              + dirs
              + " x "
              + 4 * hidden
              + " x "
              + input);
    }
    float[] xs = x.toFloat().floats();
    float[] ws = w.toFloat().floats();
    float[] rs = r.toFloat().floats();
    float[] bs = b == null ? null : b.toFloat().floats();
    float[] y = new float[seq * dirs * batch * hidden];
    float[] yh = new float[dirs * batch * hidden];
    float[] yc = new float[dirs * batch * hidden];
    float[] gates = new float[4 * hidden];
    for (int d = 0; d < dirs; d++) {
      boolean reverse = direction.equals("reverse") || d == 1;
      int wBase = d * 4 * hidden * input;
      int rBase = d * 4 * hidden * hidden;
      int bBase = d * 8 * hidden;
      for (int n = 0; n < batch; n++) {
        float[] h = new float[hidden];
        float[] c = new float[hidden];
        if (initialH != null) {
          for (int i = 0; i < hidden; i++) {
            h[i] = initialH.floatAt((d * batch + n) * hidden + i);
          }
        }
        if (initialC != null) {
          for (int i = 0; i < hidden; i++) {
            c[i] = initialC.floatAt((d * batch + n) * hidden + i);
          }
        }
        int length = sequenceLens == null ? seq : (int) sequenceLens.longAt(n);
        // The input side of every gate at every timestep is one matrix product,
        // projected[g][t] = W[g] . x[t], as onnxruntime computes it (uni_directional_lstm.cc,
        // ComputeGemm over all rows) instead of one dot product per gate per step.
        int gatesCount = 4 * hidden;
        float[] inputsByColumn = new float[input * length];
        for (int t = 0; t < length; t++) {
          int xBase = (t * batch + n) * input;
          for (int i = 0; i < input; i++) {
            inputsByColumn[i * length + t] = xs[xBase + i];
          }
        }
        float[] projected = new float[gatesCount * length];
        Simd.gemm(
            ws,
            wBase,
            input,
            inputsByColumn,
            0,
            length,
            projected,
            0,
            length,
            gatesCount,
            input,
            length);
        float[] cellTanh = new float[hidden];
        for (int step = 0; step < length; step++) {
          int t = reverse ? length - 1 - step : step;
          for (int gate = 0; gate < gatesCount; gate++) {
            float bias = bs == null ? 0f : bs[bBase + gate] + bs[bBase + gatesCount + gate];
            gates[gate] =
                projected[gate * length + t]
                    + bias
                    + Simd.dot(rs, rBase + gate * hidden, h, 0, hidden);
          }
          // Gate order is i, o, f, c: the first three take the logistic, the last one tanh.
          Activations.logistic(gates, 0, gates, 0, 3 * hidden);
          Activations.tanh(gates, 3 * hidden, gates, 3 * hidden, hidden);
          for (int j = 0; j < hidden; j++) {
            c[j] = gates[2 * hidden + j] * c[j] + gates[j] * gates[3 * hidden + j];
          }
          Activations.tanh(c, 0, cellTanh, 0, hidden);
          for (int j = 0; j < hidden; j++) {
            h[j] = gates[hidden + j] * cellTanh[j];
          }
          int yBase = ((t * dirs + d) * batch + n) * hidden;
          System.arraycopy(h, 0, y, yBase, hidden);
        }
        System.arraycopy(h, 0, yh, (d * batch + n) * hidden, hidden);
        System.arraycopy(c, 0, yc, (d * batch + n) * hidden, hidden);
      }
    }
    return new Tensor[] {
      Tensor.ofFloats(new long[] {seq, dirs, batch, hidden}, y),
      Tensor.ofFloats(new long[] {dirs, batch, hidden}, yh),
      Tensor.ofFloats(new long[] {dirs, batch, hidden}, yc)
    };
  }
}
