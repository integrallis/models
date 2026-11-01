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

/** {@code DequantizeLinear} for the int8 weight graphs. */
final class Quantization {

  private Quantization() {}

  /** {@code y = (x - zero_point) * scale}, per tensor or per {@code axis}. */
  static Tensor dequantizeLinear(Tensor x, Tensor scale, Tensor zeroPoint, long axisAttribute) {
    int count = x.size();
    float[] out = new float[count];
    if (scale.size() == 1) {
      float s = scale.floatAt(0);
      long zp = zeroPoint == null ? 0 : zeroPoint.longAt(0);
      for (int i = 0; i < count; i++) {
        out[i] = (x.longAt(i) - zp) * s;
      }
      return Tensor.ofFloats(x.shape(), out);
    }
    int axis = x.normalizeAxis((int) axisAttribute);
    Reductions.Axis a = Reductions.around(x, axis);
    if (scale.size() != a.length()) {
      throw new IllegalArgumentException(
          "DequantizeLinear scale has "
              + scale.size()
              + " entries for axis of length "
              + a.length());
    }
    for (int o = 0; o < a.outer(); o++) {
      for (int k = 0; k < a.length(); k++) {
        float s = scale.floatAt(k);
        long zp = zeroPoint == null ? 0 : zeroPoint.longAt(k);
        int base = (o * a.length() + k) * a.inner();
        for (int i = 0; i < a.inner(); i++) {
          out[base + i] = (x.longAt(base + i) - zp) * s;
        }
      }
    }
    return Tensor.ofFloats(x.shape(), out);
  }
}
