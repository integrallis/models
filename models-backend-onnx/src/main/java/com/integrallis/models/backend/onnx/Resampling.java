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

/** {@code Resize} (nearest, asymmetric, floor) and {@code Pad} (constant, reflect, edge). */
final class Resampling {

  private Resampling() {}

  /**
   * Opset-13 {@code Resize} restricted to {@code mode=nearest}, {@code
   * coordinate_transformation_mode=asymmetric}, {@code nearest_mode=floor}: the one configuration
   * the served graphs use. Other modes are refused.
   */
  static Tensor resizeNearest(
      Tensor x,
      Tensor scales,
      Tensor sizes,
      String mode,
      String coordinateMode,
      String nearestMode) {
    if (!mode.equals("nearest")
        || !coordinateMode.equals("asymmetric")
        || !nearestMode.equals("floor")) {
      throw new UnsupportedOperationException(
          "Resize mode="
              + mode
              + " coordinate_transformation_mode="
              + coordinateMode
              + " nearest_mode="
              + nearestMode
              + " is not implemented");
    }
    long[] in = x.shape();
    int rank = in.length;
    long[] out = new long[rank];
    double[] scale = new double[rank];
    if (sizes != null && sizes.size() > 0) {
      long[] s = sizes.toInt64().longs();
      for (int i = 0; i < rank; i++) {
        out[i] = s[i];
        scale[i] = (double) out[i] / in[i];
      }
    } else {
      for (int i = 0; i < rank; i++) {
        scale[i] = scales.floatAt(i);
        out[i] = (long) Math.floor(in[i] * scale[i]);
      }
    }
    int[][] maps = new int[rank][];
    for (int axis = 0; axis < rank; axis++) {
      maps[axis] = new int[(int) out[axis]];
      for (int o = 0; o < out[axis]; o++) {
        int source = (int) Math.floor(o / scale[axis]);
        maps[axis][o] = Math.min((int) in[axis] - 1, Math.max(0, source));
      }
    }
    return remap(x, out, (axis, o) -> maps[axis][o]);
  }

  /** Opset-13 {@code Pad}: {@code pads} lists begins for every axis then ends for every axis. */
  static Tensor pad(Tensor x, Tensor pads, Tensor constantValue, String mode) {
    long[] in = x.shape();
    int rank = in.length;
    long[] p = pads.toInt64().longs();
    long[] out = new long[rank];
    for (int i = 0; i < rank; i++) {
      out[i] = in[i] + p[i] + p[rank + i];
    }
    if (mode.equals("constant")) {
      Tensor fill = constantValue == null ? x.emptyLike(new long[] {1}) : constantValue;
      Tensor result = x.emptyLike(out);
      int count = result.size();
      for (int i = 0; i < count; i++) {
        result.copyElement(fill, 0, i);
      }
      long[] outStrides = result.strides();
      int inCount = x.size();
      long[] counter = new long[rank];
      for (int flat = 0; flat < inCount; flat++) {
        long target = 0;
        for (int axis = 0; axis < rank; axis++) {
          target += (counter[axis] + p[axis]) * outStrides[axis];
        }
        result.copyElement(x, flat, (int) target);
        for (int axis = rank - 1; axis >= 0; axis--) {
          if (++counter[axis] < in[axis]) {
            break;
          }
          counter[axis] = 0;
        }
      }
      return result;
    }
    boolean reflect = mode.equals("reflect");
    if (!reflect && !mode.equals("edge")) {
      throw new UnsupportedOperationException("Pad mode " + mode + " is not implemented");
    }
    return remap(
        x,
        out,
        (axis, o) -> {
          long dim = in[axis];
          long index = o - p[axis];
          if (reflect) {
            if (dim == 1) {
              return 0;
            }
            long period = 2 * (dim - 1);
            index = Math.floorMod(index, period);
            if (index >= dim) {
              index = period - index;
            }
            return (int) index;
          }
          return (int) Math.max(0, Math.min(dim - 1, index));
        });
  }

  interface AxisMap {
    int source(int axis, int outputIndex);
  }

  /**
   * Builds an output of {@code shape} whose element at each multi-index comes from the mapped input
   * index.
   */
  private static Tensor remap(Tensor x, long[] shape, AxisMap map) {
    int rank = shape.length;
    Tensor result = x.emptyLike(shape);
    long[] inStrides = x.strides();
    int count = result.size();
    long[] counter = new long[rank];
    for (int flat = 0; flat < count; flat++) {
      long source = 0;
      for (int axis = 0; axis < rank; axis++) {
        source += map.source(axis, (int) counter[axis]) * inStrides[axis];
      }
      result.copyElement(x, (int) source, flat);
      for (int axis = rank - 1; axis >= 0; axis--) {
        if (++counter[axis] < shape[axis]) {
          break;
        }
        counter[axis] = 0;
      }
    }
    return result;
  }
}
