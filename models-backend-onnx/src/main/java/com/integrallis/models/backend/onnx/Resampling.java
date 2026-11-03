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
      if (fill.sameKind(result)) {
        result.fillWith(fill);
      } else {
        for (int i = 0; i < result.size(); i++) {
          result.copyElement(fill, 0, i);
        }
      }
      if (rank == 0) {
        result.copyElement(x, 0, 0);
        return result;
      }
      long[] outStrides = result.strides();
      int width = (int) in[rank - 1];
      int rows = width == 0 ? 0 : x.size() / width;
      long[] counter = new long[rank];
      for (int row = 0; row < rows; row++) {
        long target = p[rank - 1] * outStrides[rank - 1];
        for (int axis = 0; axis < rank - 1; axis++) {
          target += (counter[axis] + p[axis]) * outStrides[axis];
        }
        result.copyStrided(x, row * width, 1, (int) target, width);
        for (int axis = rank - 2; axis >= 0; axis--) {
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
    AxisMap map =
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
        };
    if (rank == 0 || p[rank - 1] < 0 || p[2 * rank - 1] < 0) {
      return remap(x, out, map);
    }
    return padRows(x, in, p, out, map);
  }

  /**
   * Reflect or edge padding with non-negative padding on the last axis: every output row is its
   * input row copied whole into the middle, plus the few mapped elements on either side. Mapping
   * every element instead, through the axis function, measured at 1.1 ms for one 64,800-sample
   * waveform where onnxruntime's block copies ({@code pad.cc}) take a few microseconds.
   */
  private static Tensor padRows(Tensor x, long[] in, long[] p, long[] out, AxisMap map) {
    int rank = in.length;
    int last = rank - 1;
    Tensor result = x.emptyLike(out);
    long[] inStrides = x.strides();
    int width = (int) out[last];
    int dim = (int) in[last];
    int begin = (int) p[last];
    int rightStart = begin + dim;
    int[] left = new int[begin];
    for (int o = 0; o < begin; o++) {
      left[o] = map.source(last, o);
    }
    int[] right = new int[width - rightStart];
    for (int o = 0; o < right.length; o++) {
      right[o] = map.source(last, rightStart + o);
    }
    int rows = width == 0 ? 0 : result.size() / width;
    long[] counter = new long[rank];
    for (int row = 0; row < rows; row++) {
      long base = 0;
      for (int axis = 0; axis < last; axis++) {
        base += map.source(axis, (int) counter[axis]) * inStrides[axis];
      }
      int to = row * width;
      result.gatherRow(x, (int) base, left, to);
      result.copyStrided(x, (int) base, 1, to + begin, dim);
      result.gatherRow(x, (int) base, right, to + rightStart);
      for (int axis = last - 1; axis >= 0; axis--) {
        if (++counter[axis] < out[axis]) {
          break;
        }
        counter[axis] = 0;
      }
    }
    return result;
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
    if (rank == 0) {
      result.copyElement(x, 0, 0);
      return result;
    }
    long[] inStrides = x.strides();
    // The innermost axis is mapped once into source offsets; each output row then costs one
    // offset sum over the outer axes and one typed gather, never a per-element map call.
    int last = rank - 1;
    int width = (int) shape[last];
    int[] inner = new int[width];
    for (int o = 0; o < width; o++) {
      inner[o] = (int) (map.source(last, o) * inStrides[last]);
    }
    int rows = width == 0 ? 0 : result.size() / width;
    long[] counter = new long[rank];
    for (int row = 0; row < rows; row++) {
      long base = 0;
      for (int axis = 0; axis < last; axis++) {
        base += map.source(axis, (int) counter[axis]) * inStrides[axis];
      }
      result.gatherRow(x, (int) base, inner, row * width);
      for (int axis = last - 1; axis >= 0; axis--) {
        if (++counter[axis] < shape[axis]) {
          break;
        }
        counter[axis] = 0;
      }
    }
    return result;
  }
}
