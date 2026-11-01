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

/** Reductions and scans along one axis: {@code ReduceSum}, {@code CumSum}, {@code Softmax}. */
final class Reductions {

  private Reductions() {}

  /** The (outer, length, inner) decomposition of a tensor around one axis. */
  record Axis(int outer, int length, int inner) {}

  static Axis around(Tensor x, int axis) {
    long[] shape = x.shape();
    int outer = 1;
    for (int i = 0; i < axis; i++) {
      outer *= (int) shape[i];
    }
    int inner = 1;
    for (int i = axis + 1; i < shape.length; i++) {
      inner *= (int) shape[i];
    }
    return new Axis(outer, (int) shape[axis], inner);
  }

  /** Opset-13 {@code ReduceSum}: axes as an optional input, all axes when absent. */
  static Tensor reduceSum(
      Tensor x, Tensor axesTensor, boolean keepdims, boolean noopWithEmptyAxes) {
    long[] axes;
    if (axesTensor == null || axesTensor.size() == 0) {
      if (noopWithEmptyAxes) {
        return x;
      }
      axes = new long[x.rank()];
      for (int i = 0; i < axes.length; i++) {
        axes[i] = i;
      }
    } else {
      axes = axesTensor.toInt64().longs();
    }
    boolean[] reduce = new boolean[x.rank()];
    for (long axis : axes) {
      reduce[x.normalizeAxis((int) axis)] = true;
    }
    Tensor current = x;
    for (int axis = x.rank() - 1; axis >= 0; axis--) {
      if (reduce[axis]) {
        current = sumAxis(current, axis);
      }
    }
    if (!keepdims) {
      long[] shape = current.shape();
      int kept = 0;
      for (int i = 0; i < shape.length; i++) {
        kept += reduce[i] ? 0 : 1;
      }
      long[] squeezed = new long[kept];
      int j = 0;
      for (int i = 0; i < shape.length; i++) {
        if (!reduce[i]) {
          squeezed[j++] = shape[i];
        }
      }
      current = current.reshape(squeezed);
    }
    return current;
  }

  private static Tensor sumAxis(Tensor x, int axis) {
    Axis a = around(x, axis);
    long[] shape = x.shape();
    shape[axis] = 1;
    if (x.isIntegral()) {
      long[] out = new long[a.outer() * a.inner()];
      for (int o = 0; o < a.outer(); o++) {
        for (int k = 0; k < a.length(); k++) {
          int base = (o * a.length() + k) * a.inner();
          for (int i = 0; i < a.inner(); i++) {
            out[o * a.inner() + i] += x.longAt(base + i);
          }
        }
      }
      return Tensor.ofLongs(shape, out);
    }
    if (x.isFloat()) {
      float[] in = x.floats();
      float[] out = new float[a.outer() * a.inner()];
      if (a.inner() == 1) {
        for (int o = 0; o < a.outer(); o++) {
          out[o] = Simd.sum(in, o * a.length(), a.length());
        }
      } else {
        for (int o = 0; o < a.outer(); o++) {
          int oBase = o * a.inner();
          for (int k = 0; k < a.length(); k++) {
            Simd.add(out, oBase, out, oBase, in, (o * a.length() + k) * a.inner(), a.inner());
          }
        }
      }
      return Tensor.ofFloats(shape, out);
    }
    double[] acc = new double[a.outer() * a.inner()];
    for (int o = 0; o < a.outer(); o++) {
      for (int k = 0; k < a.length(); k++) {
        int base = (o * a.length() + k) * a.inner();
        for (int i = 0; i < a.inner(); i++) {
          acc[o * a.inner() + i] += x.doubleAt(base + i);
        }
      }
    }
    return Tensor.ofDoubles(shape, acc);
  }

  static Tensor cumSum(Tensor x, Tensor axisTensor, boolean exclusive, boolean reverse) {
    int axis = x.normalizeAxis((int) axisTensor.longAt(0));
    Axis a = around(x, axis);
    Tensor out = x.emptyLike(x.shape());
    for (int o = 0; o < a.outer(); o++) {
      for (int i = 0; i < a.inner(); i++) {
        double running = 0;
        long runningLong = 0;
        for (int step = 0; step < a.length(); step++) {
          int k = reverse ? a.length() - 1 - step : step;
          int index = (o * a.length() + k) * a.inner() + i;
          if (x.isIntegral()) {
            long value = x.longAt(index);
            long emitted = exclusive ? runningLong : runningLong + value;
            runningLong += value;
            out.longs()[index] = emitted;
          } else {
            double value = x.doubleAt(index);
            double emitted = exclusive ? running : running + value;
            running += value;
            if (x.isDouble()) {
              out.doubles()[index] = emitted;
            } else {
              out.floats()[index] = (float) emitted;
            }
          }
        }
      }
    }
    return out;
  }

  /** Opset-13 {@code Softmax} along one axis. */
  static Tensor softmax(Tensor x, long axisAttribute) {
    int axis = x.normalizeAxis((int) axisAttribute);
    Axis a = around(x, axis);
    Tensor input = x.isDouble() ? x : x.toFloat();
    Tensor out = input.emptyLike(input.shape());
    double[] scratch = new double[a.length()];
    for (int o = 0; o < a.outer(); o++) {
      for (int i = 0; i < a.inner(); i++) {
        double max = Double.NEGATIVE_INFINITY;
        for (int k = 0; k < a.length(); k++) {
          max = Math.max(max, input.doubleAt((o * a.length() + k) * a.inner() + i));
        }
        double sum = 0;
        for (int k = 0; k < a.length(); k++) {
          scratch[k] = Math.exp(input.doubleAt((o * a.length() + k) * a.inner() + i) - max);
          sum += scratch[k];
        }
        for (int k = 0; k < a.length(); k++) {
          int index = (o * a.length() + k) * a.inner() + i;
          if (input.isDouble()) {
            out.doubles()[index] = scratch[k] / sum;
          } else {
            out.floats()[index] = (float) (scratch[k] / sum);
          }
        }
      }
    }
    return out;
  }
}
