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

import java.util.Arrays;

/** Operators that move or select elements without arithmetic: reshape, gather, slice and kin. */
final class Structural {

  private Structural() {}

  /** {@code Shape} with the opset-15 optional {@code start}/{@code end} slice of the dimensions. */
  static Tensor shape(Tensor x, long start, long end) {
    long[] dims = x.shape();
    int rank = dims.length;
    int s = (int) clampAxis(start, rank);
    int e = (int) clampAxis(end == Long.MAX_VALUE ? rank : end, rank);
    return Tensor.ofLongs(
        new long[] {Math.max(0, e - s)}, Arrays.copyOfRange(dims, s, Math.max(s, e)));
  }

  private static long clampAxis(long axis, int rank) {
    long normalized = axis < 0 ? axis + rank : axis;
    return Math.max(0, Math.min(rank, normalized));
  }

  /** {@code Reshape} with {@code 0} copying the input dimension and {@code -1} inferred. */
  static Tensor reshape(Tensor x, Tensor shape) {
    long[] requested = shape.toInt64().longs();
    long[] input = x.shape();
    long[] result = new long[requested.length];
    int infer = -1;
    long known = 1;
    for (int i = 0; i < requested.length; i++) {
      long dim = requested[i];
      if (dim == 0) {
        dim = input[i];
      }
      if (dim == -1) {
        if (infer >= 0) {
          throw new IllegalArgumentException("Reshape has two inferred dimensions");
        }
        infer = i;
      } else {
        known *= dim;
      }
      result[i] = dim;
    }
    if (infer >= 0) {
      result[infer] = known == 0 ? 0 : x.size() / known;
    }
    return x.reshape(result);
  }

  static Tensor flatten(Tensor x, long axis) {
    int a = axis < 0 ? (int) axis + x.rank() : (int) axis;
    long outer = 1;
    for (int i = 0; i < a; i++) {
      outer *= x.dim(i);
    }
    return x.reshape(new long[] {outer, outer == 0 ? 0 : x.size() / outer});
  }

  static Tensor unsqueeze(Tensor x, Tensor axes) {
    long[] requested = axes.toInt64().longs();
    int rank = x.rank() + requested.length;
    boolean[] inserted = new boolean[rank];
    for (long axis : requested) {
      int normalized = (int) (axis < 0 ? axis + rank : axis);
      inserted[normalized] = true;
    }
    long[] shape = new long[rank];
    long[] input = x.shape();
    int source = 0;
    for (int i = 0; i < rank; i++) {
      shape[i] = inserted[i] ? 1 : input[source++];
    }
    return x.reshape(shape);
  }

  static Tensor squeeze(Tensor x, Tensor axes) {
    long[] input = x.shape();
    boolean[] drop = new boolean[input.length];
    if (axes == null) {
      for (int i = 0; i < input.length; i++) {
        drop[i] = input[i] == 1;
      }
    } else {
      for (long axis : axes.toInt64().longs()) {
        drop[x.normalizeAxis((int) axis)] = true;
      }
    }
    int kept = 0;
    for (boolean d : drop) {
      kept += d ? 0 : 1;
    }
    long[] shape = new long[kept];
    int j = 0;
    for (int i = 0; i < input.length; i++) {
      if (!drop[i]) {
        shape[j++] = input[i];
      }
    }
    return x.reshape(shape);
  }

  static Tensor transpose(Tensor x, long[] perm) {
    int rank = x.rank();
    int[] order = new int[rank];
    for (int i = 0; i < rank; i++) {
      order[i] = perm.length == 0 ? rank - 1 - i : (int) perm[i];
    }
    long[] input = x.shape();
    long[] shape = new long[rank];
    long[] inStrides = x.strides();
    long[] strides = new long[rank];
    for (int i = 0; i < rank; i++) {
      shape[i] = input[order[i]];
      strides[i] = inStrides[order[i]];
    }
    Tensor out = x.emptyLike(shape);
    int count = out.size();
    long[] counter = new long[rank];
    long index = 0;
    for (int flat = 0; flat < count; flat++) {
      out.copyElement(x, (int) index, flat);
      for (int axis = rank - 1; axis >= 0; axis--) {
        counter[axis]++;
        index += strides[axis];
        if (counter[axis] < shape[axis]) {
          break;
        }
        index -= strides[axis] * counter[axis];
        counter[axis] = 0;
      }
    }
    return out;
  }

  static Tensor concat(Tensor[] inputs, long axisAttribute) {
    Tensor first = inputs[0];
    int axis = first.normalizeAxis((int) axisAttribute);
    long[] shape = first.shape();
    long total = 0;
    for (Tensor t : inputs) {
      total += t.dim(axis);
    }
    shape[axis] = total;
    Tensor out = first.emptyLike(shape);
    long outer = 1;
    for (int i = 0; i < axis; i++) {
      outer *= shape[i];
    }
    long inner = 1;
    for (int i = axis + 1; i < shape.length; i++) {
      inner *= shape[i];
    }
    int offset = 0;
    for (Tensor t : inputs) {
      Tensor source = t.sameKind(first) ? t : promoteLike(first, t);
      long chunk = t.dim(axis) * inner;
      for (long o = 0; o < outer; o++) {
        int from = (int) (o * chunk);
        int to = (int) (o * total * inner + offset);
        for (int k = 0; k < chunk; k++) {
          out.copyElement(source, from + k, to + k);
        }
      }
      offset += (int) chunk;
    }
    return out;
  }

  private static Tensor promoteLike(Tensor like, Tensor t) {
    if (like.isDouble()) {
      return t.toDouble();
    }
    if (like.isFloat()) {
      return t.toFloat();
    }
    if (like.isBoolean()) {
      return t.toBoolean();
    }
    return t.toInt64();
  }

  /**
   * Opset-13 {@code Slice}: {@code starts}, {@code ends}, optional {@code axes} and {@code steps}.
   */
  static Tensor slice(Tensor x, Tensor starts, Tensor ends, Tensor axes, Tensor steps) {
    int rank = x.rank();
    long[] input = x.shape();
    long[] begin = new long[rank];
    long[] stride = new long[rank];
    long[] shape = input.clone();
    Arrays.fill(stride, 1);
    long[] s = starts.toInt64().longs();
    long[] e = ends.toInt64().longs();
    long[] ax = axes == null ? null : axes.toInt64().longs();
    long[] st = steps == null ? null : steps.toInt64().longs();
    for (int i = 0; i < s.length; i++) {
      int axis = ax == null ? i : x.normalizeAxis((int) ax[i]);
      long dim = input[axis];
      long step = st == null ? 1 : st[i];
      if (step == 0) {
        throw new IllegalArgumentException("Slice step of zero");
      }
      long start = s[i] < 0 ? s[i] + dim : s[i];
      long end = e[i] < 0 ? e[i] + dim : e[i];
      if (step > 0) {
        start = Math.max(0, Math.min(dim, start));
        end = Math.max(0, Math.min(dim, end));
        shape[axis] = Math.max(0, (end - start + step - 1) / step);
      } else {
        start = Math.max(0, Math.min(dim - 1, start));
        end = Math.max(-1, Math.min(dim - 1, end));
        shape[axis] = Math.max(0, (start - end - step - 1) / -step);
      }
      begin[axis] = start;
      stride[axis] = step;
    }
    Tensor out = x.emptyLike(shape);
    long[] inStrides = x.strides();
    int count = out.size();
    long[] counter = new long[rank];
    long base = 0;
    for (int i = 0; i < rank; i++) {
      base += begin[i] * inStrides[i];
    }
    long index = base;
    for (int flat = 0; flat < count; flat++) {
      out.copyElement(x, (int) index, flat);
      for (int axis = rank - 1; axis >= 0; axis--) {
        counter[axis]++;
        index += stride[axis] * inStrides[axis];
        if (counter[axis] < shape[axis]) {
          break;
        }
        index -= stride[axis] * inStrides[axis] * counter[axis];
        counter[axis] = 0;
      }
    }
    return out;
  }

  static Tensor gather(Tensor data, Tensor indices, long axisAttribute) {
    int axis = data.normalizeAxis((int) axisAttribute);
    long[] dshape = data.shape();
    long[] ishape = indices.shape();
    long[] shape = new long[dshape.length - 1 + ishape.length];
    System.arraycopy(dshape, 0, shape, 0, axis);
    System.arraycopy(ishape, 0, shape, axis, ishape.length);
    System.arraycopy(dshape, axis + 1, shape, axis + ishape.length, dshape.length - axis - 1);
    long outer = 1;
    for (int i = 0; i < axis; i++) {
      outer *= dshape[i];
    }
    long dim = dshape[axis];
    long inner = 1;
    for (int i = axis + 1; i < dshape.length; i++) {
      inner *= dshape[i];
    }
    long[] idx = indices.toInt64().longs();
    Tensor out = data.emptyLike(shape);
    int to = 0;
    for (long o = 0; o < outer; o++) {
      for (long index : idx) {
        long k = index < 0 ? index + dim : index;
        if (k < 0 || k >= dim) {
          throw new IndexOutOfBoundsException(
              "Gather index " + index + " outside dimension " + dim);
        }
        int from = (int) ((o * dim + k) * inner);
        for (int j = 0; j < inner; j++) {
          out.copyElement(data, from + j, to++);
        }
      }
    }
    return out;
  }

  static Tensor gatherElements(Tensor data, Tensor indices, long axisAttribute) {
    int axis = data.normalizeAxis((int) axisAttribute);
    long[] shape = indices.shape();
    long[] dstrides = data.strides();
    long dim = data.dim(axis);
    long[] idx = indices.toInt64().longs();
    Tensor out = data.emptyLike(shape);
    int rank = shape.length;
    long[] counter = new long[rank];
    for (int flat = 0; flat < idx.length; flat++) {
      long k = idx[flat] < 0 ? idx[flat] + dim : idx[flat];
      long from = 0;
      for (int a = 0; a < rank; a++) {
        from += (a == axis ? k : counter[a]) * dstrides[a];
      }
      out.copyElement(data, (int) from, flat);
      for (int a = rank - 1; a >= 0; a--) {
        if (++counter[a] < shape[a]) {
          break;
        }
        counter[a] = 0;
      }
    }
    return out;
  }

  static Tensor expand(Tensor x, Tensor shapeTensor) {
    long[] target = Broadcast.shape(x.shape(), shapeTensor.toInt64().longs());
    if (Broadcast.same(x.shape(), target)) {
      return x;
    }
    int[] map = Broadcast.indexMap(x.shape(), target);
    Tensor out = x.emptyLike(target);
    for (int i = 0; i < map.length; i++) {
      out.copyElement(x, map[i], i);
    }
    return out;
  }

  static Tensor constantOfShape(Tensor shapeTensor, Tensor value) {
    long[] shape = shapeTensor.toInt64().longs();
    int count = (int) Tensor.elementCount(shape);
    Tensor fill = value == null ? Tensor.scalar(0f) : value;
    Tensor out = fill.emptyLike(shape);
    for (int i = 0; i < count; i++) {
      out.copyElement(fill, 0, i);
    }
    return out;
  }

  static Tensor range(Tensor start, Tensor limit, Tensor delta) {
    if (start.isFloatingPoint() || limit.isFloatingPoint() || delta.isFloatingPoint()) {
      double s = start.doubleAt(0);
      double d = delta.doubleAt(0);
      int count = (int) Math.max(0, Math.ceil((limit.doubleAt(0) - s) / d));
      if (start.isDouble()) {
        double[] out = new double[count];
        for (int i = 0; i < count; i++) {
          out[i] = s + i * d;
        }
        return Tensor.ofDoubles(new long[] {count}, out);
      }
      float[] out = new float[count];
      for (int i = 0; i < count; i++) {
        out[i] = (float) (s + i * d);
      }
      return Tensor.ofFloats(new long[] {count}, out);
    }
    long s = start.longAt(0);
    long d = delta.longAt(0);
    long l = limit.longAt(0);
    int count = (int) Math.max(0, (l - s + d - (d > 0 ? 1 : -1)) / d);
    long[] out = new long[count];
    for (int i = 0; i < count; i++) {
      out[i] = s + i * d;
    }
    return Tensor.ofLongs(new long[] {count}, out);
  }
}
