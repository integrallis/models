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

/** Numpy-style broadcasting: result shapes and the element index mapping behind them. */
final class Broadcast {

  private Broadcast() {}

  /** The broadcast shape of two shapes, or an exception when they are incompatible. */
  static long[] shape(long[] a, long[] b) {
    int rank = Math.max(a.length, b.length);
    long[] result = new long[rank];
    for (int i = 0; i < rank; i++) {
      long da = i < rank - a.length ? 1 : a[i - (rank - a.length)];
      long db = i < rank - b.length ? 1 : b[i - (rank - b.length)];
      if (da != db && da != 1 && db != 1) {
        throw new IllegalArgumentException(
            "cannot broadcast " + Arrays.toString(a) + " with " + Arrays.toString(b));
      }
      result[i] = da == 1 ? db : da;
    }
    return result;
  }

  /**
   * Strides of {@code shape} aligned to {@code target}'s rank, with zero where the dimension is
   * broadcast, so that an output multi-index dotted with them gives the input flat index.
   */
  static long[] strides(long[] shape, long[] target) {
    long[] own = Tensor.strides(shape);
    long[] aligned = new long[target.length];
    int offset = target.length - shape.length;
    for (int i = 0; i < shape.length; i++) {
      aligned[offset + i] = shape[i] == 1 && target[offset + i] != 1 ? 0 : own[i];
    }
    return aligned;
  }

  /**
   * The flat input indices, in output order, for an input of {@code shape} broadcast to {@code
   * target}.
   */
  static int[] indexMap(long[] shape, long[] target) {
    long[] strides = strides(shape, target);
    int count = (int) Tensor.elementCount(target);
    int[] map = new int[count];
    int rank = target.length;
    long[] counter = new long[rank];
    long index = 0;
    for (int flat = 0; flat < count; flat++) {
      map[flat] = (int) index;
      // Advance the odometer from the last axis.
      for (int axis = rank - 1; axis >= 0; axis--) {
        counter[axis]++;
        index += strides[axis];
        if (counter[axis] < target[axis]) {
          break;
        }
        index -= strides[axis] * counter[axis];
        counter[axis] = 0;
      }
    }
    return map;
  }

  /** Whether {@code shape} equals {@code target}, in which case the index map is the identity. */
  static boolean same(long[] shape, long[] target) {
    return Arrays.equals(shape, target);
  }
}
