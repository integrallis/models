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

import static org.assertj.core.api.Assertions.assertThat;

import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/**
 * Transpose, Slice and Pad copy whole rows; these compare them, over randomized shapes, against
 * per-element references written from the ONNX definitions, for float, double and int64 storage.
 */
class RowCopyEquivalenceTest {
  private final SplittableRandom random = new SplittableRandom(20261103);

  private long[] randomShape(int rank) {
    long[] shape = new long[rank];
    for (int i = 0; i < rank; i++) {
      shape[i] = 1 + random.nextInt(5);
    }
    return shape;
  }

  private Tensor randomTensor(long[] shape, int kind) {
    int n = (int) Tensor.elementCount(shape);
    switch (kind) {
      case 0 -> {
        float[] v = new float[n];
        for (int i = 0; i < n; i++) {
          v[i] = i + 0.25f;
        }
        return Tensor.ofFloats(shape, v);
      }
      case 1 -> {
        double[] v = new double[n];
        for (int i = 0; i < n; i++) {
          v[i] = i + 0.5;
        }
        return Tensor.ofDoubles(shape, v);
      }
      default -> {
        long[] v = new long[n];
        for (int i = 0; i < n; i++) {
          v[i] = 1000L + i;
        }
        return Tensor.ofLongs(shape, v);
      }
    }
  }

  private static int flat(long[] index, long[] shape) {
    long f = 0;
    for (int a = 0; a < shape.length; a++) {
      f = f * shape[a] + index[a];
    }
    return (int) f;
  }

  private static boolean next(long[] index, long[] shape) {
    for (int a = shape.length - 1; a >= 0; a--) {
      if (++index[a] < shape[a]) {
        return true;
      }
      index[a] = 0;
    }
    return false;
  }

  private static void assertSameValues(Tensor got, long[] shape, double[] want) {
    assertThat(got.shape()).isEqualTo(shape);
    for (int i = 0; i < want.length; i++) {
      assertThat(got.doubleAt(i)).as("element %d", i).isEqualTo(want[i]);
    }
  }

  @Test
  void transposeMatchesThePermutationDefinition() {
    for (int trial = 0; trial < 300; trial++) {
      int rank = 1 + random.nextInt(4);
      long[] shape = randomShape(rank);
      Tensor x = randomTensor(shape, trial % 3);
      long[] perm = new long[rank];
      for (int i = 0; i < rank; i++) {
        perm[i] = i;
      }
      for (int i = rank - 1; i > 0; i--) {
        int j = random.nextInt(i + 1);
        long t = perm[i];
        perm[i] = perm[j];
        perm[j] = t;
      }
      long[] outShape = new long[rank];
      for (int i = 0; i < rank; i++) {
        outShape[i] = shape[(int) perm[i]];
      }
      double[] want = new double[x.size()];
      long[] o = new long[rank];
      do {
        long[] src = new long[rank];
        for (int i = 0; i < rank; i++) {
          src[(int) perm[i]] = o[i];
        }
        want[flat(o, outShape)] = x.doubleAt(flat(src, shape));
      } while (next(o, outShape));
      assertSameValues(Structural.transpose(x, perm), outShape, want);
    }
  }

  @Test
  void sliceMatchesStartEndStepIncludingNegativeSteps() {
    for (int trial = 0; trial < 300; trial++) {
      int rank = 1 + random.nextInt(4);
      long[] shape = randomShape(rank);
      Tensor x = randomTensor(shape, trial % 3);
      long[] starts = new long[rank];
      long[] ends = new long[rank];
      long[] steps = new long[rank];
      long[] begin = new long[rank];
      long[] outShape = new long[rank];
      for (int a = 0; a < rank; a++) {
        long dim = shape[a];
        long step = random.nextBoolean() ? 1 + random.nextInt(3) : -(1 + random.nextInt(3));
        steps[a] = step;
        if (step > 0) {
          long s = random.nextInt((int) dim);
          long e = s + 1 + random.nextInt((int) (dim - s));
          starts[a] = s;
          ends[a] = e;
          begin[a] = s;
          outShape[a] = (e - s + step - 1) / step;
        } else {
          long s = random.nextInt((int) dim);
          long e = s - 1 - random.nextInt((int) (s + 1)); // down to -1, exclusive
          starts[a] = s;
          ends[a] = e < 0 ? -dim - 1 : e; // a negative end below -dim means "past the start"
          begin[a] = s;
          outShape[a] = (s - e - step - 1) / -step;
        }
      }
      double[] want = new double[(int) Tensor.elementCount(outShape)];
      if (want.length > 0) {
        long[] o = new long[rank];
        do {
          long[] src = new long[rank];
          for (int a = 0; a < rank; a++) {
            src[a] = begin[a] + o[a] * steps[a];
          }
          want[flat(o, outShape)] = x.doubleAt(flat(src, shape));
        } while (next(o, outShape));
      }
      Tensor got =
          Structural.slice(
              x,
              Tensor.ofLongs(new long[] {rank}, starts),
              Tensor.ofLongs(new long[] {rank}, ends),
              null,
              Tensor.ofLongs(new long[] {rank}, steps));
      assertSameValues(got, outShape, want);
    }
  }

  @Test
  void padMatchesReflectEdgeAndConstantDefinitions() {
    String[] modes = {"reflect", "edge", "constant"};
    for (int trial = 0; trial < 300; trial++) {
      int rank = 1 + random.nextInt(3);
      long[] shape = randomShape(rank);
      Tensor x = randomTensor(shape, trial % 3);
      String mode = modes[trial % 3];
      long[] pads = new long[2 * rank];
      long[] outShape = new long[rank];
      for (int a = 0; a < rank; a++) {
        // Reflect needs pads below the dimension; edge and constant take any non-negative pad.
        int limit = mode.equals("reflect") ? (int) shape[a] : 4;
        pads[a] = random.nextInt(limit);
        pads[rank + a] = random.nextInt(limit);
        outShape[a] = shape[a] + pads[a] + pads[rank + a];
      }
      double[] want = new double[(int) Tensor.elementCount(outShape)];
      long[] o = new long[rank];
      do {
        long[] src = new long[rank];
        boolean inside = true;
        for (int a = 0; a < rank; a++) {
          long i = o[a] - pads[a];
          long dim = shape[a];
          if (i >= 0 && i < dim) {
            src[a] = i;
          } else if (mode.equals("constant")) {
            inside = false;
          } else if (mode.equals("edge")) {
            src[a] = i < 0 ? 0 : dim - 1;
          } else {
            src[a] = i < 0 ? -i : 2 * (dim - 1) - i;
          }
        }
        want[flat(o, outShape)] = inside ? x.doubleAt(flat(src, shape)) : 0.0;
      } while (next(o, outShape));
      Tensor got = Resampling.pad(x, Tensor.ofLongs(new long[] {2L * rank}, pads), null, mode);
      assertSameValues(got, outShape, want);
    }
  }
}
