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

/** {@code MatMul} with batch broadcasting and {@code Gemm}. */
final class Linear {

  private Linear() {}

  /** Numpy-style {@code MatMul}: 1-D operands are promoted, leading batch dims broadcast. */
  static Tensor matMul(Tensor a, Tensor b) {
    boolean promoteA = a.rank() == 1;
    boolean promoteB = b.rank() == 1;
    long[] sa = promoteA ? new long[] {1, a.dim(0)} : a.shape();
    long[] sb = promoteB ? new long[] {b.dim(0), 1} : b.shape();
    int m = (int) sa[sa.length - 2];
    int k = (int) sa[sa.length - 1];
    int kb = (int) sb[sb.length - 2];
    int n = (int) sb[sb.length - 1];
    if (k != kb) {
      throw new IllegalArgumentException(
          "MatMul inner dimensions differ: "
              + Arrays.toString(a.shape())
              + " x "
              + Arrays.toString(b.shape()));
    }
    long[] batchA = Arrays.copyOf(sa, sa.length - 2);
    long[] batchB = Arrays.copyOf(sb, sb.length - 2);
    long[] batch = Broadcast.shape(batchA, batchB);
    int batches = (int) Tensor.elementCount(batch);
    int[] mapA = Broadcast.indexMap(batchA, batch);
    int[] mapB = Broadcast.indexMap(batchB, batch);
    boolean dbl = a.isDouble() || b.isDouble();
    float[] fa = dbl ? null : a.toFloat().floats();
    float[] fb = dbl ? null : b.toFloat().floats();
    double[] da = dbl ? a.toDouble().doubles() : null;
    double[] db = dbl ? b.toDouble().doubles() : null;
    float[] outF = dbl ? null : new float[batches * m * n];
    double[] outD = dbl ? new double[batches * m * n] : null;
    for (int batchIndex = 0; batchIndex < batches; batchIndex++) {
      int baseA = mapA[batchIndex] * m * k;
      int baseB = mapB[batchIndex] * k * n;
      int baseO = batchIndex * m * n;
      if (dbl) {
        for (int i = 0; i < m; i++) {
          for (int p = 0; p < k; p++) {
            double av = da[baseA + i * k + p];
            int rowB = baseB + p * n;
            int rowO = baseO + i * n;
            for (int j = 0; j < n; j++) {
              outD[rowO + j] += av * db[rowB + j];
            }
          }
        }
      } else {
        Simd.gemm(fa, baseA, k, fb, baseB, n, outF, baseO, n, m, k, n);
      }
    }
    long[] shape;
    if (promoteA && promoteB) {
      shape = batch;
    } else if (promoteA) {
      shape = Arrays.copyOf(batch, batch.length + 1);
      shape[batch.length] = n;
    } else if (promoteB) {
      shape = Arrays.copyOf(batch, batch.length + 1);
      shape[batch.length] = m;
    } else {
      shape = Arrays.copyOf(batch, batch.length + 2);
      shape[batch.length] = m;
      shape[batch.length + 1] = n;
    }
    return dbl ? Tensor.ofDoubles(shape, outD) : Tensor.ofFloats(shape, outF);
  }

  /** {@code Gemm}: {@code alpha * op(A) * op(B) + beta * C}, with C broadcast to {@code [M, N]}. */
  static Tensor gemm(
      Tensor a, Tensor b, Tensor c, float alpha, float beta, boolean transA, boolean transB) {
    int m = (int) (transA ? a.dim(1) : a.dim(0));
    int k = (int) (transA ? a.dim(0) : a.dim(1));
    int n = (int) (transB ? b.dim(0) : b.dim(1));
    if (k != (transB ? b.dim(1) : b.dim(0))) {
      throw new IllegalArgumentException("Gemm inner dimensions differ");
    }
    float[] fa = a.toFloat().floats();
    float[] fb = b.toFloat().floats();
    float[] out = new float[m * n];
    if (!transA && transB) {
      // The common linear-layer layout: rows of A dotted with rows of B, four rows per pass.
      float[] four = new float[4];
      int i = 0;
      for (; i + 4 <= m; i += 4) {
        int rowA = i * k;
        for (int j = 0; j < n; j++) {
          Simd.dot4(fa, rowA, rowA + k, rowA + 2 * k, rowA + 3 * k, fb, j * k, k, four, 0);
          out[i * n + j] = alpha * four[0];
          out[(i + 1) * n + j] = alpha * four[1];
          out[(i + 2) * n + j] = alpha * four[2];
          out[(i + 3) * n + j] = alpha * four[3];
        }
      }
      for (; i < m; i++) {
        for (int j = 0; j < n; j++) {
          out[i * n + j] = alpha * Simd.dot(fa, i * k, fb, j * k, k);
        }
      }
    } else if (!transA) {
      Simd.gemm(fa, 0, k, fb, 0, n, out, 0, n, m, k, n);
      if (alpha != 1f) {
        for (int i = 0; i < out.length; i++) {
          out[i] *= alpha;
        }
      }
    } else {
      for (int i = 0; i < m; i++) {
        for (int p = 0; p < k; p++) {
          float av = fa[p * m + i];
          for (int j = 0; j < n; j++) {
            float bv = transB ? fb[j * k + p] : fb[p * n + j];
            out[i * n + j] += av * bv;
          }
        }
      }
      if (alpha != 1f) {
        for (int i = 0; i < out.length; i++) {
          out[i] *= alpha;
        }
      }
    }
    if (c != null && beta != 0f) {
      long[] shape = new long[] {m, n};
      int[] map = Broadcast.same(c.shape(), shape) ? null : Broadcast.indexMap(c.shape(), shape);
      for (int i = 0; i < out.length; i++) {
        out[i] += beta * c.floatAt(map == null ? i : map[i]);
      }
    }
    return Tensor.ofFloats(new long[] {m, n}, out);
  }
}
