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

import jdk.incubator.vector.FloatVector;
import jdk.incubator.vector.VectorOperators;
import jdk.incubator.vector.VectorSpecies;

/** The contiguous single-precision loops every heavy kernel reduces to, on the Vector API. */
final class Simd {

  static final VectorSpecies<Float> F = FloatVector.SPECIES_PREFERRED;
  static final int LANES = F.length();

  private Simd() {}

  /** {@code y[yi + i] += alpha * x[xi + i]} for {@code i < n}. */
  static void axpy(float[] y, int yi, float alpha, float[] x, int xi, int n) {
    int i = 0;
    int bound = F.loopBound(n);
    if (bound > 0) {
      FloatVector av = FloatVector.broadcast(F, alpha);
      for (; i < bound; i += LANES) {
        FloatVector xv = FloatVector.fromArray(F, x, xi + i);
        FloatVector yv = FloatVector.fromArray(F, y, yi + i);
        xv.fma(av, yv).intoArray(y, yi + i);
      }
    }
    for (; i < n; i++) {
      y[yi + i] = Math.fma(alpha, x[xi + i], y[yi + i]);
    }
  }

  /** {@code sum(a[ai + i] * b[bi + i])} for {@code i < n}. */
  static float dot(float[] a, int ai, float[] b, int bi, int n) {
    int i = 0;
    int bound = F.loopBound(n);
    float sum = 0f;
    if (bound > 0) {
      FloatVector acc = FloatVector.zero(F);
      for (; i < bound; i += LANES) {
        acc = FloatVector.fromArray(F, a, ai + i).fma(FloatVector.fromArray(F, b, bi + i), acc);
      }
      sum = acc.reduceLanes(VectorOperators.ADD);
    }
    for (; i < n; i++) {
      sum = Math.fma(a[ai + i], b[bi + i], sum);
    }
    return sum;
  }

  /** {@code out[oi + i] = a[ai + i] * b[bi + i]}. */
  static void mul(float[] out, int oi, float[] a, int ai, float[] b, int bi, int n) {
    int i = 0;
    int bound = F.loopBound(n);
    for (; i < bound; i += LANES) {
      FloatVector.fromArray(F, a, ai + i)
          .mul(FloatVector.fromArray(F, b, bi + i))
          .intoArray(out, oi + i);
    }
    for (; i < n; i++) {
      out[oi + i] = a[ai + i] * b[bi + i];
    }
  }

  /** {@code out[oi + i] = a[ai + i] * s}. */
  static void mulScalar(float[] out, int oi, float[] a, int ai, float s, int n) {
    int i = 0;
    int bound = F.loopBound(n);
    if (bound > 0) {
      FloatVector sv = FloatVector.broadcast(F, s);
      for (; i < bound; i += LANES) {
        FloatVector.fromArray(F, a, ai + i).mul(sv).intoArray(out, oi + i);
      }
    }
    for (; i < n; i++) {
      out[oi + i] = a[ai + i] * s;
    }
  }

  /** {@code out[oi + i] = a[ai + i] / s}, the exact division, not a reciprocal multiply. */
  static void divScalar(float[] out, int oi, float[] a, int ai, float s, int n) {
    int i = 0;
    int bound = F.loopBound(n);
    if (bound > 0) {
      FloatVector sv = FloatVector.broadcast(F, s);
      for (; i < bound; i += LANES) {
        FloatVector.fromArray(F, a, ai + i).div(sv).intoArray(out, oi + i);
      }
    }
    for (; i < n; i++) {
      out[oi + i] = a[ai + i] / s;
    }
  }

  /** {@code out[oi + i] = a[ai + i] + b[bi + i]}. */
  static void add(float[] out, int oi, float[] a, int ai, float[] b, int bi, int n) {
    int i = 0;
    int bound = F.loopBound(n);
    for (; i < bound; i += LANES) {
      FloatVector.fromArray(F, a, ai + i)
          .add(FloatVector.fromArray(F, b, bi + i))
          .intoArray(out, oi + i);
    }
    for (; i < n; i++) {
      out[oi + i] = a[ai + i] + b[bi + i];
    }
  }

  /** {@code out[oi + i] = a[ai + i] + s}. */
  static void addScalar(float[] out, int oi, float[] a, int ai, float s, int n) {
    int i = 0;
    int bound = F.loopBound(n);
    if (bound > 0) {
      FloatVector sv = FloatVector.broadcast(F, s);
      for (; i < bound; i += LANES) {
        FloatVector.fromArray(F, a, ai + i).add(sv).intoArray(out, oi + i);
      }
    }
    for (; i < n; i++) {
      out[oi + i] = a[ai + i] + s;
    }
  }

  /** {@code out[oi + i] = a[ai + i] - b[bi + i]}. */
  static void sub(float[] out, int oi, float[] a, int ai, float[] b, int bi, int n) {
    int i = 0;
    int bound = F.loopBound(n);
    for (; i < bound; i += LANES) {
      FloatVector.fromArray(F, a, ai + i)
          .sub(FloatVector.fromArray(F, b, bi + i))
          .intoArray(out, oi + i);
    }
    for (; i < n; i++) {
      out[oi + i] = a[ai + i] - b[bi + i];
    }
  }

  /** {@code out[oi + i] = a[ai + i] / b[bi + i]}. */
  static void div(float[] out, int oi, float[] a, int ai, float[] b, int bi, int n) {
    int i = 0;
    int bound = F.loopBound(n);
    for (; i < bound; i += LANES) {
      FloatVector.fromArray(F, a, ai + i)
          .div(FloatVector.fromArray(F, b, bi + i))
          .intoArray(out, oi + i);
    }
    for (; i < n; i++) {
      out[oi + i] = a[ai + i] / b[bi + i];
    }
  }

  /** Four dot products sharing one pass over {@code b}: {@code out[oi + r] = a_r . b}. */
  static void dot4(
      float[] a, int a0, int a1, int a2, int a3, float[] b, int bi, int n, float[] out, int oi) {
    int i = 0;
    int bound = F.loopBound(n);
    float s0 = 0f;
    float s1 = 0f;
    float s2 = 0f;
    float s3 = 0f;
    if (bound > 0) {
      FloatVector c0 = FloatVector.zero(F);
      FloatVector c1 = FloatVector.zero(F);
      FloatVector c2 = FloatVector.zero(F);
      FloatVector c3 = FloatVector.zero(F);
      for (; i < bound; i += LANES) {
        FloatVector bv = FloatVector.fromArray(F, b, bi + i);
        c0 = FloatVector.fromArray(F, a, a0 + i).fma(bv, c0);
        c1 = FloatVector.fromArray(F, a, a1 + i).fma(bv, c1);
        c2 = FloatVector.fromArray(F, a, a2 + i).fma(bv, c2);
        c3 = FloatVector.fromArray(F, a, a3 + i).fma(bv, c3);
      }
      s0 = c0.reduceLanes(VectorOperators.ADD);
      s1 = c1.reduceLanes(VectorOperators.ADD);
      s2 = c2.reduceLanes(VectorOperators.ADD);
      s3 = c3.reduceLanes(VectorOperators.ADD);
    }
    for (; i < n; i++) {
      float bv = b[bi + i];
      s0 = Math.fma(a[a0 + i], bv, s0);
      s1 = Math.fma(a[a1 + i], bv, s1);
      s2 = Math.fma(a[a2 + i], bv, s2);
      s3 = Math.fma(a[a3 + i], bv, s3);
    }
    out[oi] = s0;
    out[oi + 1] = s1;
    out[oi + 2] = s2;
    out[oi + 3] = s3;
  }

  /**
   * {@code C[i][j] += sum_p A[i][p] * B[p][j]} for row-major A ({@code lda}), B ({@code ldb}) and C
   * ({@code ldc}), with a 4-row by two-vector register tile over the full {@code k} and plain loops
   * for the row and column remainders.
   */
  static void gemm(
      float[] a,
      int aOff,
      int lda,
      float[] b,
      int bOff,
      int ldb,
      float[] c,
      int cOff,
      int ldc,
      int m,
      int k,
      int n) {
    OnnxKernels kernels = KernelScope.current();
    if (kernels != null && (long) m * k * n >= KernelScope.NATIVE_GEMM_MIN_MACS) {
      kernels.gemm(a, aOff, lda, b, bOff, ldb, c, cOff, ldc, m, k, n);
      return;
    }
    int tile = 2 * LANES;
    int nBound = n - n % tile;
    int i = 0;
    for (; i + 4 <= m; i += 4) {
      int r0 = aOff + i * lda;
      int r1 = r0 + lda;
      int r2 = r1 + lda;
      int r3 = r2 + lda;
      int o0 = cOff + i * ldc;
      int o1 = o0 + ldc;
      int o2 = o1 + ldc;
      int o3 = o2 + ldc;
      for (int j = 0; j < nBound; j += tile) {
        FloatVector c00 = FloatVector.fromArray(F, c, o0 + j);
        FloatVector c01 = FloatVector.fromArray(F, c, o0 + j + LANES);
        FloatVector c10 = FloatVector.fromArray(F, c, o1 + j);
        FloatVector c11 = FloatVector.fromArray(F, c, o1 + j + LANES);
        FloatVector c20 = FloatVector.fromArray(F, c, o2 + j);
        FloatVector c21 = FloatVector.fromArray(F, c, o2 + j + LANES);
        FloatVector c30 = FloatVector.fromArray(F, c, o3 + j);
        FloatVector c31 = FloatVector.fromArray(F, c, o3 + j + LANES);
        for (int p = 0; p < k; p++) {
          int bRow = bOff + p * ldb + j;
          FloatVector b0 = FloatVector.fromArray(F, b, bRow);
          FloatVector b1 = FloatVector.fromArray(F, b, bRow + LANES);
          FloatVector a0 = FloatVector.broadcast(F, a[r0 + p]);
          FloatVector a1 = FloatVector.broadcast(F, a[r1 + p]);
          FloatVector a2 = FloatVector.broadcast(F, a[r2 + p]);
          FloatVector a3 = FloatVector.broadcast(F, a[r3 + p]);
          c00 = a0.fma(b0, c00);
          c01 = a0.fma(b1, c01);
          c10 = a1.fma(b0, c10);
          c11 = a1.fma(b1, c11);
          c20 = a2.fma(b0, c20);
          c21 = a2.fma(b1, c21);
          c30 = a3.fma(b0, c30);
          c31 = a3.fma(b1, c31);
        }
        c00.intoArray(c, o0 + j);
        c01.intoArray(c, o0 + j + LANES);
        c10.intoArray(c, o1 + j);
        c11.intoArray(c, o1 + j + LANES);
        c20.intoArray(c, o2 + j);
        c21.intoArray(c, o2 + j + LANES);
        c30.intoArray(c, o3 + j);
        c31.intoArray(c, o3 + j + LANES);
      }
      if (nBound < n) {
        for (int p = 0; p < k; p++) {
          int bRow = bOff + p * ldb;
          axpy(c, o0 + nBound, a[r0 + p], b, bRow + nBound, n - nBound);
          axpy(c, o1 + nBound, a[r1 + p], b, bRow + nBound, n - nBound);
          axpy(c, o2 + nBound, a[r2 + p], b, bRow + nBound, n - nBound);
          axpy(c, o3 + nBound, a[r3 + p], b, bRow + nBound, n - nBound);
        }
      }
    }
    for (; i < m; i++) {
      int r = aOff + i * lda;
      int o = cOff + i * ldc;
      for (int p = 0; p < k; p++) {
        float av = a[r + p];
        if (av != 0f) {
          axpy(c, o, av, b, bOff + p * ldb, n);
        }
      }
    }
  }

  /** {@code sum(a[ai + i])} for {@code i < n}. */
  static float sum(float[] a, int ai, int n) {
    int i = 0;
    int bound = F.loopBound(n);
    float total = 0f;
    if (bound > 0) {
      FloatVector acc = FloatVector.zero(F);
      for (; i < bound; i += LANES) {
        acc = acc.add(FloatVector.fromArray(F, a, ai + i));
      }
      total = acc.reduceLanes(VectorOperators.ADD);
    }
    for (; i < n; i++) {
      total += a[ai + i];
    }
    return total;
  }

  /**
   * {@code out[i] = op(in[i])} for a lane-wise transcendental, with the scalar tail from {@code
   * java.lang.Math}.
   */
  static void unary(VectorOperators.Unary op, float[] in, float[] out) {
    int n = in.length;
    int i = 0;
    int bound = F.loopBound(n);
    for (; i < bound; i += LANES) {
      FloatVector.fromArray(F, in, i).lanewise(op).intoArray(out, i);
    }
    for (; i < n; i++) {
      out[i] = scalar(op, in[i]);
    }
  }

  static float scalar(VectorOperators.Unary op, float v) {
    if (op == VectorOperators.SIN) {
      return (float) Math.sin(v);
    }
    if (op == VectorOperators.COS) {
      return (float) Math.cos(v);
    }
    if (op == VectorOperators.EXP) {
      return (float) Math.exp(v);
    }
    if (op == VectorOperators.TANH) {
      return (float) Math.tanh(v);
    }
    if (op == VectorOperators.SQRT) {
      return (float) Math.sqrt(v);
    }
    throw new IllegalArgumentException("no scalar fallback for " + op);
  }
}
