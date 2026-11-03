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
 * Optional accelerated kernels for the executor's heaviest operations. An executor given an
 * implementation routes every sufficiently large matrix product (MatMul, Gemm, the im2col form of
 * every Conv, the wide ConvTranspose path) and every float {@code Sin} through it; an executor
 * without one runs the pure-Java Vector API kernels. Implementations must produce the same results
 * as the Java kernels to within float rounding; seeded parity against onnxruntime is the check.
 */
public interface OnnxKernels {

  /** A short name for diagnostics, e.g. {@code "rust-ffm"}. */
  String name();

  /**
   * {@code C[i][j] += sum_p A[i][p] * B[p][j]} for row-major matrices with leading dimensions
   * {@code lda}, {@code ldb} and {@code ldc}, starting at the given offsets.
   */
  void gemm(
      float[] a,
      int aOffset,
      int lda,
      float[] b,
      int bOffset,
      int ldb,
      float[] c,
      int cOffset,
      int ldc,
      int m,
      int k,
      int n);

  /** {@code y[i] = sin(x[i])} for the first {@code n} elements, matching {@code Math.sin}. */
  void sin(float[] x, float[] y, int n);

  /**
   * {@code y[i] = erf(x[i])} for the first {@code n} elements, with onnxruntime's MLAS
   * approximation ({@code MlasErfKernel}).
   */
  void erf(float[] x, float[] y, int n);

  /**
   * {@code y[yOffset + i] = exp(x[xOffset + i])} for {@code i < n}, with onnxruntime's MLAS
   * approximation ({@code MlasComputeExpVector}). The default is the executor's Vector API
   * translation of the same formula.
   */
  default void exp(float[] x, int xOffset, float[] y, int yOffset, int n) {
    VectorMath.exp(x, xOffset, y, yOffset, n);
  }

  /** As {@link #exp}, for tanh ({@code MlasTanhKernel}). */
  default void tanh(float[] x, int xOffset, float[] y, int yOffset, int n) {
    VectorMath.tanh(x, xOffset, y, yOffset, n);
  }

  /** As {@link #exp}, for the logistic sigmoid ({@code MlasLogisticKernel}). */
  default void logistic(float[] x, int xOffset, float[] y, int yOffset, int n) {
    VectorMath.logistic(x, xOffset, y, yOffset, n);
  }
}
