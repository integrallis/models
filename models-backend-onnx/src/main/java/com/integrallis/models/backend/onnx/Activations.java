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
 * The executor's single routing point for exp, tanh, logistic and erf over float arrays: the bound
 * {@link OnnxKernels} when there is one, otherwise {@link VectorMath}. Both compute onnxruntime's
 * MLAS formulas. The native route matters for latency as well as throughput: the Vector API
 * translation runs at full speed only once C2 has compiled it, and measured from a cold JVM its
 * first erf call over one GELU activation took 942 ms where the compiled call takes 0.4 ms, which
 * lands squarely on a speech model's first sentence.
 */
final class Activations {
  private Activations() {}

  static void exp(float[] x, int xOffset, float[] y, int yOffset, int n) {
    OnnxKernels kernels = KernelScope.current();
    if (kernels != null) {
      kernels.exp(x, xOffset, y, yOffset, n);
    } else {
      VectorMath.exp(x, xOffset, y, yOffset, n);
    }
  }

  static void tanh(float[] x, int xOffset, float[] y, int yOffset, int n) {
    OnnxKernels kernels = KernelScope.current();
    if (kernels != null) {
      kernels.tanh(x, xOffset, y, yOffset, n);
    } else {
      VectorMath.tanh(x, xOffset, y, yOffset, n);
    }
  }

  static void logistic(float[] x, int xOffset, float[] y, int yOffset, int n) {
    OnnxKernels kernels = KernelScope.current();
    if (kernels != null) {
      kernels.logistic(x, xOffset, y, yOffset, n);
    } else {
      VectorMath.logistic(x, xOffset, y, yOffset, n);
    }
  }

  static void erf(float[] x, float[] y, int n) {
    OnnxKernels kernels = KernelScope.current();
    if (kernels != null) {
      kernels.erf(x, y, n);
    } else {
      VectorMath.erf(x, y, n);
    }
  }
}
