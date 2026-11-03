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

import java.util.function.DoubleUnaryOperator;
import org.junit.jupiter.api.Test;

class VectorMathTest {

  interface Kernel {
    void apply(float[] in, float[] out, int n);
  }

  interface Scalar {
    float apply(float x);
  }

  /** A dense sweep of {@code [lo, hi]} with a length that leaves a scalar tail on every species. */
  private static float[] sweep(float lo, float hi) {
    int n = 200_003;
    float[] xs = new float[n];
    for (int i = 0; i < n; i++) {
      xs[i] = lo + (hi - lo) * i / (n - 1);
    }
    return xs;
  }

  private static double maxError(
      float[] xs, Kernel kernel, DoubleUnaryOperator reference, boolean relative) {
    float[] out = new float[xs.length];
    kernel.apply(xs, out, xs.length);
    double worst = 0;
    for (int i = 0; i < xs.length; i++) {
      double want = reference.applyAsDouble(xs[i]);
      double err = Math.abs(out[i] - want);
      if (relative) {
        err /= Math.max(Math.abs(want), Double.MIN_NORMAL);
      }
      worst = Math.max(worst, err);
    }
    return worst;
  }

  /** The vector body and the scalar tail compute the same value for the same input. */
  private static void assertBodyMatchesTail(float[] xs, Kernel kernel, Scalar scalar) {
    float[] out = new float[xs.length];
    kernel.apply(xs, out, xs.length);
    for (int i = 0; i < xs.length; i++) {
      assertThat(Float.floatToRawIntBits(out[i]))
          .as("x=%s", xs[i])
          .isEqualTo(Float.floatToRawIntBits(scalar.apply(xs[i])));
    }
  }

  private static double erf(double x) {
    return Elementwise.erf(x);
  }

  @Test
  void expTracksMathExpToAFewUlpsAcrossItsRange() {
    float[] xs = sweep(-87f, 88f);
    assertThat(maxError(xs, VectorMath::exp, Math::exp, true)).isLessThan(5e-7);
    assertBodyMatchesTail(xs, VectorMath::exp, VectorMath::expScalar);
  }

  @Test
  void tanhTracksMathTanh() {
    float[] xs = sweep(-12f, 12f);
    assertThat(maxError(xs, VectorMath::tanh, Math::tanh, false)).isLessThan(1e-6);
    assertBodyMatchesTail(xs, VectorMath::tanh, VectorMath::tanhScalar);
  }

  @Test
  void logisticTracksTheExactSigmoid() {
    float[] xs = sweep(-25f, 25f);
    assertThat(maxError(xs, VectorMath::logistic, x -> 1.0 / (1.0 + Math.exp(-x)), false))
        .isLessThan(1e-6);
    assertBodyMatchesTail(xs, VectorMath::logistic, VectorMath::logisticScalar);
  }

  @Test
  void erfTracksTheDoublePrecisionErf() {
    float[] xs = sweep(-6f, 6f);
    assertThat(maxError(xs, VectorMath::erf, VectorMathTest::erf, false)).isLessThan(1e-6);
    assertBodyMatchesTail(xs, VectorMath::erf, VectorMath::erfScalar);
  }

  @Test
  void specialValuesFollowTheReferenceRuntime() {
    float[] xs = {
      0f, -0f, Float.NaN, Float.POSITIVE_INFINITY, Float.NEGATIVE_INFINITY, 1e-30f, -1e-30f, 0f
    };
    float[] out = new float[xs.length];
    VectorMath.erf(xs, out, xs.length);
    assertThat(Float.floatToRawIntBits(out[1])).isEqualTo(Float.floatToRawIntBits(-0f));
    assertThat(out[3]).isEqualTo(1f);
    assertThat(out[4]).isEqualTo(-1f);
    VectorMath.tanh(xs, out, xs.length);
    assertThat(out[2]).isNaN();
    assertThat(out[3]).isEqualTo(1f, org.assertj.core.data.Offset.offset(1e-6f));
    VectorMath.exp(xs, out, xs.length);
    assertThat(out[0]).isEqualTo(1f);
    assertThat(out[4]).isEqualTo(0f);
    VectorMath.logistic(xs, out, xs.length);
    assertThat(out[0]).isEqualTo(0.5f);
    assertThat(out[3]).isEqualTo(1f, org.assertj.core.data.Offset.offset(1e-6f));
  }

  @Test
  void offsetsAddressTheRequestedWindowOnly() {
    float[] in = {9f, 1f, 2f, 9f};
    float[] out = new float[4];
    VectorMath.tanh(in, 1, out, 2, 2);
    assertThat(out[0]).isZero();
    assertThat(out[1]).isZero();
    assertThat(out[2]).isEqualTo(VectorMath.tanhScalar(1f));
    assertThat(out[3]).isEqualTo(VectorMath.tanhScalar(2f));
  }
}
