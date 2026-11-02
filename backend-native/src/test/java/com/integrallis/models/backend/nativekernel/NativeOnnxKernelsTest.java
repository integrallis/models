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
package com.integrallis.models.backend.nativekernel;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.file.Path;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

/** The Rust ONNX kernels against plain Java references, through the real bundled library. */
class NativeOnnxKernelsTest {

  private static NativeOnnxKernels kernels() {
    return NativeOnnxKernels.load(Path.of(System.getProperty("models.native.kernels.library")));
  }

  @Test
  void gemmAccumulatesLikeTheJavaReferenceAtOffsetsAndLeadingDimensions() {
    NativeOnnxKernels k = kernels();
    SplittableRandom random = new SplittableRandom(7);
    int m = 37, kk = 53, n = 71, lda = kk + 3, ldb = n + 5, ldc = n + 2, ao = 4, bo = 9, co = 6;
    float[] a = new float[ao + m * lda];
    float[] b = new float[bo + kk * ldb];
    float[] c = new float[co + m * ldc];
    for (int i = 0; i < a.length; i++) a[i] = (float) random.nextDouble(-1, 1);
    for (int i = 0; i < b.length; i++) b[i] = (float) random.nextDouble(-1, 1);
    for (int i = 0; i < c.length; i++) c[i] = (float) random.nextDouble(-1, 1);
    float[] expected = c.clone();
    for (int i = 0; i < m; i++) {
      for (int j = 0; j < n; j++) {
        double acc = 0;
        for (int p = 0; p < kk; p++) acc += a[ao + i * lda + p] * b[bo + p * ldb + j];
        expected[co + i * ldc + j] += (float) acc;
      }
    }
    k.gemm(a, ao, lda, b, bo, ldb, c, co, ldc, m, kk, n);
    for (int i = 0; i < c.length; i++) {
      assertThat(c[i])
          .as("element " + i)
          .isCloseTo(expected[i], org.assertj.core.data.Offset.offset(1e-4f));
    }
    assertThatThrownBy(() -> k.gemm(a, 0, lda, b, 0, ldb, new float[3], 0, ldc, m, kk, n))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void sineMatchesMathSinRoundedToFloatAcrossTheGraphsPhaseRange() {
    NativeOnnxKernels k = kernels();
    float[] x = new float[400_000];
    for (int i = 0; i < x.length; i++) x[i] = -200_000f + i * 1.000_37f;
    float[] y = new float[x.length];
    k.sin(x, y, x.length);
    int worst = 0;
    for (int i = 0; i < x.length; i++) {
      int ulps =
          Math.abs(Float.floatToIntBits(y[i]) - Float.floatToIntBits((float) Math.sin(x[i])));
      worst = Math.max(worst, ulps);
    }
    assertThat(worst).as("worst disagreement with Math.sin in ulps").isLessThanOrEqualTo(1);
    assertThat(k.name()).isEqualTo("rust-ffm");
  }

  @Test
  void erfMatchesKnownValues() {
    float[] x = {0f, 0.5f, 1f, -1.5f, 3f, 8f};
    float[] y = new float[x.length];
    kernels().erf(x, y, x.length);
    // Python math.erf, rounded to float.
    float[] expected = {0f, 0.5204999f, 0.8427008f, -0.9661051f, 0.9999779f, 1f};
    for (int i = 0; i < x.length; i++) {
      assertThat(y[i])
          .as("erf(" + x[i] + ")")
          .isCloseTo(expected[i], org.assertj.core.data.Offset.offset(1e-6f));
    }
  }
}
