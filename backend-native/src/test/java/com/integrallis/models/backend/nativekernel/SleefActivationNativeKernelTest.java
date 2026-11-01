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

import com.integrallis.vectors.core.VectorUtil;
import java.nio.file.Path;
import java.util.SplittableRandom;
import org.junit.jupiter.api.Test;

class SleefActivationNativeKernelTest {
  @Test
  void gatesPreserveOffsetsSpecialValuesAndAliasedOperationOrder() {
    try (var kernel = RustGgufBatchedMatrixKernel.open(libraryPath())) {
      for (int n : new int[] {1, 3, 4, 7, 16, 48, 257}) {
        for (int alias = 0; alias < 6; alias++) {
          float[][] a = arrays(n + 12, 5);
          a[0][4] = Float.POSITIVE_INFINITY;
          a[1][5] = -100f;
          if (alias > 0) a[alias - 1] = a[4];
          float[][] expected = copyAliases(a);
          VectorUtil.sigmoidAndScaledSoftplus(
              expected[0], 2, expected[1], 3, expected[2], 1, expected[3], 4, expected[4], 5, n);
          boolean accepted =
              kernel.trySigmoidAndScaledSoftplus(a[0], 2, a[1], 3, a[2], 1, a[3], 4, a[4], 5, n);
          assertThat(accepted).isEqualTo(arm128());
          if (accepted) for (int i = 0; i < a.length; i++) equalBits(expected[i], a[i]);
        }
      }
    }
  }

  @Test
  void convolutionPreservesHistoryTailsAndAliasesAcrossSuccessiveTokens() {
    try (var kernel = RustGgufBatchedMatrixKernel.open(libraryPath())) {
      for (int n : new int[] {1, 3, 4, 7, 48, 259}) {
        for (int taps : new int[] {2, 3, 4, 5}) {
          for (int alias = 0; alias < 4; alias++) {
            float[][] a = arrays(n * taps + 12, 3);
            if (alias == 1) a[1] = a[0];
            if (alias == 2) a[2] = a[0];
            if (alias == 3) a[2] = a[1];
            float[][] expected = copyAliases(a);
            for (int token = 0; token < 5; token++) {
              VectorUtil.causalDepthwiseConv1dSilu(
                  expected[0], 2, expected[1], 1, expected[2], 3, n, taps);
              boolean accepted =
                  kernel.tryCausalDepthwiseConv1dSilu(a[0], 2, a[1], 1, a[2], 3, n, taps);
              assertThat(accepted).isEqualTo(arm128());
              if (accepted) for (int i = 0; i < a.length; i++) equalBits(expected[i], a[i]);
            }
          }
        }
      }
    }
  }

  @Test
  void preparedViewsMatchArrayRoutesWithoutExtendingProviderLifetime() {
    var kernel = RustGgufBatchedMatrixKernel.open(libraryPath());
    float[][] a = arrays(64, 5);
    float[][] expected = copyAliases(a);
    var sig = kernel.prepareFloatArray(a[0]);
    var soft = kernel.prepareFloatArray(a[1]);
    var bias = kernel.prepareFloatArray(a[2]);
    var scale = kernel.prepareFloatArray(a[3]);
    var out = kernel.prepareFloatArray(a[4]);
    assertThat(sig.values()).isSameAs(a[0]);
    boolean served =
        kernel.trySigmoidAndScaledSoftplus(sig, 1, soft, 2, bias, 3, scale, 4, out, 5, 48);
    boolean arrayServed =
        kernel.trySigmoidAndScaledSoftplus(
            expected[0], 1, expected[1], 2, expected[2], 3, expected[3], 4, expected[4], 5, 48);
    assertThat(served).isEqualTo(arrayServed);
    for (int i = 0; i < a.length; i++) equalBits(expected[i], a[i]);
    assertThat(kernel.tryCausalDepthwiseConv1dSilu(sig, 0, soft, 0, bias, 0, 15, 4))
        .isEqualTo(
            kernel.tryCausalDepthwiseConv1dSilu(
                expected[0], 0, expected[1], 0, expected[2], 0, 15, 4));
    for (int i = 0; i < a.length; i++) equalBits(expected[i], a[i]);
    kernel.close();
    assertThatThrownBy(
            () ->
                kernel.trySigmoidAndScaledSoftplus(sig, 1, soft, 2, bias, 3, scale, 4, out, 5, 48))
        .isInstanceOf(IllegalStateException.class);
    assertThatThrownBy(() -> kernel.tryCausalDepthwiseConv1dSilu(sig, 0, soft, 0, bias, 0, 15, 4))
        .isInstanceOf(IllegalStateException.class);
  }

  @Test
  void validatedLibraryRejectsRangesBeforeWritesAndClosedCalls() {
    var library = NativeKernelLibrary.open(libraryPath());
    float[] a = {1, 2, 3, 4};
    assertThatThrownBy(() -> library.trySigmoidAndScaledSoftplus(a, 0, a, 0, a, 0, a, 0, a, 2, 4))
        .isInstanceOf(IndexOutOfBoundsException.class);
    assertThatThrownBy(() -> library.tryCausalDepthwiseConv1dSilu(a, 0, a, 0, a, 0, 4, 1, true))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(
            () ->
                library.tryCausalDepthwiseConv1dSilu(a, 0, a, 0, a, 0, 4, Integer.MAX_VALUE, true))
        .isInstanceOf(ArithmeticException.class);
    equalBits(new float[] {1, 2, 3, 4}, a);
    library.close();
    assertThatThrownBy(() -> library.trySigmoidAndScaledSoftplus(a, 0, a, 0, a, 0, a, 0, a, 0, 4))
        .isInstanceOf(IllegalStateException.class);
    assertThatThrownBy(() -> library.tryCausalDepthwiseConv1dSilu(a, 0, a, 0, a, 0, 1, 2, true))
        .isInstanceOf(IllegalStateException.class);
  }

  private static boolean arm128() {
    return System.getProperty("os.arch").equals("aarch64")
        && VectorUtil.runtimeCapabilities().activeVectorBits() == 128;
  }

  private static float[][] arrays(int length, int count) {
    var random = new SplittableRandom(389193);
    float[][] a = new float[count][length];
    for (float[] row : a)
      for (int i = 0; i < row.length; i++) row[i] = (float) random.nextDouble(-2, 2);
    return a;
  }

  private static float[][] copyAliases(float[][] arrays) {
    float[][] copy = new float[arrays.length][];
    for (int i = 0; i < arrays.length; i++) {
      copy[i] = arrays[i].clone();
      for (int j = 0; j < i; j++)
        if (arrays[j] == arrays[i]) {
          copy[i] = copy[j];
          break;
        }
    }
    return copy;
  }

  private static void equalBits(float[] expected, float[] actual) {
    for (int i = 0; i < expected.length; i++)
      assertThat(Float.floatToIntBits(actual[i]))
          .as("index %s", i)
          .isEqualTo(Float.floatToIntBits(expected[i]));
  }

  private static Path libraryPath() {
    return Path.of(System.getProperty("models.native.kernels.library"));
  }
}
