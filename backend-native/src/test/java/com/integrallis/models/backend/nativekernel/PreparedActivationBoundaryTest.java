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

import com.integrallis.models.backend.purejava.spi.GgufBatchedMatrixKernel;
import java.lang.foreign.MemorySegment;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;

class PreparedActivationBoundaryTest {
  @Test
  void librarySegmentAndArrayRoutesShareBoundsAndUnsupportedNoWriteContract() {
    var library = NativeKernelLibrary.open(libraryPath());
    float[] a = {1, 2, 3, 4, 5, 6, 7, 8};
    float[] before = a.clone();
    var view = MemorySegment.ofArray(a);
    boolean supported = library.supports(NativeKernelCapability.SLEEF_ACTIVATIONS_F32);
    assertThat(library.trySigmoidAndScaledSoftplus(view, 0, view, 0, view, 0, view, 0, view, 0, 0))
        .isFalse();
    assertThat(library.tryCausalDepthwiseConv1dSilu(view, 0, view, 0, view, 0, 0, 2, true))
        .isFalse();
    assertThat(library.trySigmoidAndScaledSoftplus(a, 0, a, 0, a, 0, a, 0, a, 0, 0)).isFalse();
    assertThat(library.tryCausalDepthwiseConv1dSilu(a, 0, a, 0, a, 0, 0, 2, true)).isFalse();
    for (int badAt : new int[] {-1, 8, Integer.MAX_VALUE}) {
      assertThatThrownBy(
              () ->
                  library.trySigmoidAndScaledSoftplus(
                      view, badAt, view, 0, view, 0, view, 0, view, 0, 1))
          .isInstanceOf(IndexOutOfBoundsException.class);
      assertThatThrownBy(
              () -> library.tryCausalDepthwiseConv1dSilu(view, badAt, view, 0, view, 0, 1, 2, true))
          .isInstanceOf(IndexOutOfBoundsException.class);
    }
    assertThatThrownBy(
            () -> library.tryCausalDepthwiseConv1dSilu(view, 0, view, 0, view, 0, 4, 1, true))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(
            () ->
                library.tryCausalDepthwiseConv1dSilu(
                    view, 0, view, 0, view, 0, 4, Integer.MAX_VALUE, true))
        .isInstanceOf(ArithmeticException.class);
    assertThat(a).containsExactly(before);
    if (!supported) {
      assertThat(
              library.trySigmoidAndScaledSoftplus(view, 0, view, 0, view, 0, view, 0, view, 0, 4))
          .isFalse();
      assertThat(library.tryCausalDepthwiseConv1dSilu(view, 0, view, 0, view, 0, 2, 4, true))
          .isFalse();
      assertThat(library.trySigmoidAndScaledSoftplus(a, 0, a, 0, a, 0, a, 0, a, 0, 4)).isFalse();
      assertThat(library.tryCausalDepthwiseConv1dSilu(a, 0, a, 0, a, 0, 2, 4, true)).isFalse();
      assertThat(a).containsExactly(before);
    }
    library.close();
    assertThatThrownBy(
            () ->
                library.trySigmoidAndScaledSoftplus(view, 0, view, 0, view, 0, view, 0, view, 0, 1))
        .isInstanceOf(IllegalStateException.class);
    assertThatThrownBy(
            () -> library.tryCausalDepthwiseConv1dSilu(view, 0, view, 0, view, 0, 1, 2, true))
        .isInstanceOf(IllegalStateException.class);
  }

  @Test
  void foreignPreparedViewsAreDeclinedWithoutWritesAndClosedProviderStillRejectsThem() {
    float[] a = {1, 2, 3, 4};
    GgufBatchedMatrixKernel.FloatArrayView foreign = () -> a;
    var provider = RustGgufBatchedMatrixKernel.open(libraryPath());
    var nativeView = provider.prepareFloatArray(a);
    for (int position = 0; position < 5; position++) {
      var v =
          new GgufBatchedMatrixKernel.FloatArrayView[] {
            nativeView, nativeView, nativeView, nativeView, nativeView
          };
      v[position] = foreign;
      assertThat(
              provider.trySigmoidAndScaledSoftplus(v[0], 0, v[1], 0, v[2], 0, v[3], 0, v[4], 0, 4))
          .isFalse();
    }
    for (int position = 0; position < 3; position++) {
      var v = new GgufBatchedMatrixKernel.FloatArrayView[] {nativeView, nativeView, nativeView};
      v[position] = foreign;
      assertThat(provider.trySwiGlu(v[0], 0, v[1], 0, v[2], 0, 4)).isFalse();
      assertThat(provider.tryCausalDepthwiseConv1dSilu(v[0], 0, v[1], 0, v[2], 0, 1, 2)).isFalse();
    }
    assertThat(a).containsExactly(1, 2, 3, 4);
    boolean advertised = provider.supportsRecurrentActivations();
    assertThat(advertised)
        .isEqualTo(
            System.getProperty("os.arch").equals("aarch64")
                && com.integrallis.vectors.core.VectorUtil.runtimeCapabilities().activeVectorBits()
                    == 128);
    assertThatThrownBy(() -> provider.prepareFloatArray(null))
        .isInstanceOf(NullPointerException.class);
    provider.close();
    assertThatThrownBy(() -> provider.prepareFloatArray(a))
        .isInstanceOf(IllegalStateException.class);
    assertThatThrownBy(() -> provider.trySwiGlu(foreign, 0, foreign, 0, foreign, 0, 4))
        .isInstanceOf(IllegalStateException.class);
  }

  @Test
  void recurrentAndAttentionSpiForwardExactBuffersAndRespectClosedLifetime() {
    float[] q = {.25f, -.5f, .75f, .125f},
        k = {.125f, .75f, -.25f, .5f},
        v = {.75f, .25f, -.5f, .125f};
    float[] decay = {-.25f, -.5f}, beta = {.5f, .75f};
    float[] expectedState = {.125f, .25f, .375f, .5f}, actualState = expectedState.clone();
    float[] expected = new float[4], actual = new float[4];
    var provider = RustGgufBatchedMatrixKernel.open(libraryPath());
    try (var library = NativeKernelLibrary.open(libraryPath())) {
      var capabilities = com.integrallis.vectors.core.VectorUtil.runtimeCapabilities();
      // The native recurrence is fused. A host without the same Java arithmetic must decline it.
      boolean matchingFma =
          capabilities.fastScalarFma()
              && (!capabilities.vectorApi() || capabilities.fastVectorFma());
      assertThat(provider.supportsGatedDeltaNet()).isEqualTo(matchingFma);
      assertThat(provider.supportsGroupedGatedDeltaNet()).isEqualTo(matchingFma);
      int[] slots = {1, 0};
      if (matchingFma) {
        library.gatedDeltaNetF32(q, k, v, decay, beta, expectedState, expected, 2, 1, 1, 2, 2);
        provider.gatedDeltaNet(q, k, v, decay, beta, actualState, actual, 2, 1, 1, 2, 2);
        assertThat(actual).containsExactly(expected);
        assertThat(actualState).containsExactly(expectedState);
        float[] gs = {.1f, .2f, .3f, .4f, .5f, .6f, .7f, .8f};
        float[] ps = gs.clone();
        library.groupedGatedDeltaNetF32(
            q, k, v, decay, beta, gs, expected, slots, 2, 2, 1, 1, 2, 2);
        provider.groupedGatedDeltaNet(q, k, v, decay, beta, ps, actual, slots, 2, 2, 1, 1, 2, 2);
        assertThat(actual).containsExactly(expected);
        assertThat(ps).containsExactly(gs);
      } else {
        assertThat(provider.supportsGatedDeltaNet(2, 2)).isFalse();
      }
      assertThat(provider.supportsGroupedAttention()).isTrue();
      float[] scoresA = new float[2], scoresB = new float[2], empty = new float[0];
      library.groupedAttentionF32(
          q, 0, k, 0, v, 0, 2, empty, 0, empty, 0, 0, expected, 0, scoresA, 2, 2, 2, 2, 1, 1, .5f);
      provider.groupedAttention(
          q, 0, k, 0, v, 0, 2, empty, 0, empty, 0, 0, actual, 0, scoresB, 2, 2, 2, 2, 1, 1, .5f);
      assertThat(actual).containsExactly(expected);
      assertThat(scoresB).containsExactly(scoresA);
      assertThat(q).containsExactly(.25f, -.5f, .75f, .125f);
      assertThat(slots).containsExactly(1, 0);
    }
    provider.close();
    assertThatThrownBy(
            () -> provider.gatedDeltaNet(q, k, v, decay, beta, actualState, actual, 2, 1, 1, 2, 2))
        .isInstanceOf(IllegalStateException.class);
    assertThatThrownBy(
            () ->
                provider.groupedGatedDeltaNet(
                    q, k, v, decay, beta, new float[8], actual, new int[] {1, 0}, 2, 2, 1, 1, 2, 2))
        .isInstanceOf(IllegalStateException.class);
  }

  private static Path libraryPath() {
    return Path.of(System.getProperty("models.native.kernels.library"));
  }
}
