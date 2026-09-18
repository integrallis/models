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
package com.integrallis.models.backend.tornado;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.plan.PureJavaPlanConfiguration;
import java.lang.foreign.MemorySegment;
import org.junit.jupiter.api.Test;

class TornadoGgufBatchedMatrixKernelTest {

  @Test
  void acceleratesQ4PrefillButLeavesDecodeAndUnsupportedFormatsOnTheJavaPath() {
    try (TornadoGgufBatchedMatrixKernel kernel = new TornadoGgufBatchedMatrixKernel()) {
      assertThat(kernel.executionBatchSize()).isEqualTo(32);
      assertThat(kernel.supports(GgufTensorType.Q4_0)).isTrue();
      assertThat(kernel.supports(GgufTensorType.Q4_K)).isTrue();
      assertThat(kernel.supports(GgufTensorType.Q6_K)).isTrue();
      assertThat(kernel.supports(GgufTensorType.Q5_K)).isFalse();
      assertThat(kernel.supports(GgufTensorType.Q8_0)).isFalse();
      assertThat(kernel.supports(GgufTensorType.F32)).isFalse();
      assertThat(kernel.isEligible(GgufTensorType.Q4_0, 1, 3072, 1024)).isFalse();
      assertThat(kernel.isEligible(GgufTensorType.Q4_0, 4, 3072, 1024)).isTrue();
      assertThat(kernel.isEligible(GgufTensorType.Q4_0, 30, 3072, 1024)).isTrue();
      assertThat(kernel.isEligible(GgufTensorType.Q4_0, 32, 3072, 1024)).isTrue();
      assertThat(kernel.isEligible(GgufTensorType.Q4_0, 33, 3072, 1024)).isFalse();
      assertThat(kernel.isEligible(GgufTensorType.Q4_0, 4, 32, 32)).isFalse();
      assertThat(kernel.supportsDual(GgufTensorType.Q4_0, GgufTensorType.Q4_0)).isTrue();
      assertThat(
              kernel.supportsTriple(GgufTensorType.Q4_0, GgufTensorType.Q4_0, GgufTensorType.Q4_0))
          .isTrue();
    }
  }

  @Test
  void acceptsAnExplicitFixedExecutionBatch() {
    try (TornadoGgufBatchedMatrixKernel kernel = new TornadoGgufBatchedMatrixKernel(64)) {
      assertThat(kernel.executionBatchSize()).isEqualTo(64);
      assertThat(kernel.isEligible(GgufTensorType.Q4_0, 64, 3072, 1024)).isTrue();
      assertThat(kernel.isEligible(GgufTensorType.Q4_0, 65, 3072, 1024)).isFalse();
    }
  }

  @Test
  void decodeAccelerationIsExplicitAndUsesItsOwnSingleTokenShape() {
    try (TornadoGgufBatchedMatrixKernel kernel = new TornadoGgufBatchedMatrixKernel(32, true)) {
      assertThat(kernel.acceleratesDecode()).isTrue();
      assertThat(kernel.executionBatchSizeFor(1)).isEqualTo(1);
      assertThat(kernel.executionBatchSizeFor(4)).isEqualTo(32);
      assertThat(kernel.isEligible(GgufTensorType.Q4_0, 1, 3072, 1024)).isTrue();
      assertThat(
              kernel.isDualEligible(GgufTensorType.Q4_0, 1024, GgufTensorType.Q4_0, 1024, 1, 1024))
          .isTrue();
    }
  }

  @Test
  void rejectsAnExecutionBatchBelowTheGpuThreshold() {
    assertThatThrownBy(() -> new TornadoGgufBatchedMatrixKernel(3))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("executionBatchSize");
  }

  @Test
  void padsTheLastPromptChunkWithoutReusingStaleActivations() {
    float[] padded = {9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f, 9.0f};

    float[] executionInput =
        TornadoGgufBatchedMatrixKernel.prepareExecutionInput(
            new float[] {1.0f, 2.0f, 3.0f, 4.0f}, padded, 2, 4, 2);

    assertThat(executionInput).isSameAs(padded);
    assertThat(executionInput).containsExactly(1.0f, 2.0f, 3.0f, 4.0f, 0.0f, 0.0f, 0.0f, 0.0f);
  }

  @Test
  void recommendsTheGroupedProjectionPlanThatTheExperimentImplements() {
    try (TornadoGgufBatchedMatrixKernel kernel = new TornadoGgufBatchedMatrixKernel()) {
      assertThat(kernel.planRecommendations())
          .containsEntry(PureJavaPlanConfiguration.GROUPED_PROJECTIONS_PROPERTY, "true")
          .containsEntry(PureJavaPlanConfiguration.STAGED_QUANTIZED_FFN_PROPERTY, "false")
          .containsEntry(PureJavaPlanConfiguration.STAGED_QUANTIZED_LAYER_PROPERTY, "false");
    }
  }

  @Test
  void admitsTheMixedKProjectionGroupThatQ4KMModelsPresent() {
    try (TornadoGgufBatchedMatrixKernel kernel = new TornadoGgufBatchedMatrixKernel()) {
      assertThat(kernel.planRecommendations())
          .containsEntry(PureJavaPlanConfiguration.MIXED_K_PROJECTIONS_PROPERTY, "true");
      assertThat(
              kernel.supportsTriple(GgufTensorType.Q4_K, GgufTensorType.Q4_K, GgufTensorType.Q6_K))
          .isTrue();
      assertThat(
              kernel.isTripleEligible(
                  GgufTensorType.Q4_K,
                  2048,
                  GgufTensorType.Q4_K,
                  512,
                  GgufTensorType.Q6_K,
                  512,
                  8,
                  2048))
          .isTrue();
    }
  }

  @Test
  void keepsTheTwoActivationFamiliesOutOfOneGroupedDispatch() {
    try (TornadoGgufBatchedMatrixKernel kernel = new TornadoGgufBatchedMatrixKernel()) {
      // Q4_0 needs Q8_0 activations and the K-quants need Q8_K, so one prepared activation can
      // never serve both. Mixed groups must fall back rather than silently use the wrong scales.
      assertThat(kernel.supportsDual(GgufTensorType.Q4_0, GgufTensorType.Q4_K)).isFalse();
      assertThat(kernel.supportsDual(GgufTensorType.Q4_K, GgufTensorType.Q4_0)).isFalse();
      assertThat(
              kernel.supportsTriple(GgufTensorType.Q4_K, GgufTensorType.Q4_0, GgufTensorType.Q4_K))
          .isFalse();
      // Q6_K has no dual kernel and is only ever the third matrix of a grouped dispatch.
      assertThat(kernel.supportsDual(GgufTensorType.Q6_K, GgufTensorType.Q6_K)).isFalse();
      assertThat(
              kernel.supportsTriple(GgufTensorType.Q6_K, GgufTensorType.Q4_K, GgufTensorType.Q4_K))
          .isFalse();
      assertThat(kernel.supportsDual(GgufTensorType.Q4_K, GgufTensorType.Q4_K)).isTrue();
      assertThat(kernel.supportsDual(GgufTensorType.Q5_K, GgufTensorType.Q5_K)).isFalse();
    }
  }

  @Test
  void requiresWholeSuperBlocksForKQuantProjections() {
    try (TornadoGgufBatchedMatrixKernel kernel = new TornadoGgufBatchedMatrixKernel()) {
      assertThat(kernel.isEligible(GgufTensorType.Q4_K, 8, 4096, 1024)).isTrue();
      assertThat(kernel.isEligible(GgufTensorType.Q6_K, 8, 4096, 1024)).isTrue();
      // 1120 is a multiple of 32 but not of 256: legal for Q4_0, never for a K-quant.
      assertThat(kernel.isEligible(GgufTensorType.Q4_0, 8, 4096, 1120)).isTrue();
      assertThat(kernel.isEligible(GgufTensorType.Q4_K, 8, 4096, 1120)).isFalse();
      assertThat(
              kernel.isDualEligible(GgufTensorType.Q4_K, 4096, GgufTensorType.Q4_K, 4096, 8, 1120))
          .isFalse();
    }
  }

  @Test
  void refusesTensorsTooLargeForAnIntIndexedDeviceBuffer() {
    try (TornadoGgufBatchedMatrixKernel kernel = new TornadoGgufBatchedMatrixKernel()) {
      // Q4_K stores 144 bytes per 256 values, so a tensor needs about 3.8e9 values before it
      // stops fitting an int-indexed device buffer. A 27B-class vocabulary projection
      // (262144 x 5120 = 755 MiB of Q4_K) is comfortably inside that; a tensor four times
      // taller is not, and must stay on the Vector API rather than fail the load.
      assertThat(kernel.isEligible(GgufTensorType.Q4_K, 8, 262_144, 5120)).isTrue();
      assertThat(kernel.isEligible(GgufTensorType.Q4_K, 8, 1_000_000, 5120)).isFalse();
      assertThat(kernel.isEligible(GgufTensorType.Q6_K, 8, 1_000_000, 5120)).isFalse();
      assertThat(kernel.isEligible(GgufTensorType.Q4_0, 8, 4_000_000, 5120)).isFalse();
      assertThat(
              kernel.isTripleEligible(
                  GgufTensorType.Q4_K,
                  1_000_000,
                  GgufTensorType.Q4_K,
                  512,
                  GgufTensorType.Q6_K,
                  512,
                  8,
                  5120))
          .isFalse();
    }
  }

  @Test
  void reportsNoRoutedProjectionsBeforeAnyDispatch() {
    try (TornadoGgufBatchedMatrixKernel kernel = new TornadoGgufBatchedMatrixKernel()) {
      assertThat(kernel.routedProjectionsByFormat()).isEmpty();
      assertThat(kernel.projectionPlanCount()).isZero();
      assertThat(kernel.calls()).isZero();
      assertThat(kernel.totalMillis()).isZero();
    }
  }

  @Test
  void refusesToDispatchProjectionsItDeclaredIneligible() {
    try (TornadoGgufBatchedMatrixKernel kernel = new TornadoGgufBatchedMatrixKernel()) {
      assertThatThrownBy(
              () ->
                  kernel.multiply(
                      new float[8],
                      new float[8],
                      MemorySegment.ofArray(new byte[8]),
                      GgufTensorType.Q5_K,
                      8,
                      1,
                      1))
          .isInstanceOf(UnsupportedOperationException.class)
          .hasMessageContaining("not eligible");
    }
  }
}
