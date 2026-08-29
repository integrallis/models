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
package com.integrallis.models.accelerator;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.backend.purejava.gguf.GgufTensorType;
import com.integrallis.models.backend.purejava.plan.PureJavaPlanConfiguration;
import org.junit.jupiter.api.Test;

class TornadoGgufBatchedMatrixKernelTest {

  @Test
  void acceleratesQ4PrefillButLeavesDecodeAndUnsupportedFormatsOnTheJavaPath() {
    try (TornadoGgufBatchedMatrixKernel kernel = new TornadoGgufBatchedMatrixKernel()) {
      assertThat(kernel.supports(GgufTensorType.Q4_0)).isTrue();
      assertThat(kernel.supports(GgufTensorType.Q4_K)).isFalse();
      assertThat(kernel.isEligible(GgufTensorType.Q4_0, 1, 3072, 1024)).isFalse();
      assertThat(kernel.isEligible(GgufTensorType.Q4_0, 4, 3072, 1024)).isTrue();
      assertThat(kernel.isEligible(GgufTensorType.Q4_0, 4, 32, 32)).isFalse();
    }
  }

  @Test
  void recommendsTheSingleProjectionPlanThatTheExperimentImplements() {
    try (TornadoGgufBatchedMatrixKernel kernel = new TornadoGgufBatchedMatrixKernel()) {
      assertThat(kernel.planRecommendations())
          .containsEntry(PureJavaPlanConfiguration.GROUPED_PROJECTIONS_PROPERTY, "false")
          .containsEntry(PureJavaPlanConfiguration.STAGED_QUANTIZED_FFN_PROPERTY, "false")
          .containsEntry(PureJavaPlanConfiguration.STAGED_QUANTIZED_LAYER_PROPERTY, "false");
    }
  }
}
