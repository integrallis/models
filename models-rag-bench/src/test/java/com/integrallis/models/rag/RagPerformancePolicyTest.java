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
package com.integrallis.models.rag;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class RagPerformancePolicyTest {

  @Test
  void productionReadyRequiresLatencyRetrievalAndGroundingGates() {
    RagPerformanceSummary passing =
        new RagPerformanceSummary(20, 20, 20, 80, 700, 75, 3_800, 1.0, 1.0, 0.95, 0.95, 1.0);
    RagPerformanceSummary slow =
        new RagPerformanceSummary(20, 20, 20, 80, 1_200, 75, 4_500, 1.0, 1.0, 0.95, 0.95, 1.0);

    assertThat(RagPerformancePolicy.classify(passing))
        .isEqualTo(RagPerformanceTier.PRODUCTION_READY);
    assertThat(RagPerformancePolicy.classify(slow)).isEqualTo(RagPerformanceTier.USABLE);
  }

  @Test
  void aFastUngroundedRunFailsTheProductionGate() {
    RagPerformanceSummary ungrounded =
        new RagPerformanceSummary(20, 20, 20, 50, 500, 50, 2_000, 1.0, 1.0, 0.70, 0.70, 0.0);

    assertThat(RagPerformancePolicy.classify(ungrounded))
        .isEqualTo(RagPerformanceTier.FAILED_QUALITY);
  }
}
