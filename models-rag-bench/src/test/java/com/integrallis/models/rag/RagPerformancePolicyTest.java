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
  void aSoundModelOverTheLatencySloIsPerformanceReducedNotOffline() {
    // Quality gates all pass; only latency is over the USABLE bound. This used to return OFFLINE,
    // which put it in the same tier as a model that could not be loaded at all.
    RagPerformanceSummary slowButSound =
        new RagPerformanceSummary(20, 20, 20, 80, 2_500, 75, 4_500, 1.0, 1.0, 0.95, 0.95, 1.0);

    assertThat(RagPerformancePolicy.classify(slowButSound))
        .isEqualTo(RagPerformanceTier.PERFORMANCE_REDUCED);
    assertThat(RagPerformancePolicy.classify(slowButSound))
        .isNotEqualTo(RagPerformanceTier.OFFLINE);
  }

  @Test
  void theMarginalCaseIsPerformanceReducedRatherThanTheBottomTier() {
    // gemma_3_4b_it as measured on the small/medium campaign: ttft p95 2090 against a 2000 bound --
    // over by 90ms, or 4.5% -- while holding better than 2x margin on retrieval, tpot and
    // end-to-end, with every quality gate at 1.0. Its ttft MEDIAN was 1783 with a 95% confidence
    // interval of [1610, 1824], so only the tail breached, on 3 of 27 requests.
    RagPerformanceSummary marginal =
        new RagPerformanceSummary(20, 20, 20, 4, 2_090, 91, 4_857, 1.0, 1.0, 0.95, 0.95, 1.0);

    assertThat(RagPerformancePolicy.classify(marginal))
        .isEqualTo(RagPerformanceTier.PERFORMANCE_REDUCED);
  }

  @Test
  void qualityFailuresStillOutrankLatencySoTheyAreNotHiddenAsPerformanceReduced() {
    // A model that is both slow AND ungrounded must report the quality failure: PERFORMANCE_REDUCED
    // says "works, just slower", and reporting it for a model producing wrong answers would be a
    // strictly worse lie than the old OFFLINE was.
    RagPerformanceSummary slowAndUngrounded =
        new RagPerformanceSummary(20, 20, 20, 80, 2_500, 75, 4_500, 1.0, 1.0, 0.70, 0.70, 0.0);

    assertThat(RagPerformancePolicy.classify(slowAndUngrounded))
        .isEqualTo(RagPerformanceTier.FAILED_QUALITY);
  }

  @Test
  void aFastUngroundedRunFailsTheProductionGate() {
    RagPerformanceSummary ungrounded =
        new RagPerformanceSummary(20, 20, 20, 50, 500, 50, 2_000, 1.0, 1.0, 0.70, 0.70, 0.0);

    assertThat(RagPerformancePolicy.classify(ungrounded))
        .isEqualTo(RagPerformanceTier.FAILED_QUALITY);
  }
}
