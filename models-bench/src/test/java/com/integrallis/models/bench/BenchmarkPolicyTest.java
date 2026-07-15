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
package com.integrallis.models.bench;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Test;

class BenchmarkPolicyTest {

  @Test
  void classifiesMlperfAnchoredUsabilityTiers() {
    assertThat(BenchmarkPolicy.classify(summary(450, 35))).isEqualTo(PerformanceTier.INTERACTIVE);
    assertThat(BenchmarkPolicy.classify(summary(900, 12))).isEqualTo(PerformanceTier.RESPONSIVE);
    assertThat(BenchmarkPolicy.classify(summary(1_900, 5))).isEqualTo(PerformanceTier.USABLE);
    assertThat(BenchmarkPolicy.classify(summary(2_100, 4.9))).isEqualTo(PerformanceTier.OFFLINE);
  }

  @Test
  void requiresEveryMeasuredTrialToComplete() {
    PerformanceSummary incomplete =
        new PerformanceSummary(100, 300, 450, 20, 25, 40, 35, 2_000_000_000L, 9, 10);

    assertThat(BenchmarkPolicy.classify(incomplete)).isEqualTo(PerformanceTier.FAILED);
  }

  @Test
  void requiresTenMeasuredTrialsForAUsabilityClaim() {
    PerformanceSummary undersampled =
        new PerformanceSummary(100, 300, 450, 20, 25, 40, 35, 2_000_000_000L, 9, 9);

    assertThat(BenchmarkPolicy.classify(undersampled)).isEqualTo(PerformanceTier.FAILED);
  }

  @Test
  void labelsBackendRatiosAsProjectPolicy() {
    assertThat(BenchmarkPolicy.relativeTier(0.82)).isEqualTo(RelativePerformance.COMPETITIVE);
    assertThat(BenchmarkPolicy.relativeTier(0.50)).isEqualTo(RelativePerformance.VIABLE);
    assertThat(BenchmarkPolicy.relativeTier(0.49))
        .isEqualTo(RelativePerformance.NEEDS_OPTIMIZATION);
  }

  private static PerformanceSummary summary(double p95TtftMillis, double p95DecodeTokensPerSecond) {
    return new PerformanceSummary(
        100,
        p95TtftMillis * 0.8,
        p95TtftMillis,
        1_000.0 / (p95DecodeTokensPerSecond * 1.2),
        1_000.0 / p95DecodeTokensPerSecond,
        50,
        p95DecodeTokensPerSecond,
        2_000_000_000L,
        10,
        10);
  }
}
