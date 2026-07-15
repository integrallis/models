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
import static org.assertj.core.api.Assertions.within;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class BenchmarkStatisticsTest {

  @Test
  void usesNearestRankPercentilesForTailLatency() {
    List<Double> values = new ArrayList<>();
    for (int value = 1; value <= 20; value++) {
      values.add((double) value);
    }

    assertThat(BenchmarkStatistics.percentile(values, 0.50)).isEqualTo(10);
    assertThat(BenchmarkStatistics.percentile(values, 0.95)).isEqualTo(19);
  }

  @Test
  void summarizesOnlySuccessfulTrialsAndRetainsFailureCount() {
    List<TrialMeasurement> trials =
        List.of(
            TrialMeasurement.success(100, 400, 10, 40, 8, 1_000, 20, "a"),
            TrialMeasurement.success(200, 500, 8, 40, 8, 1_200, 25, "b"),
            TrialMeasurement.failure("backend failed"));

    PerformanceSummary summary = BenchmarkStatistics.summarize(75, trials);

    assertThat(summary.loadMillis()).isEqualTo(75);
    assertThat(summary.p50TtftMillis()).isEqualTo(100);
    assertThat(summary.p95TtftMillis()).isEqualTo(200);
    assertThat(summary.successfulTrials()).isEqualTo(2);
    assertThat(summary.totalTrials()).isEqualTo(3);
    assertThat(summary.peakRssBytes()).isEqualTo(1_200);
  }

  @Test
  void derivesDecodeThroughputFromClientObservedPostFirstTokenLatency() {
    TrialMeasurement measurement =
        TrialMeasurement.success(100, 500, 20, 40, 8, 1_000, 20, "output-sha");

    assertThat(measurement.decodeTokensPerSecond()).isEqualTo(17.5);
    assertThat(measurement.tpotMillis()).isCloseTo(400.0 / 7, within(0.000_001));
  }
}
