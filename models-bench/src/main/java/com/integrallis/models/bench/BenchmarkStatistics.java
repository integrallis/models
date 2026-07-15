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

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.function.ToDoubleFunction;

/** Deterministic aggregation utilities for benchmark measurements. */
public final class BenchmarkStatistics {

  private BenchmarkStatistics() {}

  public static PerformanceSummary summarize(double loadMillis, List<TrialMeasurement> trials) {
    List<TrialMeasurement> successes =
        trials.stream().filter(TrialMeasurement::successful).toList();
    if (successes.isEmpty()) {
      return new PerformanceSummary(loadMillis, 0, 0, 0, 0, 0, 0, 0, 0, trials.size());
    }

    return new PerformanceSummary(
        loadMillis,
        percentile(values(successes, TrialMeasurement::ttftMillis), 0.50),
        percentile(values(successes, TrialMeasurement::ttftMillis), 0.95),
        percentile(values(successes, TrialMeasurement::tpotMillis), 0.50),
        percentile(values(successes, TrialMeasurement::tpotMillis), 0.95),
        percentile(values(successes, TrialMeasurement::prefillTokensPerSecond), 0.50),
        percentile(values(successes, TrialMeasurement::decodeTokensPerSecond), 0.50),
        successes.stream().mapToLong(TrialMeasurement::peakRssBytes).max().orElse(0),
        successes.size(),
        trials.size());
  }

  static double percentile(List<Double> values, double percentile) {
    if (values.isEmpty()) {
      throw new IllegalArgumentException("values must not be empty");
    }
    if (!(percentile > 0 && percentile <= 1)) {
      throw new IllegalArgumentException("percentile must be in (0, 1]");
    }
    List<Double> sorted = new ArrayList<>(values);
    sorted.sort(Comparator.naturalOrder());
    int index = Math.max(0, (int) Math.ceil(percentile * sorted.size()) - 1);
    return sorted.get(index);
  }

  private static List<Double> values(
      List<TrialMeasurement> measurements, ToDoubleFunction<TrialMeasurement> extractor) {
    return measurements.stream().mapToDouble(extractor).boxed().toList();
  }
}
