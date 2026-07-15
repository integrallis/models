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

/** Threshold policy used to classify local inference benchmark reports. */
public final class BenchmarkPolicy {

  private BenchmarkPolicy() {}

  /** Returns the highest latency tier satisfied by the complete benchmark run. */
  public static PerformanceTier classify(PerformanceSummary summary) {
    if (summary.totalTrials() < 10 || summary.successfulTrials() != summary.totalTrials()) {
      return PerformanceTier.FAILED;
    }
    if (meets(summary, 500, 40)) {
      return PerformanceTier.INTERACTIVE;
    }
    if (meets(summary, 1_000, 100)) {
      return PerformanceTier.RESPONSIVE;
    }
    if (meets(summary, 2_000, 200)) {
      return PerformanceTier.USABLE;
    }
    return PerformanceTier.OFFLINE;
  }

  /** Classifies pure-Java decode throughput relative to the same artifact in llama.cpp. */
  public static RelativePerformance relativeTier(double throughputRatio) {
    if (!Double.isFinite(throughputRatio) || throughputRatio < 0) {
      throw new IllegalArgumentException("throughputRatio must be finite and non-negative");
    }
    if (throughputRatio >= 0.80) {
      return RelativePerformance.COMPETITIVE;
    }
    if (throughputRatio >= 0.50) {
      return RelativePerformance.VIABLE;
    }
    return RelativePerformance.NEEDS_OPTIMIZATION;
  }

  private static boolean meets(
      PerformanceSummary summary, double maxTtftMillis, double maxTpotMillis) {
    return summary.p95TtftMillis() <= maxTtftMillis && summary.p95TpotMillis() <= maxTpotMillis;
  }
}
