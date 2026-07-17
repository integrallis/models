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

/** Explicit project SLO and quality gates for local production RAG. */
public final class RagPerformancePolicy {
  private RagPerformancePolicy() {}

  public static RagPerformanceTier classify(RagPerformanceSummary summary) {
    if (summary.totalCases() < 1
        || summary.measuredCases() < 1
        || summary.successfulCases() != summary.totalCases()) {
      return RagPerformanceTier.FAILED_RUNTIME;
    }
    if (summary.retrievalRecall() < 0.95
        || summary.meanReciprocalRank() < 0.90
        || summary.factCoverage() < 0.90
        || summary.citationRecall() < 0.90
        || summary.citationPrecision() < 0.90
        || summary.abstentionAccuracy() < 1.0
        || summary.correctAnswerRate() < 0.90) {
      return RagPerformanceTier.FAILED_QUALITY;
    }
    if (meetsLatency(summary, 100, 1_000, 100, 5_000)) {
      return RagPerformanceTier.PRODUCTION_READY;
    }
    if (meetsLatency(summary, 250, 2_000, 200, 10_000)) {
      return RagPerformanceTier.USABLE;
    }
    return RagPerformanceTier.OFFLINE;
  }

  private static boolean meetsLatency(
      RagPerformanceSummary summary,
      double retrievalMillis,
      double ttftMillis,
      double tpotMillis,
      double endToEndMillis) {
    return summary.p95RetrievalMillis() <= retrievalMillis
        && summary.p95TtftMillis() <= ttftMillis
        && summary.p95TpotMillis() <= tpotMillis
        && summary.p95EndToEndMillis() <= endToEndMillis;
  }
}
