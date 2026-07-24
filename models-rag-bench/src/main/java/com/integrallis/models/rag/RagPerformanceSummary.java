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

/** Aggregate production metrics for one framework/backend/model combination. */
public record RagPerformanceSummary(
    int totalCases,
    int successfulCases,
    int measuredCases,
    double p95RetrievalMillis,
    double p95TtftMillis,
    double p95TpotMillis,
    double p95EndToEndMillis,
    double retrievalRecall,
    double meanReciprocalRank,
    double factCoverage,
    double citationRecall,
    double abstentionAccuracy,
    double citationPrecision,
    double correctAnswerRate) {
  public RagPerformanceSummary(
      int totalCases,
      int successfulCases,
      int measuredCases,
      double p95RetrievalMillis,
      double p95TtftMillis,
      double p95TpotMillis,
      double p95EndToEndMillis,
      double retrievalRecall,
      double meanReciprocalRank,
      double factCoverage,
      double citationRecall,
      double abstentionAccuracy) {
    this(
        totalCases,
        successfulCases,
        measuredCases,
        p95RetrievalMillis,
        p95TtftMillis,
        p95TpotMillis,
        p95EndToEndMillis,
        retrievalRecall,
        meanReciprocalRank,
        factCoverage,
        citationRecall,
        abstentionAccuracy,
        citationRecall,
        Math.min(Math.min(factCoverage, citationRecall), abstentionAccuracy));
  }
}
