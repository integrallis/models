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

/** Complete aggregate of latency, throughput, resources, and deterministic quality. */
public record RagBenchmarkSummary(
    int totalAttempts,
    int successfulAttempts,
    double loadMillis,
    LatencyPercentiles retrievalMillis,
    LatencyPercentiles frameworkOverheadMillis,
    LatencyPercentiles ttftMillis,
    LatencyPercentiles tpotMillis,
    LatencyPercentiles endToEndMillis,
    double p50PrefillTokensPerSecond,
    double p50DecodeTokensPerSecond,
    long peakRssBytes,
    double totalCpuMillis,
    long totalInputTokens,
    long totalCacheReadInputTokens,
    long totalCacheWriteInputTokens,
    long totalOutputTokens,
    Double totalEstimatedApiCostUsd,
    Double projectedApiCostPer1kRequestsUsd,
    double retrievalRecall,
    double meanReciprocalRank,
    double factCoverage,
    double citationRecall,
    double citationPrecision,
    double abstentionAccuracy,
    double correctAnswerRate) {

  public RagPerformanceSummary policyMetrics() {
    return new RagPerformanceSummary(
        totalAttempts,
        successfulAttempts,
        totalAttempts,
        retrievalMillis.p95(),
        ttftMillis.p95(),
        tpotMillis.p95(),
        endToEndMillis.p95(),
        retrievalRecall,
        meanReciprocalRank,
        factCoverage,
        citationRecall,
        abstentionAccuracy,
        citationPrecision,
        correctAnswerRate);
  }
}
