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
    double correctAnswerRate,
    double rawCorrectAnswerRate,
    double modelAnswerRate,
    double modelAnswerCorrectRate,
    double extractiveFallbackRate) {

  /**
   * The share of attempts whose answer came from the model, and how often that answer was right.
   *
   * <p>Published beside {@code correctAnswerRate} because that rate measures the <b>pipeline</b>,
   * not the model: when a generated answer fails citation screening, the grounding policy replaces
   * it with text extracted from the retrieved document and the replacement is what gets scored. A
   * model that emits nothing usable can therefore report {@code correctAnswerRate} of 1.000.
   *
   * <p>That is not hypothetical. On 2026-09-29, fourteen models were reported as scoring 1.000
   * across every quality metric when 69% of all 378 attempts had in fact been answered by {@code
   * EXTRACTIVE_FALLBACK}, six of the fourteen had a {@code rawCorrectAnswerRate} of zero, and only
   * five passed {@link RagProductionQualificationPolicy}'s model-contribution gate. The numbers
   * needed to see that were only in {@code runs[]}, so every consumer that read the summary -- a
   * report, a release note, a person -- read the pipeline's score as the model's.
   *
   * <p>{@code rawCorrectAnswerRate} scores the model's own text before grounding; {@code
   * modelAnswerRate} and {@code modelAnswerCorrectRate} are the two quantities the qualification
   * gate actually thresholds; {@code extractiveFallbackRate} is how often the harness answered
   * instead. Anything consuming a summary now gets all of them without walking the runs.
   */
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
