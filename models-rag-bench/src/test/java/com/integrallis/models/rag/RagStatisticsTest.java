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

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class RagStatisticsTest {

  @Test
  void summarizesLatencyThroughputResourcesQualityAndFailures() {
    RagCase answerable = new RagCase("known", "question", List.of("source"), List.of("fact"), true);
    RagCase unknown = new RagCase("unknown", "question", List.of(), List.of(), false);
    RagDocument source = new RagDocument("source", "Source", "fact");
    RetrievedDocument hit = new RetrievedDocument(source, 1, 1);
    RagRun knownRun =
        run(
            answerable,
            List.of(hit),
            new GenerationResult(
                "fact [source]", 100, 20, 10, 11, 200, 1_200, 500, 20, 1_000, 30, 0.001),
            10,
            1_250,
            new RagEvaluation(1, 1, 1, 1, 1, false, true));
    RagRun unknownRun =
        run(
            unknown,
            List.of(hit),
            new GenerationResult(
                "INSUFFICIENT_CONTEXT", 90, 0, 0, 6, 400, 1_400, 300, 20, 2_000, 40, 0.002),
            30,
            1_500,
            new RagEvaluation(1, 1, 1, 1, 1, true, true));

    RagBenchmarkSummary summary =
        RagStatistics.summarize(
            List.of(knownRun, unknownRun), 3, Map.of("known", answerable, "unknown", unknown));

    assertThat(summary.totalAttempts()).isEqualTo(3);
    assertThat(summary.successfulAttempts()).isEqualTo(2);
    assertThat(summary.retrievalMillis().p50()).isEqualTo(20);
    assertThat(summary.ttftMillis().p50()).isEqualTo(300);
    assertThat(summary.endToEndMillis().p95()).isEqualTo(1_487.5);
    assertThat(summary.p50DecodeTokensPerSecond()).isEqualTo(7.5);
    assertThat(summary.peakRssBytes()).isEqualTo(2_000);
    assertThat(summary.totalCpuMillis()).isEqualTo(70);
    assertThat(summary.totalInputTokens()).isEqualTo(190);
    assertThat(summary.totalCacheReadInputTokens()).isEqualTo(20);
    assertThat(summary.totalCacheWriteInputTokens()).isEqualTo(10);
    assertThat(summary.totalOutputTokens()).isEqualTo(17);
    assertThat(summary.totalEstimatedApiCostUsd()).isEqualTo(0.003);
    assertThat(summary.projectedApiCostPer1kRequestsUsd()).isEqualTo(1.5);
    assertThat(summary.abstentionAccuracy()).isEqualTo(1);
    assertThat(summary.policyMetrics().successfulCases()).isEqualTo(2);
  }

  private static RagRun run(
      RagCase testCase,
      List<RetrievedDocument> retrieved,
      GenerationResult generation,
      double retrievalMillis,
      double endToEndMillis,
      RagEvaluation evaluation) {
    return new RagRun(
        "plain-java",
        "test",
        "model",
        testCase.id(),
        retrieved,
        "hash",
        retrievalMillis,
        5,
        endToEndMillis,
        generation,
        evaluation);
  }
}
