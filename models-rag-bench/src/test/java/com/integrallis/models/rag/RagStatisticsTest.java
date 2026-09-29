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

  /**
   * A model that contributes nothing still reports a perfect {@code correctAnswerRate}.
   *
   * <p>This is the 2026-09-29 misreading as a test. Fourteen models were reported as scoring 1.000
   * across every quality metric; 69% of all 378 attempts had actually been answered by {@code
   * EXTRACTIVE_FALLBACK}, six models had a raw correct rate of zero, and only five passed the
   * qualification gate. Nothing in the summary contradicted the claim, because the numbers that
   * would have were only in {@code runs[]}.
   *
   * <p>So the assertion that matters is the pairing: {@code correctAnswerRate} of 1.000 sitting
   * beside a {@code modelAnswerRate} of 1/9 and an {@code extractiveFallbackRate} of 8/9. Any
   * consumer reading the summary now sees both, and the last assertion states the consequence --
   * this is below the qualification gate's model-answer floor, so a report like this is not a
   * qualification.
   */
  @Test
  void aSummaryShowsWhenTheHarnessAnsweredRatherThanTheModel() {
    RagCase answerable = new RagCase("known", "question", List.of("source"), List.of("fact"), true);
    RagDocument source = new RagDocument("source", "Source", "fact");
    RetrievedDocument hit = new RetrievedDocument(source, 1, 1);
    RagEvaluation correct = new RagEvaluation(1, 1, 1, 1, 1, false, true);
    RagEvaluation wrong = new RagEvaluation(1, 1, 0, 0, 0, false, false);

    List<RagRun> runs = new java.util.ArrayList<>();
    // Eight attempts where the model's own text failed screening and the harness substituted an
    // extractive answer: raw wrong, grounded right.
    for (int i = 0; i < 8; i++) {
      runs.add(substituted(answerable, hit, GroundingDecision.EXTRACTIVE_FALLBACK, wrong, correct));
    }
    // One attempt the model actually answered, correctly.
    runs.add(substituted(answerable, hit, GroundingDecision.MODEL_ANSWER, correct, correct));

    RagBenchmarkSummary summary = RagStatistics.summarize(runs, 9, Map.of("known", answerable));

    assertThat(summary.correctAnswerRate())
        .describedAs("the pipeline answered every attempt correctly")
        .isEqualTo(1.0);
    assertThat(summary.rawCorrectAnswerRate())
        .describedAs("the model's own text was right once in nine")
        .isEqualTo(1.0 / 9.0);
    assertThat(summary.modelAnswerRate()).isEqualTo(1.0 / 9.0);
    assertThat(summary.modelAnswerCorrectRate())
        .describedAs("of the answers the model did contribute, all were correct")
        .isEqualTo(1.0);
    assertThat(summary.extractiveFallbackRate()).isEqualTo(8.0 / 9.0);

    assertThat(summary.modelAnswerRate())
        .describedAs(
            "below RagProductionQualificationPolicy's model-answer floor, so a report like this"
                + " is not a qualification however good correctAnswerRate looks")
        .isLessThan(RagProductionQualificationPolicy.MINIMUM_MODEL_ANSWER_RATE);
  }

  private static RagRun substituted(
      RagCase testCase,
      RetrievedDocument hit,
      GroundingDecision decision,
      RagEvaluation rawEvaluation,
      RagEvaluation evaluation) {
    return new RagRun(
        "plain-java",
        "test",
        "model",
        testCase.id(),
        List.of(hit),
        "hash",
        10,
        5,
        1_250,
        new GenerationResult("###", 100, 20, 10, 11, 200, 1_200, 500, 20, 1_000, 30, 0.001),
        new GroundedAnswer("###", "fact [source]", decision),
        rawEvaluation,
        evaluation);
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
