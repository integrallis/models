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

import com.integrallis.models.api.BackendDiagnostics;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class RagProductionQualificationPolicyTest {

  @Test
  void qwenQualifiesAgainstBothSameHostComparators() {
    RagBenchmarkReport candidate = report("pure-java", "sha-qwen", 57.82, 1_591.55);
    RagBenchmarkReport llama = report("llama.cpp", "sha-qwen", 108.59, 860.06);
    RagBenchmarkReport ollama = report("ollama", "sha-qwen", 41.65, 1_966.92);

    RagProductionQualification qualification =
        RagProductionQualificationPolicy.assess(candidate, List.of(llama, ollama));

    assertThat(qualification.qualified()).isTrue();
    assertThat(qualification.qualifyingComparators()).containsExactly("llama.cpp", "ollama");
    assertThat(qualification.comparisons())
        .extracting(RagComparatorAssessment::comparatorBackend)
        .containsExactly("llama.cpp", "ollama");
  }

  @Test
  void smolLmQualifiesAgainstOllamaButNotLlamaCpp() {
    RagBenchmarkReport candidate = report("pure-java", "sha-smol", 43.26, 2_392.60);
    RagBenchmarkReport llama = report("llama.cpp", "sha-smol", 103.75, 983.00);
    RagBenchmarkReport ollama = report("ollama", "sha-smol", 40.90, 2_024.30);

    RagProductionQualification qualification =
        RagProductionQualificationPolicy.assess(candidate, List.of(llama, ollama));

    assertThat(qualification.qualified()).isTrue();
    assertThat(qualification.qualifyingComparators()).containsExactly("ollama");
    assertThat(qualification.comparisons().getFirst().qualified()).isFalse();
    assertThat(qualification.comparisons().getFirst().decodeThroughputRatio())
        .isBetween(0.41, 0.42);
  }

  @Test
  void rejectsReportsForDifferentArtifactsOrHosts() {
    RagBenchmarkReport candidate = report("pure-java", "sha-one", 80, 1_000);
    RagBenchmarkReport otherArtifact = report("ollama", "sha-two", 80, 1_000);
    RagBenchmarkReport otherHost =
        withEnvironment(report("llama.cpp", "sha-one", 80, 1_000), environment("other-host"));

    RagProductionQualification qualification =
        RagProductionQualificationPolicy.assess(candidate, List.of(otherArtifact, otherHost));

    assertThat(qualification.qualified()).isFalse();
    assertThat(qualification.verdict()).isEqualTo(RagQualificationVerdict.NO_COMPARABLE_BASELINE);
    assertThat(qualification.exclusions())
        .containsEntry("ollama", "artifact SHA-256 differs")
        .containsEntry("llama.cpp", "host hardware differs");
  }

  @Test
  void absoluteRagFailureCannotBeOverriddenByFastRelativePerformance() {
    RagBenchmarkReport candidate =
        withSummary(
            report("pure-java", "sha", 200, 100),
            summary(200, 100, 0.80, 1.0, RagPerformanceTier.FAILED_QUALITY));
    RagBenchmarkReport ollama = report("ollama", "sha", 100, 500);

    RagProductionQualification qualification =
        RagProductionQualificationPolicy.assess(candidate, List.of(ollama));

    assertThat(qualification.qualified()).isFalse();
    assertThat(qualification.verdict()).isEqualTo(RagQualificationVerdict.FAILED_ABSOLUTE_GATE);
  }

  @Test
  void llamaCppPerformanceCannotSubstituteForTheRequiredOllamaFloor() {
    RagBenchmarkReport candidate = report("rust-ffm", "sha", 79, 1_000);
    RagBenchmarkReport llama = report("llama.cpp", "sha", 100, 900);
    RagBenchmarkReport ollama = report("ollama", "sha", 100, 1_000);

    RagProductionQualification qualification =
        RagProductionQualificationPolicy.assess(candidate, List.of(llama, ollama));

    assertThat(qualification.qualified()).isFalse();
    assertThat(qualification.qualifyingComparators()).containsExactly("llama.cpp");
    assertThat(qualification.verdict()).isEqualTo(RagQualificationVerdict.FAILED_RELATIVE_GATE);
  }

  @Test
  void acceptsEquivalentSamplingWithBackendSpecificPromptCacheMetadata() {
    RagBenchmarkReport candidate =
        withSettings(
            report("rust-ffm", "sha", 100, 1_000),
            settings(
                Map.of(
                    "temperature", "0",
                    "topK", "1",
                    "topP", "1",
                    "seed", "42",
                    "repetitionPenalty", "1",
                    "promptCache", "longest-common-prefix")));
    RagBenchmarkReport ollama =
        withSettings(
            report("ollama", "sha", 100, 1_000),
            settings(
                Map.of(
                    "temperature", "0",
                    "topK", "1",
                    "topP", "1",
                    "seed", "42",
                    "repetitionPenalty", "1",
                    "rawPrompt", "true")));

    RagProductionQualification qualification =
        RagProductionQualificationPolicy.assess(candidate, List.of(ollama));

    assertThat(qualification.qualified()).isTrue();
    assertThat(qualification.exclusions()).isEmpty();
    assertThat(qualification.qualifyingComparators()).containsExactly("ollama");
  }

  @Test
  void rejectsQualityProducedEntirelyByExtractiveFallback() {
    RagBenchmarkReport candidate =
        withGroundingDecisions(report("rust-ffm", "sha", 100, 1_000), 0, 24, 3, 0);
    RagBenchmarkReport ollama = report("ollama", "sha", 100, 1_000);

    RagProductionQualification qualification =
        RagProductionQualificationPolicy.assess(candidate, List.of(ollama));

    assertThat(qualification.qualified()).isFalse();
    assertThat(qualification.verdict())
        .isEqualTo(RagQualificationVerdict.FAILED_MODEL_CONTRIBUTION_GATE);
    assertThat(qualification.modelAnswerRate()).isZero();
  }

  @Test
  void rejectsAcceptedModelAnswersBelowTheCorrectnessFloor() {
    RagBenchmarkReport candidate =
        withGroundingDecisions(report("rust-ffm", "sha", 100, 1_000), 9, 15, 3, 1);
    RagBenchmarkReport ollama = report("ollama", "sha", 100, 1_000);

    RagProductionQualification qualification =
        RagProductionQualificationPolicy.assess(candidate, List.of(ollama));

    assertThat(qualification.qualified()).isFalse();
    assertThat(qualification.verdict())
        .isEqualTo(RagQualificationVerdict.FAILED_MODEL_CONTRIBUTION_GATE);
    assertThat(qualification.modelAnswerRate()).isEqualTo(1.0 / 3.0);
    assertThat(qualification.modelAnswerCorrectRate()).isEqualTo(8.0 / 9.0);
  }

  @Test
  void acceptsOneThirdCorrectModelAnswersWithFallbackGuardrails() {
    RagBenchmarkReport candidate =
        withGroundingDecisions(report("rust-ffm", "sha", 100, 1_000), 9, 15, 3, 0);
    RagBenchmarkReport ollama = report("ollama", "sha", 100, 1_000);

    RagProductionQualification qualification =
        RagProductionQualificationPolicy.assess(candidate, List.of(ollama));

    assertThat(qualification.qualified()).isTrue();
    assertThat(qualification.modelAnswerRate()).isEqualTo(1.0 / 3.0);
    assertThat(qualification.modelAnswerCorrectRate()).isEqualTo(1.0);
  }

  private static RagBenchmarkReport report(
      String backend,
      String artifactSha256,
      double decodeTokensPerSecond,
      double p95EndToEndMillis) {
    return new RagBenchmarkReport(
        RagBenchmarkReport.CURRENT_SCHEMA_VERSION,
        "2026-07-24T00:00:00Z",
        "plain-java",
        backend,
        "test",
        "model",
        "model.gguf",
        artifactSha256,
        1_000,
        settings(),
        environment("qualification-host"),
        BackendDiagnostics.unavailable(backend),
        null,
        summary(
            decodeTokensPerSecond,
            p95EndToEndMillis,
            1.0,
            1.0,
            RagPerformanceTier.PRODUCTION_READY),
        RagPerformanceTier.PRODUCTION_READY,
        runs(backend, 27, 0, 0, 0),
        List.of());
  }

  private static RagBenchmarkReport withGroundingDecisions(
      RagBenchmarkReport report,
      int modelAnswers,
      int extractiveFallbacks,
      int retrievalAbstentions,
      int incorrectModelAnswers) {
    return new RagBenchmarkReport(
        report.schemaVersion(),
        report.generatedAt(),
        report.framework(),
        report.backend(),
        report.backendVersion(),
        report.modelId(),
        report.model(),
        report.artifactSha256(),
        report.artifactSizeBytes(),
        report.settings(),
        report.environment(),
        report.backendDiagnostics(),
        report.hostedApiPricing(),
        report.summary(),
        report.performanceTier(),
        runs(
            report.backend(),
            modelAnswers,
            extractiveFallbacks,
            retrievalAbstentions,
            incorrectModelAnswers),
        report.failures());
  }

  private static List<RagRun> runs(
      String backend,
      int modelAnswers,
      int extractiveFallbacks,
      int retrievalAbstentions,
      int incorrectModelAnswers) {
    ArrayList<RagRun> runs = new ArrayList<>();
    for (int index = 0; index < modelAnswers; index++) {
      runs.add(
          run(
              backend,
              runs.size(),
              GroundingDecision.MODEL_ANSWER,
              index >= incorrectModelAnswers));
    }
    for (int index = 0; index < extractiveFallbacks; index++) {
      runs.add(run(backend, runs.size(), GroundingDecision.EXTRACTIVE_FALLBACK, true));
    }
    for (int index = 0; index < retrievalAbstentions; index++) {
      runs.add(run(backend, runs.size(), GroundingDecision.RETRIEVAL_ABSTENTION, true));
    }
    return List.copyOf(runs);
  }

  private static RagRun run(
      String backend, int index, GroundingDecision decision, boolean correct) {
    GenerationResult generation = new GenerationResult("answer", 100, 2, 100, 120, 200, 0);
    RagEvaluation evaluation = new RagEvaluation(1, 1, 1, 1, 1, false, correct);
    return new RagRun(
        "plain-java",
        backend,
        "model",
        "case-" + index,
        List.of(),
        "prompt-" + index,
        1,
        1,
        125,
        generation,
        new GroundedAnswer("answer", "answer", decision),
        evaluation,
        evaluation);
  }

  private static RagBenchmarkSummary summary(
      double decodeTokensPerSecond,
      double p95EndToEndMillis,
      double correctAnswerRate,
      double abstentionAccuracy,
      RagPerformanceTier ignoredTier) {
    return new RagBenchmarkSummary(
        27,
        27,
        100,
        new LatencyPercentiles(1, 2),
        new LatencyPercentiles(1, 2),
        new LatencyPercentiles(500, 800),
        new LatencyPercentiles(15, 20),
        new LatencyPercentiles(p95EndToEndMillis * 0.8, p95EndToEndMillis),
        200,
        decodeTokensPerSecond,
        1_000,
        10_000,
        1_000,
        0,
        0,
        500,
        null,
        null,
        1.0,
        1.0,
        correctAnswerRate,
        correctAnswerRate,
        correctAnswerRate,
        abstentionAccuracy,
        correctAnswerRate);
  }

  private static RagBenchmarkSettings settings() {
    return settings(
        Map.of(
            "temperature", "0",
            "topK", "1",
            "topP", "1",
            "seed", "42",
            "repetitionPenalty", "1"));
  }

  private static RagBenchmarkSettings settings(Map<String, String> generationControls) {
    return new RagBenchmarkSettings(
        "corpus-sha",
        List.of("case-one", "case-two"),
        "chatml-no-think",
        1,
        64,
        1,
        3,
        2_048,
        8,
        GroundedAnswerPolicy.POLICY_ID,
        2.0f,
        generationControls);
  }

  private static RagBenchmarkEnvironment environment(String hostname) {
    return new RagBenchmarkEnvironment(
        hostname,
        "Linux",
        "6.8",
        "amd64",
        "AMD EPYC-Milan Processor",
        8,
        32_000_000_000L,
        2_000_000_000L,
        "25",
        "GraalVM",
        "OpenJDK 64-Bit Server VM");
  }

  private static RagBenchmarkReport withEnvironment(
      RagBenchmarkReport report, RagBenchmarkEnvironment environment) {
    return new RagBenchmarkReport(
        report.schemaVersion(),
        report.generatedAt(),
        report.framework(),
        report.backend(),
        report.backendVersion(),
        report.modelId(),
        report.model(),
        report.artifactSha256(),
        report.artifactSizeBytes(),
        report.settings(),
        environment,
        report.backendDiagnostics(),
        report.hostedApiPricing(),
        report.summary(),
        report.performanceTier(),
        report.runs(),
        report.failures());
  }

  private static RagBenchmarkReport withSummary(
      RagBenchmarkReport report, RagBenchmarkSummary summary) {
    return new RagBenchmarkReport(
        report.schemaVersion(),
        report.generatedAt(),
        report.framework(),
        report.backend(),
        report.backendVersion(),
        report.modelId(),
        report.model(),
        report.artifactSha256(),
        report.artifactSizeBytes(),
        report.settings(),
        report.environment(),
        report.backendDiagnostics(),
        report.hostedApiPricing(),
        summary,
        RagPerformancePolicy.classify(summary.policyMetrics()),
        report.runs(),
        report.failures());
  }

  private static RagBenchmarkReport withSettings(
      RagBenchmarkReport report, RagBenchmarkSettings settings) {
    return new RagBenchmarkReport(
        report.schemaVersion(),
        report.generatedAt(),
        report.framework(),
        report.backend(),
        report.backendVersion(),
        report.modelId(),
        report.model(),
        report.artifactSha256(),
        report.artifactSizeBytes(),
        settings,
        report.environment(),
        report.backendDiagnostics(),
        report.hostedApiPricing(),
        report.summary(),
        report.performanceTier(),
        report.runs(),
        report.failures());
  }
}
