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

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/** Production gate combining grounded RAG SLOs with same-host local-engine comparisons. */
public final class RagProductionQualificationPolicy {
  public static final String POLICY_ID = "production-rag-model-contribution-v3";
  public static final double MINIMUM_MODEL_ANSWER_RATE = 1.0 / 3.0;
  public static final double MINIMUM_MODEL_ANSWER_CORRECT_RATE = 0.90;

  private static final List<String> MATCHED_GENERATION_CONTROLS =
      List.of("temperature", "topK", "topP", "seed", "repetitionPenalty");
  private static final Map<String, RelativeThreshold> THRESHOLDS =
      Map.of(
          "llama.cpp", new RelativeThreshold(0.45, 2.0),
          "ollama", new RelativeThreshold(0.80, 1.5));

  private RagProductionQualificationPolicy() {}

  public static RagProductionQualification assess(
      RagBenchmarkReport candidate, List<RagBenchmarkReport> baselines) {
    Objects.requireNonNull(candidate, "candidate");
    Objects.requireNonNull(baselines, "baselines");

    ModelContribution contribution = modelContribution(candidate);
    RagPerformanceTier absoluteTier =
        RagPerformancePolicy.classify(candidate.summary().policyMetrics());
    if (absoluteTier != RagPerformanceTier.PRODUCTION_READY
        && absoluteTier != RagPerformanceTier.USABLE) {
      return result(
          candidate,
          absoluteTier,
          RagQualificationVerdict.FAILED_ABSOLUTE_GATE,
          List.of(),
          List.of(),
          Map.of(),
          contribution);
    }
    if (!contribution.qualifies()) {
      return result(
          candidate,
          absoluteTier,
          RagQualificationVerdict.FAILED_MODEL_CONTRIBUTION_GATE,
          List.of(),
          List.of(),
          Map.of(),
          contribution);
    }

    List<RagComparatorAssessment> comparisons = new ArrayList<>();
    Map<String, String> exclusions = new LinkedHashMap<>();
    for (RagBenchmarkReport baseline : baselines) {
      Objects.requireNonNull(baseline, "baseline");
      RelativeThreshold threshold = THRESHOLDS.get(baseline.backend());
      if (threshold == null) {
        exclusions.put(baseline.backend(), "unsupported comparator backend");
        continue;
      }
      String exclusion = exclusion(candidate, baseline);
      if (exclusion != null) {
        exclusions.put(baseline.backend(), exclusion);
        continue;
      }
      comparisons.add(compare(candidate, baseline, threshold));
    }

    if (comparisons.isEmpty()) {
      return result(
          candidate,
          absoluteTier,
          RagQualificationVerdict.NO_COMPARABLE_BASELINE,
          List.of(),
          comparisons,
          exclusions,
          contribution);
    }

    List<String> qualifyingComparators =
        comparisons.stream()
            .filter(RagComparatorAssessment::qualified)
            .map(RagComparatorAssessment::comparatorBackend)
            .toList();
    RagQualificationVerdict verdict =
        !qualifyingComparators.contains("ollama")
            ? RagQualificationVerdict.FAILED_RELATIVE_GATE
            : RagQualificationVerdict.QUALIFIED;
    return result(
        candidate,
        absoluteTier,
        verdict,
        qualifyingComparators,
        comparisons,
        exclusions,
        contribution);
  }

  private static RagProductionQualification result(
      RagBenchmarkReport candidate,
      RagPerformanceTier absoluteTier,
      RagQualificationVerdict verdict,
      List<String> qualifyingComparators,
      List<RagComparatorAssessment> comparisons,
      Map<String, String> exclusions,
      ModelContribution contribution) {
    return new RagProductionQualification(
        candidate.modelId(),
        candidate.backend(),
        candidate.artifactSha256(),
        absoluteTier,
        verdict,
        verdict == RagQualificationVerdict.QUALIFIED,
        contribution.modelAnswerCount(),
        contribution.modelAnswerRate(),
        contribution.modelAnswerCorrectRate(),
        MINIMUM_MODEL_ANSWER_RATE,
        MINIMUM_MODEL_ANSWER_CORRECT_RATE,
        qualifyingComparators,
        comparisons,
        exclusions);
  }

  private static ModelContribution modelContribution(RagBenchmarkReport report) {
    int totalAttempts = report.summary().totalAttempts();
    int successfulAttempts = report.summary().successfulAttempts();
    int modelAnswerCount =
        Math.toIntExact(
            report.runs().stream()
                .filter(run -> run.grounding().decision().modelContributed())
                .count());
    int correctModelAnswerCount =
        Math.toIntExact(
            report.runs().stream()
                .filter(run -> run.grounding().decision().modelContributed())
                .filter(run -> run.evaluation().correct())
                .count());
    double modelAnswerRate = totalAttempts == 0 ? 0 : (double) modelAnswerCount / totalAttempts;
    double modelAnswerCorrectRate =
        modelAnswerCount == 0 ? 0 : (double) correctModelAnswerCount / modelAnswerCount;
    boolean completeRunEvidence = report.runs().size() == successfulAttempts;
    boolean qualifies =
        completeRunEvidence
            && modelAnswerRate >= MINIMUM_MODEL_ANSWER_RATE
            && modelAnswerCorrectRate >= MINIMUM_MODEL_ANSWER_CORRECT_RATE;
    return new ModelContribution(
        modelAnswerCount, modelAnswerRate, modelAnswerCorrectRate, qualifies);
  }

  private static String exclusion(RagBenchmarkReport candidate, RagBenchmarkReport baseline) {
    if (!Objects.equals(candidate.modelId(), baseline.modelId())) {
      return "model ID differs";
    }
    if (candidate.artifactSha256() == null
        || candidate.artifactSha256().isBlank()
        || !candidate.artifactSha256().equals(baseline.artifactSha256())) {
      return "artifact SHA-256 differs";
    }
    if (candidate.artifactSizeBytes() != baseline.artifactSizeBytes()) {
      return "artifact size differs";
    }
    if (!sameHardware(candidate.environment(), baseline.environment())) {
      return "host hardware differs";
    }
    if (!sameWorkload(candidate.settings(), baseline.settings())) {
      return "benchmark workload differs";
    }
    RagPerformanceTier baselineTier =
        RagPerformancePolicy.classify(baseline.summary().policyMetrics());
    if (baselineTier == RagPerformanceTier.FAILED_RUNTIME
        || baselineTier == RagPerformanceTier.FAILED_QUALITY) {
      return "baseline failed runtime or quality gate";
    }
    return null;
  }

  private static boolean sameWorkload(RagBenchmarkSettings left, RagBenchmarkSettings right) {
    return left.workload().equals(right.workload())
        && left.corpusSha256().equals(right.corpusSha256())
        && left.caseIds().equals(right.caseIds())
        && left.promptTemplate().equals(right.promptTemplate())
        && left.retrievalTopK() == right.retrievalTopK()
        && left.maxOutputTokens() == right.maxOutputTokens()
        && left.warmups() == right.warmups()
        && left.iterations() == right.iterations()
        && left.contextLength() == right.contextLength()
        && left.threads() == right.threads()
        && left.groundingPolicy().equals(right.groundingPolicy())
        && Float.compare(left.minimumRetrievalScore(), right.minimumRetrievalScore()) == 0
        && MATCHED_GENERATION_CONTROLS.stream()
            .allMatch(
                control ->
                    left.generationControls().containsKey(control)
                        && Objects.equals(
                            left.generationControls().get(control),
                            right.generationControls().get(control)));
  }

  private static boolean sameHardware(RagBenchmarkEnvironment left, RagBenchmarkEnvironment right) {
    return left.hostname().equals(right.hostname())
        && left.osName().equals(right.osName())
        && left.architecture().equals(right.architecture())
        && left.cpuModel().equals(right.cpuModel())
        && left.availableProcessors() == right.availableProcessors()
        && left.totalMemoryBytes() == right.totalMemoryBytes();
  }

  private static RagComparatorAssessment compare(
      RagBenchmarkReport candidate, RagBenchmarkReport baseline, RelativeThreshold threshold) {
    double decodeRatio =
        ratio(
            candidate.summary().p50DecodeTokensPerSecond(),
            baseline.summary().p50DecodeTokensPerSecond());
    double endToEndRatio =
        ratio(
            candidate.summary().endToEndMillis().p95(), baseline.summary().endToEndMillis().p95());

    List<String> failures = new ArrayList<>();
    if (decodeRatio < threshold.minimumDecodeRatio()) {
      failures.add("decode throughput below relative floor");
    }
    if (endToEndRatio > threshold.maximumEndToEndRatio()) {
      failures.add("end-to-end latency above relative ceiling");
    }
    return new RagComparatorAssessment(
        baseline.backend(),
        decodeRatio,
        endToEndRatio,
        threshold.minimumDecodeRatio(),
        threshold.maximumEndToEndRatio(),
        failures.isEmpty(),
        failures);
  }

  private static double ratio(double numerator, double denominator) {
    if (!Double.isFinite(numerator)
        || !Double.isFinite(denominator)
        || numerator < 0
        || denominator <= 0) {
      return Double.POSITIVE_INFINITY;
    }
    return numerator / denominator;
  }

  private record ModelContribution(
      int modelAnswerCount,
      double modelAnswerRate,
      double modelAnswerCorrectRate,
      boolean qualifies) {}

  private record RelativeThreshold(double minimumDecodeRatio, double maximumEndToEndRatio) {}
}
