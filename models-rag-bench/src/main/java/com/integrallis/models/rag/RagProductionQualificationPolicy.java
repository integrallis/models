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
  private static final Map<String, RelativeThreshold> THRESHOLDS =
      Map.of(
          "llama.cpp", new RelativeThreshold(0.45, 2.0),
          "ollama", new RelativeThreshold(0.80, 1.5));

  private RagProductionQualificationPolicy() {}

  public static RagProductionQualification assess(
      RagBenchmarkReport candidate, List<RagBenchmarkReport> baselines) {
    Objects.requireNonNull(candidate, "candidate");
    Objects.requireNonNull(baselines, "baselines");

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
          Map.of());
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
          exclusions);
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
    return result(candidate, absoluteTier, verdict, qualifyingComparators, comparisons, exclusions);
  }

  private static RagProductionQualification result(
      RagBenchmarkReport candidate,
      RagPerformanceTier absoluteTier,
      RagQualificationVerdict verdict,
      List<String> qualifyingComparators,
      List<RagComparatorAssessment> comparisons,
      Map<String, String> exclusions) {
    return new RagProductionQualification(
        candidate.modelId(),
        candidate.backend(),
        candidate.artifactSha256(),
        absoluteTier,
        verdict,
        verdict == RagQualificationVerdict.QUALIFIED,
        qualifyingComparators,
        comparisons,
        exclusions);
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
    if (!candidate.settings().equals(baseline.settings())) {
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

  private record RelativeThreshold(double minimumDecodeRatio, double maximumEndToEndRatio) {}
}
