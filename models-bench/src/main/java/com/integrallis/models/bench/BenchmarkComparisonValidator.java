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

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;

/** Rejects invalid cross-backend comparisons before computing relative performance. */
public final class BenchmarkComparisonValidator {

  private static final int MINIMUM_TRIALS = 10;

  private BenchmarkComparisonValidator() {}

  public static BenchmarkComparison compare(
      BenchmarkReport candidate, BenchmarkReport reference) {
    Objects.requireNonNull(candidate, "candidate");
    Objects.requireNonNull(reference, "reference");
    List<String> violations = new ArrayList<>();

    require("pure-java".equals(candidate.backend()), "candidate backend must be pure-java", violations);
    require("llama.cpp".equals(reference.backend()), "reference backend must be llama.cpp", violations);
    require(
        candidate.schemaVersion() == reference.schemaVersion(),
        "report schema versions differ",
        violations);
    require(
        Objects.equals(candidate.modelId(), reference.modelId()),
        "model identifiers differ",
        violations);
    require(
        Objects.equals(candidate.artifactSha256(), reference.artifactSha256()),
        "artifact SHA-256 values differ",
        violations);
    validateArtifactAttestation(candidate, "candidate", violations);
    validateArtifactAttestation(reference, "reference", violations);
    require(Objects.equals(candidate.run(), reference.run()), "benchmark workloads differ", violations);
    validateEnvironment(candidate.environment(), reference.environment(), violations);
    validateTrials(candidate, "candidate", violations);
    validateTrials(reference, "reference", violations);

    boolean exactOutputMatch = exactOutputMatch(candidate.trials(), reference.trials());
    require(exactOutputMatch, "generated output or token counts differ", violations);

    boolean comparable = violations.isEmpty();
    double throughputRatio =
        comparable
            ? ratio(
                candidate.summary().p50DecodeTokensPerSecond(),
                reference.summary().p50DecodeTokensPerSecond())
            : 0;
    double ttftRatio =
        comparable
            ? ratio(candidate.summary().p50TtftMillis(), reference.summary().p50TtftMillis())
            : 0;
    RelativePerformance relativePerformance =
        comparable ? BenchmarkPolicy.relativeTier(throughputRatio) : null;

    return new BenchmarkComparison(
        1,
        candidate.modelId(),
        candidate.artifactSha256(),
        candidate.backend(),
        reference.backend(),
        comparable,
        violations,
        exactOutputMatch,
        throughputRatio,
        ttftRatio,
        relativePerformance,
        candidate.run(),
        candidate.summary(),
        reference.summary(),
        candidate.environment() == null ? "unknown" : candidate.environment().sourceRevision(),
        reference.backendIdentity() == null ? "unknown" : reference.backendIdentity().build());
  }

  private static void validateArtifactAttestation(
      BenchmarkReport report, String label, List<String> violations) {
    BackendIdentity identity = report.backendIdentity();
    require(identity != null, label + " backend identity is missing", violations);
    if (identity == null) {
      return;
    }
    require(identity.artifactVerified(), label + " artifact is not verified", violations);
    require(
        Objects.equals(report.artifactSha256(), identity.runtimeArtifactSha256()),
        label + " runtime artifact does not match its report",
        violations);
  }

  private static void validateEnvironment(
      BenchmarkEnvironment candidate,
      BenchmarkEnvironment reference,
      List<String> violations) {
    require(candidate != null && reference != null, "benchmark environment is missing", violations);
    if (candidate == null || reference == null) {
      return;
    }
    require(
        Objects.equals(candidate.architecture(), reference.architecture())
            && Objects.equals(candidate.cpuModel(), reference.cpuModel())
            && candidate.processors() == reference.processors()
            && candidate.physicalMemoryBytes() == reference.physicalMemoryBytes(),
        "benchmark hardware differs",
        violations);
    require(
        Objects.equals(candidate.sourceRevision(), reference.sourceRevision()),
        "benchmark source revisions differ",
        violations);
    require(
        candidate.sourceDirtyKnown() && reference.sourceDirtyKnown(),
        "benchmark source dirty state is unknown",
        violations);
    require(
        !candidate.sourceDirty() && !reference.sourceDirty(),
        "benchmark source is dirty",
        violations);
  }

  private static void validateTrials(
      BenchmarkReport report, String label, List<String> violations) {
    PerformanceSummary summary = report.summary();
    require(summary != null, label + " summary is missing", violations);
    if (summary == null) {
      return;
    }
    require(
        summary.totalTrials() >= MINIMUM_TRIALS
            && summary.successfulTrials() == summary.totalTrials()
            && report.trials().size() == summary.totalTrials(),
        label + " must contain at least ten successful measured trials",
        violations);
  }

  private static boolean exactOutputMatch(
      List<TrialMeasurement> candidate, List<TrialMeasurement> reference) {
    if (candidate == null || reference == null || candidate.size() != reference.size()) {
      return false;
    }
    for (int index = 0; index < candidate.size(); index++) {
      TrialMeasurement left = candidate.get(index);
      TrialMeasurement right = reference.get(index);
      if (left.inputTokens() != right.inputTokens()
          || left.outputTokens() != right.outputTokens()
          || !Objects.equals(left.outputSha256(), right.outputSha256())) {
        return false;
      }
    }
    return true;
  }

  private static double ratio(double numerator, double denominator) {
    if (!Double.isFinite(numerator)
        || !Double.isFinite(denominator)
        || numerator < 0
        || denominator <= 0) {
      throw new IllegalArgumentException("comparison metrics must be finite and positive");
    }
    return numerator / denominator;
  }

  private static void require(boolean condition, String violation, List<String> violations) {
    if (!condition) {
      violations.add(violation);
    }
  }
}
