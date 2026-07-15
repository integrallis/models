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

import java.time.Instant;
import java.util.HashSet;
import java.util.List;
import java.util.Set;

/** Validates and combines benchmark reports that are safe to compare. */
public final class BenchmarkComparison {

  private BenchmarkComparison() {}

  public static ComparisonReport compare(List<BenchmarkReport> reports) {
    if (reports == null || reports.size() < 2) {
      throw new IllegalArgumentException("at least two benchmark reports are required");
    }
    BenchmarkReport reference = reports.getFirst();
    BenchmarkReport llamaCpp =
        reports.stream()
            .filter(report -> "llama.cpp".equals(report.backend()))
            .findFirst()
            .orElseThrow(() -> new IllegalArgumentException("a llama.cpp report is required"));
    double llamaDecode = llamaCpp.summary().p50DecodeTokensPerSecond();
    if (!(llamaDecode > 0)) {
      throw new IllegalArgumentException("llama.cpp decode throughput must be positive");
    }

    Set<String> backends = new HashSet<>();
    for (BenchmarkReport report : reports) {
      validateComparable(reference, report);
      if (!backends.add(report.backend())) {
        throw new IllegalArgumentException("duplicate backend report: " + report.backend());
      }
    }

    List<BackendComparison> rows =
        reports.stream()
            .map(
                report -> {
                  PerformanceSummary summary = report.summary();
                  double ratio = summary.p50DecodeTokensPerSecond() / llamaDecode;
                  return new BackendComparison(
                      report.backend(),
                      report.backendVersion(),
                      report.performanceTier(),
                      summary.loadMillis(),
                      summary.p95TtftMillis(),
                      summary.p95TpotMillis(),
                      summary.p50PrefillTokensPerSecond(),
                      summary.p50DecodeTokensPerSecond(),
                      summary.peakRssBytes(),
                      ratio,
                      BenchmarkPolicy.relativeTier(ratio));
                })
            .toList();
    return new ComparisonReport(
        1,
        Instant.now().toString(),
        reference.modelId(),
        reference.artifactSha256(),
        reference.artifactSizeBytes(),
        reference.run(),
        reference.environment(),
        rows);
  }

  private static void validateComparable(BenchmarkReport reference, BenchmarkReport candidate) {
    requireEqual("model ID", reference.modelId(), candidate.modelId());
    requireEqual("artifact SHA-256", reference.artifactSha256(), candidate.artifactSha256());
    if (reference.artifactSha256() == null || reference.artifactSha256().isBlank()) {
      throw new IllegalArgumentException("artifact SHA-256 is required");
    }
    if (reference.artifactSizeBytes() != candidate.artifactSizeBytes()) {
      throw new IllegalArgumentException("artifact size does not match");
    }
    requireEqual("benchmark run", reference.run(), candidate.run());
    requireEqual("benchmark environment", reference.environment(), candidate.environment());
    PerformanceSummary summary = candidate.summary();
    if (summary.totalTrials() < 10 || summary.successfulTrials() != summary.totalTrials()) {
      throw new IllegalArgumentException(
          "backend does not have ten successful measured trials: " + candidate.backend());
    }
  }

  private static void requireEqual(String field, Object expected, Object actual) {
    if (!java.util.Objects.equals(expected, actual)) {
      throw new IllegalArgumentException(field + " does not match");
    }
  }
}
