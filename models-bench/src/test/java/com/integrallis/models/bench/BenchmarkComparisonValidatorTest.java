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

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;

class BenchmarkComparisonValidatorTest {

  @Test
  void comparesOnlyEquivalentSuccessfulEvidence() {
    BenchmarkReport candidate = report("pure-java", "artifact-a", "output-a", false, 10, 500);
    BenchmarkReport reference = report("llama.cpp", "artifact-a", "output-a", false, 20, 100);

    BenchmarkComparison comparison = BenchmarkComparisonValidator.compare(candidate, reference);

    assertThat(comparison.comparable()).isTrue();
    assertThat(comparison.violations()).isEmpty();
    assertThat(comparison.exactOutputMatch()).isTrue();
    assertThat(comparison.decodeThroughputRatio()).isEqualTo(0.5);
    assertThat(comparison.ttftRatio()).isEqualTo(5.0);
    assertThat(comparison.relativePerformance()).isEqualTo(RelativePerformance.VIABLE);
  }

  @Test
  void rejectsDifferentArtifactsOrGeneratedOutputs() {
    BenchmarkReport candidate = report("pure-java", "artifact-a", "output-a", false, 10, 500);
    BenchmarkReport reference = report("llama.cpp", "artifact-b", "output-b", false, 20, 100);

    BenchmarkComparison comparison = BenchmarkComparisonValidator.compare(candidate, reference);

    assertThat(comparison.comparable()).isFalse();
    assertThat(comparison.violations())
        .anyMatch(message -> message.contains("artifact"))
        .anyMatch(message -> message.contains("output"));
  }

  @Test
  void rejectsDirtyOrUndersampledEvidence() {
    BenchmarkReport candidate = report("pure-java", "artifact-a", "output-a", true, 10, 500);
    BenchmarkReport reference = report("llama.cpp", "artifact-a", "output-a", false, 20, 100);
    candidate =
        new BenchmarkReport(
            candidate.schemaVersion(),
            candidate.createdAt(),
            candidate.backend(),
            candidate.modelId(),
            candidate.model(),
            candidate.artifactSha256(),
            candidate.artifactSizeBytes(),
            candidate.backendIdentity(),
            candidate.load(),
            candidate.run(),
            candidate.environment(),
            new PerformanceSummary(1, 400, 500, 100, 120, 5, 10, 100, 9, 9),
            candidate.performanceTier(),
            candidate.trials().subList(0, 9));

    BenchmarkComparison comparison = BenchmarkComparisonValidator.compare(candidate, reference);

    assertThat(comparison.comparable()).isFalse();
    assertThat(comparison.violations())
        .anyMatch(message -> message.contains("dirty"))
        .anyMatch(message -> message.contains("ten successful"));
  }

  static BenchmarkReport report(
      String backend,
      String artifactSha256,
      String outputSha256,
      boolean dirty,
      double decodeTokensPerSecond,
      double ttftMillis) {
    List<TrialMeasurement> trials =
        java.util.stream.IntStream.range(0, 10)
            .mapToObj(
                ignored ->
                    TrialMeasurement.success(
                        ttftMillis,
                        ttftMillis + 1_000,
                        5,
                        decodeTokensPerSecond,
                        4,
                        16,
                        100,
                        50,
                        outputSha256))
            .toList();
    BenchmarkEnvironment environment =
        new BenchmarkEnvironment(
            "host",
            "Test OS",
            "1",
            "x86_64",
            "Test CPU",
            8,
            16_000,
            "25",
            "Vendor",
            "VM",
            "25",
            List.of("--add-modules=jdk.incubator.vector"),
            "PanamaVectorUtilSupport",
            true,
            256,
            256,
            Map.of(),
            "source-a",
            dirty,
            true);
    PerformanceSummary summary =
        new PerformanceSummary(
            1,
            ttftMillis,
            ttftMillis,
            1_000 / decodeTokensPerSecond,
            1_000 / decodeTokensPerSecond,
            5,
            decodeTokensPerSecond,
            100,
            10,
            10);
    return new BenchmarkReport(
        3,
        "2026-07-14T00:00:00Z",
        backend,
        "fixture",
        "fixture.gguf",
        artifactSha256,
        123,
        new BackendIdentity(
            backend,
            "1",
            "build",
            "fixture",
            "fixture.gguf",
            artifactSha256,
            true,
            Map.of()),
        LoadMeasurement.notMeasured(),
        new BenchmarkRun("prompt-a", 16, 2, 10, 256, 1, 0, 42),
        environment,
        summary,
        PerformanceTier.OFFLINE,
        trials);
  }
}
