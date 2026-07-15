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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class BenchmarkComparisonTest {

  @TempDir Path directory;

  @Test
  void comparesThroughputOnlyWhenArtifactRunAndEnvironmentMatch() {
    ComparisonReport comparison =
        BenchmarkComparison.compare(
            List.of(report("pure-java", "sha", 10), report("llama.cpp", "sha", 20)));

    BackendComparison pureJava = comparison.backends().getFirst();
    assertThat(pureJava.backend()).isEqualTo("pure-java");
    assertThat(pureJava.decodeRatioToLlamaCpp()).isEqualTo(0.5);
    assertThat(pureJava.relativePerformance()).isEqualTo(RelativePerformance.VIABLE);
    assertThat(ComparisonMarkdown.render(comparison))
        .contains("| pure-java |")
        .contains("| llama.cpp |")
        .contains("50.0%")
        .contains("| VIABLE |");
  }

  @Test
  void rejectsDifferentModelBytes() {
    assertThatThrownBy(
            () ->
                BenchmarkComparison.compare(
                    List.of(report("pure-java", "sha-a", 10), report("llama.cpp", "sha-b", 20))))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("artifact SHA-256");
  }

  @Test
  void compareCommandWritesJsonAndMarkdownEvidence() throws Exception {
    ObjectMapper mapper = new ObjectMapper();
    Path pureJava = directory.resolve("pure-java.json");
    Path llamaCpp = directory.resolve("llama.cpp.json");
    Path json = directory.resolve("comparison.json");
    Path markdown = directory.resolve("comparison.md");
    mapper.writeValue(pureJava.toFile(), report("pure-java", "sha", 10));
    mapper.writeValue(llamaCpp.toFile(), report("llama.cpp", "sha", 20));

    InferenceBenchmarkCli.main(
        new String[] {
          "compare",
          "--pure-java",
          pureJava.toString(),
          "--llama-cpp",
          llamaCpp.toString(),
          "--json",
          json.toString(),
          "--markdown",
          markdown.toString()
        });

    assertThat(json).isRegularFile();
    assertThat(Files.readString(markdown)).contains("| pure-java |");
  }

  private static BenchmarkReport report(String backend, String sha, double decodeTokensPerSecond) {
    BenchmarkRun run = new BenchmarkRun("prompt-sha", 64, 2, 10, 2048, 8, 0, 1, 1, 1, true, 42);
    BenchmarkEnvironment environment =
        new BenchmarkEnvironment(
            "bench-host",
            "Linux",
            "6.8",
            "amd64",
            "EPYC",
            8,
            32L << 30,
            "25.0.3",
            "Temurin",
            "OpenJDK");
    PerformanceSummary summary =
        new PerformanceSummary(100, 200, 250, 50, 60, 100, decodeTokensPerSecond, 1L << 30, 10, 10);
    return new BenchmarkReport(
        2,
        "2026-07-14T00:00:00Z",
        backend,
        "version",
        "fixture",
        "fixture.gguf",
        sha,
        1024,
        run,
        environment,
        summary,
        PerformanceTier.RESPONSIVE,
        List.of());
  }
}
