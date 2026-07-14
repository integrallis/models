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

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/** Validates two raw reports and writes a portable relative-performance summary. */
public final class BenchmarkComparisonCli {

  private BenchmarkComparisonCli() {}

  public static void main(String[] args) throws IOException {
    if (args.length != 3) {
      throw new IllegalArgumentException(
          "compare requires: <pure-java-report> <llama.cpp-report> <output>");
    }
    Path output = Path.of(args[2]);
    BenchmarkComparison comparison = compare(Path.of(args[0]), Path.of(args[1]), output);
    System.out.printf(
        "%s: comparable=%s decode-ratio=%.3f ttft-ratio=%.3f tier=%s%nreport: %s%n",
        comparison.modelId(),
        comparison.comparable(),
        comparison.decodeThroughputRatio(),
        comparison.ttftRatio(),
        comparison.relativePerformance(),
        output.toAbsolutePath());
    if (!comparison.comparable()) {
      throw new IllegalStateException(
          "benchmark reports are not comparable: " + comparison.violations());
    }
  }

  static BenchmarkComparison compare(Path candidate, Path reference, Path output)
      throws IOException {
    ObjectMapper mapper = new ObjectMapper();
    BenchmarkReport candidateReport = mapper.readValue(candidate.toFile(), BenchmarkReport.class);
    BenchmarkReport referenceReport = mapper.readValue(reference.toFile(), BenchmarkReport.class);
    BenchmarkComparison comparison =
        BenchmarkComparisonValidator.compare(candidateReport, referenceReport);
    Path parent = output.toAbsolutePath().getParent();
    if (parent != null) {
      Files.createDirectories(parent);
    }
    mapper
        .enable(SerializationFeature.INDENT_OUTPUT)
        .writeValue(output.toFile(), comparison);
    return comparison;
  }
}
