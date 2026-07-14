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

import com.fasterxml.jackson.databind.ObjectMapper;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class BenchmarkComparisonCliTest {

  @TempDir Path directory;

  @Test
  void writesAValidatedPortableComparisonReport() throws Exception {
    ObjectMapper mapper = new ObjectMapper();
    Path candidate = directory.resolve("candidate.json");
    Path reference = directory.resolve("reference.json");
    Path output = directory.resolve("comparison.json");
    mapper.writeValue(
        candidate.toFile(),
        BenchmarkComparisonValidatorTest.report(
            "pure-java", "artifact-a", "output-a", false, 10, 500));
    mapper.writeValue(
        reference.toFile(),
        BenchmarkComparisonValidatorTest.report(
            "llama.cpp", "artifact-a", "output-a", false, 20, 100));

    BenchmarkComparison comparison = BenchmarkComparisonCli.compare(candidate, reference, output);

    assertThat(comparison.comparable()).isTrue();
    assertThat(output).exists();
    assertThat(mapper.readTree(output.toFile()).path("decodeThroughputRatio").asDouble())
        .isEqualTo(0.5);
    assertThat(mapper.readTree(output.toFile()).path("candidateSummary").path("p50TtftMillis").asDouble())
        .isEqualTo(500);
  }
}
