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

import com.fasterxml.jackson.databind.ObjectMapper;
import java.io.IOException;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;

class CertifiedRagEvidenceTest {
  private static final Path EVIDENCE_DIRECTORY =
      Path.of(System.getProperty("models.repositoryRoot"))
          .resolve(
              "benchmark-results/certified-20260724/rag/"
                  + "native-q8-prefix-cache/smollm2-360m-q8_0");

  private final ObjectMapper mapper = new ObjectMapper();

  @Test
  void cachedRustFfmQualifiesAgainstTheSameHostOllamaControl() throws Exception {
    RagBenchmarkReport candidate = report("smollm2-360m-rust-ffm-prefix-cache-grounded.json");
    RagBenchmarkReport ollama = report("smollm2-360m-ollama-current-grounded.json");
    RagBenchmarkReport llama = report("smollm2-360m-llama-current-grounded.json");

    RagProductionQualification qualification =
        RagProductionQualificationPolicy.assess(candidate, List.of(llama, ollama));

    assertThat(candidate.performanceTier()).isEqualTo(RagPerformanceTier.PRODUCTION_READY);
    assertThat(qualification.qualified()).isTrue();
    assertThat(qualification.qualifyingComparators()).containsExactly("ollama");
    assertThat(qualification.exclusions()).isEmpty();
    assertThat(qualification.comparisons())
        .filteredOn(comparison -> comparison.comparatorBackend().equals("ollama"))
        .singleElement()
        .satisfies(
            comparison -> {
              assertThat(comparison.decodeThroughputRatio()).isBetween(1.00, 1.01);
              assertThat(comparison.endToEndLatencyRatio()).isBetween(0.77, 0.79);
            });
  }

  @Test
  void prefixReusePreservesOutputsAndFrameworkContracts() throws Exception {
    RagBenchmarkReport uncached = report("smollm2-360m-rust-ffm-grounded.json");
    RagBenchmarkReport plain = report("smollm2-360m-rust-ffm-prefix-cache-grounded.json");
    RagBenchmarkReport langChain4j =
        report("smollm2-360m-langchain4j-rust-ffm-prefix-cache-grounded.json");
    RagBenchmarkReport springAi =
        report("smollm2-360m-spring-ai-rust-ffm-prefix-cache-grounded.json");

    assertThat(plain.summary().totalCacheReadInputTokens()).isEqualTo(3_153);
    assertThat(plain.summary().totalCacheWriteInputTokens()).isEqualTo(1_911);
    assertThat(plain.runs())
        .extracting(run -> run.grounding().rawText())
        .containsExactlyElementsOf(
            uncached.runs().stream().map(run -> run.grounding().rawText()).toList());

    List<String> expectedPromptHashes = plain.runs().stream().map(RagRun::promptSha256).toList();
    List<String> expectedRawOutputs =
        plain.runs().stream().map(run -> run.grounding().rawText()).toList();
    for (RagBenchmarkReport frameworkReport : List.of(langChain4j, springAi)) {
      assertThat(frameworkReport.performanceTier()).isEqualTo(RagPerformanceTier.PRODUCTION_READY);
      assertThat(frameworkReport.runs())
          .extracting(RagRun::promptSha256)
          .containsExactlyElementsOf(expectedPromptHashes);
      assertThat(frameworkReport.runs())
          .extracting(run -> run.grounding().rawText())
          .containsExactlyElementsOf(expectedRawOutputs);
    }
  }

  private RagBenchmarkReport report(String filename) throws IOException {
    return mapper.readValue(
        EVIDENCE_DIRECTORY.resolve(filename).toFile(), RagBenchmarkReport.class);
  }
}
