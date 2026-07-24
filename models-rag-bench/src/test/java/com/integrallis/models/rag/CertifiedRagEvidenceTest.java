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
  private static final Path EVIDENCE_ROOT =
      Path.of(System.getProperty("models.repositoryRoot"))
          .resolve("benchmark-results/certified-20260724/rag/native-q8-prefix-cache");
  private static final Path SMOLLM2_EVIDENCE = EVIDENCE_ROOT.resolve("smollm2-360m-q8_0");
  private static final Path QWEN3_1_7B_EVIDENCE = EVIDENCE_ROOT.resolve("qwen3-1.7b-q8_0");
  private static final Path QWEN2_5_CODER_EVIDENCE =
      EVIDENCE_ROOT.resolve("qwen2.5-coder-0.5b-q8_0");

  private final ObjectMapper mapper = new ObjectMapper();

  @Test
  void cachedRustFfmQualifiesAgainstTheSameHostOllamaControl() throws Exception {
    RagBenchmarkReport candidate =
        report(SMOLLM2_EVIDENCE, "smollm2-360m-rust-ffm-prefix-cache-grounded.json");
    RagBenchmarkReport ollama =
        report(SMOLLM2_EVIDENCE, "smollm2-360m-ollama-current-grounded.json");
    RagBenchmarkReport llama = report(SMOLLM2_EVIDENCE, "smollm2-360m-llama-current-grounded.json");

    RagProductionQualification qualification =
        RagProductionQualificationPolicy.assess(candidate, List.of(llama, ollama));

    assertThat(candidate.performanceTier()).isEqualTo(RagPerformanceTier.PRODUCTION_READY);
    assertThat(qualification.qualified()).isTrue();
    assertThat(qualification.modelAnswerCount()).isEqualTo(9);
    assertThat(qualification.modelAnswerRate()).isEqualTo(1.0 / 3.0);
    assertThat(qualification.modelAnswerCorrectRate()).isEqualTo(1.0);
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
  void qwen3RustFfmQualifiesAgainstBothExactArtifactControls() throws Exception {
    RagBenchmarkReport candidate =
        report(QWEN3_1_7B_EVIDENCE, "qwen3-1.7b-rust-ffm-prefix-cache-grounded.json");
    RagBenchmarkReport pureJava =
        report(QWEN3_1_7B_EVIDENCE, "qwen3-1.7b-pure-java-prefix-cache-grounded.json");
    RagBenchmarkReport ollama = report(QWEN3_1_7B_EVIDENCE, "qwen3-1.7b-ollama-grounded.json");
    RagBenchmarkReport llama = report(QWEN3_1_7B_EVIDENCE, "qwen3-1.7b-llama-grounded.json");

    RagProductionQualification qualification =
        RagProductionQualificationPolicy.assess(candidate, List.of(llama, ollama));

    assertThat(candidate.artifactSha256())
        .isEqualTo("061b54daade076b5d3362dac252678d17da8c68f07560be70818cace6590cb1a");
    assertThat(candidate.performanceTier()).isEqualTo(RagPerformanceTier.USABLE);
    assertThat(pureJava.performanceTier()).isEqualTo(RagPerformanceTier.OFFLINE);
    assertThat(qualification.qualified()).isTrue();
    assertThat(qualification.modelAnswerCount()).isEqualTo(12);
    assertThat(qualification.modelAnswerRate()).isEqualTo(12.0 / 27.0);
    assertThat(qualification.modelAnswerCorrectRate()).isEqualTo(1.0);
    assertThat(qualification.qualifyingComparators()).containsExactly("llama.cpp", "ollama");
    assertThat(candidate.summary().ttftMillis().p95())
        .isLessThan(pureJava.summary().ttftMillis().p95() * 0.38);
    assertThat(candidate.summary().endToEndMillis().p95())
        .isLessThan(pureJava.summary().endToEndMillis().p95() * 0.66);
    assertThat(candidate.runs())
        .extracting(run -> run.grounding().decision())
        .containsExactlyElementsOf(
            pureJava.runs().stream().map(run -> run.grounding().decision()).toList());
    assertThat(candidate.runs())
        .extracting(run -> run.evaluation().correct())
        .containsExactlyElementsOf(
            pureJava.runs().stream().map(run -> run.evaluation().correct()).toList());
    assertThat(qualification.comparisons())
        .filteredOn(comparison -> comparison.comparatorBackend().equals("llama.cpp"))
        .singleElement()
        .satisfies(
            comparison -> {
              assertThat(comparison.decodeThroughputRatio()).isBetween(0.70, 0.71);
              assertThat(comparison.endToEndLatencyRatio()).isBetween(1.26, 1.27);
            });
    assertThat(qualification.comparisons())
        .filteredOn(comparison -> comparison.comparatorBackend().equals("ollama"))
        .singleElement()
        .satisfies(
            comparison -> {
              assertThat(comparison.decodeThroughputRatio()).isBetween(1.00, 1.01);
              assertThat(comparison.endToEndLatencyRatio()).isBetween(1.15, 1.16);
            });
  }

  @Test
  void qwen25CoderRustFfmQualifiesOnTheCodingWorkload() throws Exception {
    RagBenchmarkReport candidate =
        report(QWEN2_5_CODER_EVIDENCE, "qwen2.5-coder-0.5b-rust-ffm-coding-v4-grounded.json");
    RagBenchmarkReport pureJava =
        report(QWEN2_5_CODER_EVIDENCE, "qwen2.5-coder-0.5b-pure-java-coding-v4-grounded.json");
    RagBenchmarkReport ollama =
        report(QWEN2_5_CODER_EVIDENCE, "qwen2.5-coder-0.5b-ollama-coding-v4-grounded.json");
    RagBenchmarkReport llama =
        report(QWEN2_5_CODER_EVIDENCE, "qwen2.5-coder-0.5b-llama-coding-v4-grounded.json");

    RagProductionQualification qualification =
        RagProductionQualificationPolicy.assess(candidate, List.of(llama, ollama));
    RagProductionQualification pureJavaQualification =
        RagProductionQualificationPolicy.assess(pureJava, List.of(llama, ollama));

    assertThat(candidate.artifactSha256())
        .isEqualTo("e1a77721fa97d412f121878223eec81fb4ae6f271e18f922d746711f67b344d1");
    assertThat(candidate.settings().workload()).isEqualTo(RagWorkload.CODING.id());
    assertThat(candidate.settings().corpusSha256())
        .isEqualTo("6841c286837b4c45c06fe8d103b2e044b61a1bfe75a61b64fa04c7ca31b20e45");
    assertThat(candidate.settings().groundingPolicy())
        .isEqualTo("trusted-provenance-clause-anchors-extractive-fallback-v4");
    assertThat(candidate.performanceTier()).isEqualTo(RagPerformanceTier.PRODUCTION_READY);
    assertThat(qualification.qualified()).isTrue();
    assertThat(pureJava.performanceTier()).isEqualTo(RagPerformanceTier.PRODUCTION_READY);
    assertThat(pureJavaQualification.qualified()).isTrue();
    assertThat(qualification.modelAnswerCount()).isEqualTo(15);
    assertThat(qualification.modelAnswerRate()).isEqualTo(15.0 / 27.0);
    assertThat(qualification.modelAnswerCorrectRate()).isEqualTo(1.0);
    assertThat(qualification.qualifyingComparators()).containsExactly("llama.cpp", "ollama");
    assertThat(candidate.runs())
        .filteredOn(
            run ->
                run.grounding().decision() == GroundingDecision.MODEL_ANSWER_WITH_DERIVED_CITATIONS)
        .hasSize(15)
        .allSatisfy(run -> assertThat(run.evaluation().correct()).isTrue());
    assertThat(candidate.runs())
        .extracting(run -> run.grounding().decision())
        .containsExactlyElementsOf(
            pureJava.runs().stream().map(run -> run.grounding().decision()).toList())
        .containsExactlyElementsOf(
            ollama.runs().stream().map(run -> run.grounding().decision()).toList())
        .containsExactlyElementsOf(
            llama.runs().stream().map(run -> run.grounding().decision()).toList());
    assertThat(candidate.runs())
        .extracting(run -> run.grounding().text())
        .containsExactlyElementsOf(
            pureJava.runs().stream().map(run -> run.grounding().text()).toList());
    assertThat(candidate.summary().p50PrefillTokensPerSecond())
        .isGreaterThan(pureJava.summary().p50PrefillTokensPerSecond() * 1.43);
    assertThat(candidate.summary().ttftMillis().p95())
        .isLessThan(pureJava.summary().ttftMillis().p95() * 0.71);
    assertThat(candidate.summary().endToEndMillis().p95())
        .isLessThan(pureJava.summary().endToEndMillis().p95() * 0.84);
    assertThat(candidate.summary().totalCpuMillis())
        .isLessThan(pureJava.summary().totalCpuMillis() * 0.80);
    assertThat(qualification.comparisons())
        .filteredOn(comparison -> comparison.comparatorBackend().equals("llama.cpp"))
        .singleElement()
        .satisfies(
            comparison -> {
              assertThat(comparison.decodeThroughputRatio()).isBetween(0.66, 0.68);
              assertThat(comparison.endToEndLatencyRatio()).isBetween(1.33, 1.34);
            });
    assertThat(qualification.comparisons())
        .filteredOn(comparison -> comparison.comparatorBackend().equals("ollama"))
        .singleElement()
        .satisfies(
            comparison -> {
              assertThat(comparison.decodeThroughputRatio()).isBetween(1.32, 1.33);
              assertThat(comparison.endToEndLatencyRatio()).isBetween(0.84, 0.85);
            });
  }

  @Test
  void prefixReusePreservesOutputsAndFrameworkContracts() throws Exception {
    RagBenchmarkReport uncached = report(SMOLLM2_EVIDENCE, "smollm2-360m-rust-ffm-grounded.json");
    RagBenchmarkReport plain =
        report(SMOLLM2_EVIDENCE, "smollm2-360m-rust-ffm-prefix-cache-grounded.json");
    RagBenchmarkReport langChain4j =
        report(SMOLLM2_EVIDENCE, "smollm2-360m-langchain4j-rust-ffm-prefix-cache-grounded.json");
    RagBenchmarkReport springAi =
        report(SMOLLM2_EVIDENCE, "smollm2-360m-spring-ai-rust-ffm-prefix-cache-grounded.json");

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

  private RagBenchmarkReport report(Path directory, String filename) throws IOException {
    return mapper.readValue(directory.resolve(filename).toFile(), RagBenchmarkReport.class);
  }
}
