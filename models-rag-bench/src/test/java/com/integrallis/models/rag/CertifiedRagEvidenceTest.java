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
import com.integrallis.models.api.OptimizationStatus;
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
  private static final Path QWEN2_5_CODER_1_5B_Q8_EVIDENCE =
      EVIDENCE_ROOT.resolve("qwen2.5-coder-1.5b-q8_0");
  private static final Path QWEN3_0_6B_EVIDENCE =
      Path.of(System.getProperty("models.repositoryRoot"))
          .resolve("benchmark-results/certified-20260724/rag/native-q4-profiled/qwen3-0.6b-q4_0");
  private static final Path QWEN2_5_CODER_Q4_EVIDENCE =
      Path.of(System.getProperty("models.repositoryRoot"))
          .resolve(
              "benchmark-results/certified-20260724/rag/native-q4-profiled/"
                  + "qwen2.5-coder-0.5b-q4_0");
  private static final Path QWEN2_5_CODER_1_5B_Q4_EVIDENCE =
      Path.of(System.getProperty("models.repositoryRoot"))
          .resolve(
              "benchmark-results/certified-20260724/rag/native-q4-profiled/"
                  + "qwen2.5-coder-1.5b-q4_0");
  private static final Path QWEN2_5_GENERAL_Q4_K_M_EVIDENCE =
      Path.of(System.getProperty("models.repositoryRoot"))
          .resolve("benchmark-results/certified-20260725/rag/qwen2.5-0.5b-q4_k_m");
  private static final Path QWEN2_5_GENERAL_1_5B_Q4_K_M_EVIDENCE =
      Path.of(System.getProperty("models.repositoryRoot"))
          .resolve("benchmark-results/certified-20260725/rag/qwen2.5-1.5b-q4_k_m");

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
  void qwen3SixHundredMillionQualifiesPureJavaWithTheProfiledQ4Kernel() throws Exception {
    RagBenchmarkReport candidate =
        report(
            QWEN3_0_6B_EVIDENCE, "qwen3-0.6b-q4_0-pure-java-q4-unsigned-coding-v4-grounded.json");
    RagBenchmarkReport rustFfm =
        report(QWEN3_0_6B_EVIDENCE, "qwen3-0.6b-q4_0-rust-ffm-q4-unsigned-coding-v4-grounded.json");
    RagBenchmarkReport ollama =
        report(QWEN3_0_6B_EVIDENCE, "qwen3-0.6b-q4_0-ollama-coding-v4-grounded.json");
    RagBenchmarkReport llama =
        report(QWEN3_0_6B_EVIDENCE, "qwen3-0.6b-q4_0-llama-coding-v4-grounded.json");

    RagProductionQualification qualification =
        RagProductionQualificationPolicy.assess(candidate, List.of(llama, ollama));

    assertThat(candidate.artifactSha256())
        .isEqualTo("da2572f16c06133561ce56accaa822216f2391ef4d37fba427801cd6736417d4");
    assertThat(candidate.settings().workload()).isEqualTo(RagWorkload.CODING.id());
    assertThat(candidate.performanceTier()).isEqualTo(RagPerformanceTier.PRODUCTION_READY);
    assertThat(qualification.qualified()).isTrue();
    assertThat(qualification.modelAnswerCount()).isEqualTo(12);
    assertThat(qualification.modelAnswerRate()).isEqualTo(12.0 / 27.0);
    assertThat(qualification.modelAnswerCorrectRate()).isEqualTo(1.0);
    assertThat(qualification.qualifyingComparators()).containsExactly("llama.cpp", "ollama");
    assertThat(candidate.backendDiagnostics().optimization("q4-kernel"))
        .get()
        .satisfies(
            optimization -> {
              assertThat(optimization.status()).isEqualTo(OptimizationStatus.ENABLED);
              assertThat(optimization.settings()).containsEntry("requested", "unsigned-pairwise");
            });
    assertThat(candidate.summary().p50DecodeTokensPerSecond())
        .isGreaterThan(rustFfm.summary().p50DecodeTokensPerSecond());
    assertThat(candidate.summary().ttftMillis().p95())
        .isLessThan(rustFfm.summary().ttftMillis().p95());
    assertThat(candidate.runs())
        .filteredOn(run -> run.grounding().decision().modelContributed())
        .hasSize(12)
        .allSatisfy(run -> assertThat(run.evaluation().correct()).isTrue());
    assertThat(qualification.comparisons())
        .filteredOn(comparison -> comparison.comparatorBackend().equals("llama.cpp"))
        .singleElement()
        .satisfies(
            comparison -> {
              assertThat(comparison.decodeThroughputRatio()).isBetween(0.46, 0.47);
              assertThat(comparison.endToEndLatencyRatio()).isBetween(1.77, 1.79);
            });
    assertThat(qualification.comparisons())
        .filteredOn(comparison -> comparison.comparatorBackend().equals("ollama"))
        .singleElement()
        .satisfies(
            comparison -> {
              assertThat(comparison.decodeThroughputRatio()).isBetween(0.99, 1.00);
              assertThat(comparison.endToEndLatencyRatio()).isBetween(1.02, 1.03);
            });
  }

  @Test
  void qwen25CoderQ4QualifiesWithRustPrefillAndTheProfiledJavaDecodeKernel() throws Exception {
    RagBenchmarkReport candidate =
        report(
            QWEN2_5_CODER_Q4_EVIDENCE,
            "qwen2.5-coder-0.5b-q4_0-rust-ffm-q4-unsigned-t4-coding-v4-grounded.json");
    RagBenchmarkReport pureJava =
        report(
            QWEN2_5_CODER_Q4_EVIDENCE,
            "qwen2.5-coder-0.5b-q4_0-pure-java-q4-unsigned-coding-v4-grounded.json");
    RagBenchmarkReport unprofiledRust =
        report(
            QWEN2_5_CODER_Q4_EVIDENCE,
            "qwen2.5-coder-0.5b-q4_0-rust-ffm-default-coding-v4-grounded.json");
    RagBenchmarkReport ollama =
        report(QWEN2_5_CODER_Q4_EVIDENCE, "qwen2.5-coder-0.5b-q4_0-ollama-coding-v4-grounded.json");
    RagBenchmarkReport llama =
        report(QWEN2_5_CODER_Q4_EVIDENCE, "qwen2.5-coder-0.5b-q4_0-llama-coding-v4-grounded.json");

    RagProductionQualification qualification =
        RagProductionQualificationPolicy.assess(candidate, List.of(llama, ollama));
    RagProductionQualification pureJavaQualification =
        RagProductionQualificationPolicy.assess(pureJava, List.of(llama, ollama));

    assertThat(candidate.artifactSha256())
        .isEqualTo("9739055e046d62a937e5b7879012209ef40ebea8a1569a96028de491f3f091d5");
    assertThat(candidate.settings().workload()).isEqualTo(RagWorkload.CODING.id());
    assertThat(candidate.performanceTier()).isEqualTo(RagPerformanceTier.PRODUCTION_READY);
    assertThat(qualification.qualified()).isTrue();
    assertThat(qualification.qualifyingComparators()).containsExactly("ollama");
    assertThat(qualification.modelAnswerCount()).isEqualTo(12);
    assertThat(qualification.modelAnswerRate()).isEqualTo(12.0 / 27.0);
    assertThat(qualification.modelAnswerCorrectRate()).isEqualTo(1.0);
    assertThat(pureJava.performanceTier()).isEqualTo(RagPerformanceTier.PRODUCTION_READY);
    assertThat(pureJavaQualification.verdict())
        .isEqualTo(RagQualificationVerdict.FAILED_RELATIVE_GATE);
    assertThat(candidate.backendDiagnostics().optimization("q4-kernel"))
        .get()
        .satisfies(
            optimization -> {
              assertThat(optimization.status()).isEqualTo(OptimizationStatus.ENABLED);
              assertThat(optimization.settings()).containsEntry("requested", "unsigned-pairwise");
            });
    assertThat(candidate.backendDiagnostics().optimization("rust-q4-0-batched-matmul"))
        .get()
        .satisfies(
            optimization ->
                assertThat(optimization.status()).isEqualTo(OptimizationStatus.ENABLED));
    assertThat(candidate.backendDiagnostics().environment()).containsEntry("gguf-threads", "4");
    assertThat(unprofiledRust.backendDiagnostics().optimization("q4-kernel"))
        .get()
        .satisfies(
            optimization -> {
              assertThat(optimization.status()).isEqualTo(OptimizationStatus.DISABLED);
              assertThat(optimization.settings()).containsEntry("requested", "widened");
            });
    assertThat(candidate.runs())
        .extracting(run -> run.grounding().rawText())
        .containsExactlyElementsOf(
            unprofiledRust.runs().stream().map(run -> run.grounding().rawText()).toList());
    assertThat(candidate.summary().p50DecodeTokensPerSecond())
        .isGreaterThan(unprofiledRust.summary().p50DecodeTokensPerSecond() * 1.67);
    assertThat(candidate.summary().endToEndMillis().p95())
        .isLessThan(unprofiledRust.summary().endToEndMillis().p95() * 0.66);
    assertThat(candidate.summary().totalCpuMillis())
        .isLessThan(unprofiledRust.summary().totalCpuMillis() * 0.48);
    assertThat(candidate.runs())
        .filteredOn(run -> run.grounding().decision().modelContributed())
        .hasSize(12)
        .allSatisfy(run -> assertThat(run.evaluation().correct()).isTrue());
    assertThat(candidate.runs())
        .extracting(run -> run.grounding().decision())
        .containsExactlyElementsOf(
            pureJava.runs().stream().map(run -> run.grounding().decision()).toList());
    assertThat(candidate.summary().p50PrefillTokensPerSecond())
        .isGreaterThan(pureJava.summary().p50PrefillTokensPerSecond() * 2.5);
    assertThat(candidate.summary().ttftMillis().p95())
        .isLessThan(pureJava.summary().ttftMillis().p95() * 0.42);
    assertThat(candidate.summary().endToEndMillis().p95())
        .isLessThan(pureJava.summary().endToEndMillis().p95() * 0.78);
    assertThat(candidate.summary().totalCpuMillis())
        .isLessThan(pureJava.summary().totalCpuMillis() * 0.51);
    assertThat(qualification.comparisons())
        .filteredOn(comparison -> comparison.comparatorBackend().equals("llama.cpp"))
        .singleElement()
        .satisfies(
            comparison -> {
              assertThat(comparison.decodeThroughputRatio()).isBetween(0.35, 0.36);
              assertThat(comparison.endToEndLatencyRatio()).isBetween(2.31, 2.33);
              assertThat(comparison.qualified()).isFalse();
            });
    assertThat(qualification.comparisons())
        .filteredOn(comparison -> comparison.comparatorBackend().equals("ollama"))
        .singleElement()
        .satisfies(
            comparison -> {
              assertThat(comparison.decodeThroughputRatio()).isBetween(0.80, 0.81);
              assertThat(comparison.endToEndLatencyRatio()).isBetween(1.21, 1.22);
              assertThat(comparison.qualified()).isTrue();
            });
  }

  @Test
  void qwen25Coder15BQ4QualifiesAsUsableAgainstBothLocalControls() throws Exception {
    RagBenchmarkReport candidate =
        report(
            QWEN2_5_CODER_1_5B_Q4_EVIDENCE,
            "qwen2.5-coder-1.5b-q4_0-rust-ffm-q4-unsigned-t4-coding-v4-grounded.json");
    RagBenchmarkReport unprofiledRust =
        report(
            QWEN2_5_CODER_1_5B_Q4_EVIDENCE,
            "qwen2.5-coder-1.5b-q4_0-rust-ffm-default-coding-v4-grounded.json");
    RagBenchmarkReport ollama =
        report(
            QWEN2_5_CODER_1_5B_Q4_EVIDENCE,
            "qwen2.5-coder-1.5b-q4_0-ollama-coding-v4-grounded.json");
    RagBenchmarkReport llama =
        report(
            QWEN2_5_CODER_1_5B_Q4_EVIDENCE,
            "qwen2.5-coder-1.5b-q4_0-llama-coding-v4-grounded.json");

    RagProductionQualification qualification =
        RagProductionQualificationPolicy.assess(candidate, List.of(llama, ollama));

    assertThat(candidate.artifactSha256())
        .isEqualTo("aa8353e0d0fca3a0041828701e90db7635197400f040676d11d7798665fa316e");
    assertThat(candidate.performanceTier()).isEqualTo(RagPerformanceTier.USABLE);
    assertThat(qualification.qualified()).isTrue();
    assertThat(qualification.qualifyingComparators()).containsExactly("llama.cpp", "ollama");
    assertThat(qualification.modelAnswerCount()).isEqualTo(15);
    assertThat(qualification.modelAnswerRate()).isEqualTo(15.0 / 27.0);
    assertThat(qualification.modelAnswerCorrectRate()).isEqualTo(1.0);
    assertThat(candidate.backendDiagnostics().optimization("q4-kernel"))
        .get()
        .satisfies(
            optimization -> {
              assertThat(optimization.status()).isEqualTo(OptimizationStatus.ENABLED);
              assertThat(optimization.settings()).containsEntry("requested", "unsigned-pairwise");
            });
    assertThat(candidate.backendDiagnostics().environment()).containsEntry("gguf-threads", "4");
    assertThat(candidate.runs())
        .extracting(run -> run.grounding().rawText())
        .containsExactlyElementsOf(
            unprofiledRust.runs().stream().map(run -> run.grounding().rawText()).toList());
    assertThat(candidate.summary().p50DecodeTokensPerSecond())
        .isGreaterThan(unprofiledRust.summary().p50DecodeTokensPerSecond() * 1.8);
    assertThat(candidate.summary().endToEndMillis().p95())
        .isLessThan(unprofiledRust.summary().endToEndMillis().p95() * 0.70);
    assertThat(candidate.summary().totalCpuMillis())
        .isLessThan(unprofiledRust.summary().totalCpuMillis() * 0.51);
    assertThat(candidate.runs())
        .filteredOn(run -> run.grounding().decision().modelContributed())
        .hasSize(15)
        .allSatisfy(run -> assertThat(run.evaluation().correct()).isTrue());
    assertThat(qualification.comparisons())
        .filteredOn(comparison -> comparison.comparatorBackend().equals("llama.cpp"))
        .singleElement()
        .satisfies(
            comparison -> {
              assertThat(comparison.decodeThroughputRatio()).isBetween(0.48, 0.50);
              assertThat(comparison.endToEndLatencyRatio()).isBetween(1.53, 1.54);
              assertThat(comparison.qualified()).isTrue();
            });
    assertThat(qualification.comparisons())
        .filteredOn(comparison -> comparison.comparatorBackend().equals("ollama"))
        .singleElement()
        .satisfies(
            comparison -> {
              assertThat(comparison.decodeThroughputRatio()).isBetween(0.80, 0.81);
              assertThat(comparison.endToEndLatencyRatio()).isBetween(1.39, 1.40);
              assertThat(comparison.qualified()).isTrue();
            });
  }

  @Test
  void qwen25Coder15BQ8RequiresRustPrefillToClearTheOllamaGate() throws Exception {
    RagBenchmarkReport candidate =
        report(
            QWEN2_5_CODER_1_5B_Q8_EVIDENCE,
            "qwen2.5-coder-1.5b-q8_0-rust-ffm-coding-v4-grounded.json");
    RagBenchmarkReport pureJava =
        report(
            QWEN2_5_CODER_1_5B_Q8_EVIDENCE,
            "qwen2.5-coder-1.5b-q8_0-pure-java-coding-v4-grounded.json");
    RagBenchmarkReport ollama =
        report(
            QWEN2_5_CODER_1_5B_Q8_EVIDENCE,
            "qwen2.5-coder-1.5b-q8_0-ollama-coding-v4-grounded.json");
    RagBenchmarkReport llama =
        report(
            QWEN2_5_CODER_1_5B_Q8_EVIDENCE,
            "qwen2.5-coder-1.5b-q8_0-llama-coding-v4-grounded.json");

    RagProductionQualification qualification =
        RagProductionQualificationPolicy.assess(candidate, List.of(llama, ollama));
    RagProductionQualification pureJavaQualification =
        RagProductionQualificationPolicy.assess(pureJava, List.of(llama, ollama));

    assertThat(candidate.artifactSha256())
        .isEqualTo("507de59046601282ba768a9789900e6ccf60ed93ddf346730b7c68eb0715bc47");
    assertThat(candidate.performanceTier()).isEqualTo(RagPerformanceTier.USABLE);
    assertThat(qualification.qualified()).isTrue();
    assertThat(qualification.qualifyingComparators()).containsExactly("llama.cpp", "ollama");
    assertThat(qualification.modelAnswerCount()).isEqualTo(15);
    assertThat(qualification.modelAnswerRate()).isEqualTo(15.0 / 27.0);
    assertThat(qualification.modelAnswerCorrectRate()).isEqualTo(1.0);
    assertThat(pureJava.performanceTier()).isEqualTo(RagPerformanceTier.USABLE);
    assertThat(pureJavaQualification.verdict())
        .isEqualTo(RagQualificationVerdict.FAILED_RELATIVE_GATE);
    assertThat(candidate.backendDiagnostics().optimization("rust-q8-0-batched-matmul"))
        .get()
        .satisfies(
            optimization ->
                assertThat(optimization.status()).isEqualTo(OptimizationStatus.ENABLED));
    assertThat(candidate.summary().p50PrefillTokensPerSecond())
        .isGreaterThan(pureJava.summary().p50PrefillTokensPerSecond() * 1.45);
    assertThat(candidate.summary().ttftMillis().p95())
        .isLessThan(pureJava.summary().ttftMillis().p95() * 0.69);
    assertThat(candidate.summary().endToEndMillis().p95())
        .isLessThan(pureJava.summary().endToEndMillis().p95() * 0.84);
    assertThat(candidate.summary().totalCpuMillis())
        .isLessThan(pureJava.summary().totalCpuMillis() * 0.81);
    assertThat(candidate.runs())
        .filteredOn(run -> run.grounding().decision().modelContributed())
        .hasSize(15)
        .allSatisfy(run -> assertThat(run.evaluation().correct()).isTrue());
    assertThat(pureJava.runs())
        .filteredOn(run -> run.grounding().decision().modelContributed())
        .hasSize(18)
        .allSatisfy(run -> assertThat(run.evaluation().correct()).isTrue());
    assertThat(qualification.comparisons())
        .filteredOn(comparison -> comparison.comparatorBackend().equals("llama.cpp"))
        .singleElement()
        .satisfies(
            comparison -> {
              assertThat(comparison.decodeThroughputRatio()).isBetween(0.69, 0.70);
              assertThat(comparison.endToEndLatencyRatio()).isBetween(1.29, 1.30);
              assertThat(comparison.qualified()).isTrue();
            });
    assertThat(qualification.comparisons())
        .filteredOn(comparison -> comparison.comparatorBackend().equals("ollama"))
        .singleElement()
        .satisfies(
            comparison -> {
              assertThat(comparison.decodeThroughputRatio()).isBetween(0.89, 0.90);
              assertThat(comparison.endToEndLatencyRatio()).isBetween(1.29, 1.30);
              assertThat(comparison.qualified()).isTrue();
            });
  }

  @Test
  void qwen25GeneralMarkerProfileQualifiesWithIdenticalBaselineOutputs() throws Exception {
    RagBenchmarkReport baseline =
        report(QWEN2_5_GENERAL_Q4_K_M_EVIDENCE, "models-rust-ffm-baseline.json");
    RagBenchmarkReport candidate = report(QWEN2_5_GENERAL_Q4_K_M_EVIDENCE, "models-rust-ffm.json");
    RagBenchmarkReport ollama = report(QWEN2_5_GENERAL_Q4_K_M_EVIDENCE, "ollama.json");
    RagBenchmarkReport llama = report(QWEN2_5_GENERAL_Q4_K_M_EVIDENCE, "llama.cpp.json");

    RagProductionQualification qualification =
        RagProductionQualificationPolicy.assess(candidate, List.of(llama, ollama));

    assertThat(candidate.artifactSha256())
        .isEqualTo("74a4da8c9fdbcd15bd1f6d01d621410d31c6fc00986f5eb687824e7b93d7a9db");
    assertThat(candidate.performanceTier()).isEqualTo(RagPerformanceTier.PRODUCTION_READY);
    assertThat(qualification.qualified()).isTrue();
    assertThat(qualification.qualifyingComparators()).containsExactly("llama.cpp", "ollama");
    assertThat(qualification.modelAnswerCount()).isEqualTo(12);
    assertThat(qualification.modelAnswerCorrectRate()).isEqualTo(1.0);
    assertThat(candidate.backendDiagnostics().optimization("rust-quantized-decode"))
        .get()
        .satisfies(
            optimization ->
                assertThat(optimization.status()).isEqualTo(OptimizationStatus.ENABLED));
    assertThat(baseline.backendDiagnostics().optimization("rust-quantized-decode"))
        .get()
        .satisfies(
            optimization ->
                assertThat(optimization.status()).isEqualTo(OptimizationStatus.DISABLED));
    assertThat(candidate.backendDiagnostics().optimizations())
        .filteredOn(optimization -> optimization.id().startsWith("modeljars.profile."))
        .singleElement()
        .satisfies(
            optimization -> {
              assertThat(optimization.status()).isEqualTo(OptimizationStatus.ENABLED);
              assertThat(optimization.settings())
                  .containsEntry("profile-id", "qwen2_5_0_5b_q4_k_m_epyc_milan_jdk25_rust_ffm");
            });
    assertThat(candidate.summary().p50DecodeTokensPerSecond())
        .isGreaterThan(baseline.summary().p50DecodeTokensPerSecond() * 2.9);
    assertThat(candidate.runs())
        .extracting(run -> run.grounding().rawText())
        .containsExactlyElementsOf(
            baseline.runs().stream().map(run -> run.grounding().rawText()).toList());
    assertThat(candidate.runs())
        .extracting(run -> run.grounding().decision())
        .containsExactlyElementsOf(
            baseline.runs().stream().map(run -> run.grounding().decision()).toList());
  }

  @Test
  void qwen25General15BMarkerProfileClearsTheOllamaLatencyGate() throws Exception {
    RagBenchmarkReport baseline =
        report(QWEN2_5_GENERAL_1_5B_Q4_K_M_EVIDENCE, "models-rust-ffm-baseline.json");
    RagBenchmarkReport candidate =
        report(QWEN2_5_GENERAL_1_5B_Q4_K_M_EVIDENCE, "models-rust-ffm.json");
    RagBenchmarkReport ollama = report(QWEN2_5_GENERAL_1_5B_Q4_K_M_EVIDENCE, "ollama.json");
    RagBenchmarkReport llama = report(QWEN2_5_GENERAL_1_5B_Q4_K_M_EVIDENCE, "llama.cpp.json");

    RagProductionQualification baselineQualification =
        RagProductionQualificationPolicy.assess(baseline, List.of(llama, ollama));
    RagProductionQualification qualification =
        RagProductionQualificationPolicy.assess(candidate, List.of(llama, ollama));

    assertThat(candidate.artifactSha256())
        .isEqualTo("6a1a2eb6d15622bf3c96857206351ba97e1af16c30d7a74ee38970e434e9407e");
    assertThat(baselineQualification.verdict())
        .isEqualTo(RagQualificationVerdict.FAILED_RELATIVE_GATE);
    assertThat(qualification.qualified()).isTrue();
    assertThat(qualification.qualifyingComparators()).containsExactly("llama.cpp", "ollama");
    assertThat(qualification.modelAnswerCount()).isEqualTo(9);
    assertThat(qualification.modelAnswerCorrectRate()).isEqualTo(1.0);
    assertThat(candidate.backendDiagnostics().optimizations())
        .filteredOn(optimization -> optimization.id().startsWith("modeljars.profile."))
        .singleElement()
        .satisfies(
            optimization -> {
              assertThat(optimization.status()).isEqualTo(OptimizationStatus.ENABLED);
              assertThat(optimization.settings())
                  .containsEntry("profile-id", "qwen2_5_1_5b_q4_k_m_epyc_milan_jdk25_rust_ffm");
            });
    assertThat(candidate.backendDiagnostics().optimization("batched-attention-scores"))
        .get()
        .satisfies(
            optimization ->
                assertThat(optimization.status()).isEqualTo(OptimizationStatus.ENABLED));
    assertThat(candidate.backendDiagnostics().optimization("batched-attention-values"))
        .get()
        .satisfies(
            optimization ->
                assertThat(optimization.status()).isEqualTo(OptimizationStatus.ENABLED));
    assertThat(candidate.summary().endToEndMillis().p95())
        .isLessThan(ollama.summary().endToEndMillis().p95() * 1.5);
    assertThat(candidate.runs())
        .extracting(run -> run.grounding().rawText())
        .containsExactlyElementsOf(
            baseline.runs().stream().map(run -> run.grounding().rawText()).toList());
    assertThat(candidate.runs())
        .extracting(run -> run.grounding().decision())
        .containsExactlyElementsOf(
            baseline.runs().stream().map(run -> run.grounding().decision()).toList());
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
