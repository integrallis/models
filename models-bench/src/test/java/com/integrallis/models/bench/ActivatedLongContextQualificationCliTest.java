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
import static org.assertj.core.api.Assertions.assertThatIllegalArgumentException;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ActivatedLongContextQualificationCliTest {
  private static final String REVISION = "a".repeat(40);

  @TempDir Path temporary;

  @Test
  void parsesPinnedArtifactsAndOutput() throws Exception {
    Path model = Files.writeString(temporary.resolve("model.gguf"), "model");
    Path adapter = Files.createDirectory(temporary.resolve("adapter"));
    Path report = temporary.resolve("report.json");

    ActivatedLongContextQualificationCli.Configuration configuration =
        ActivatedLongContextQualificationCli.parse(
            new String[] {
              "--model",
              model.toString(),
              "--adapter",
              adapter.toString(),
              "--models-revision",
              REVISION,
              "--max-tokens",
              "40",
              "--report",
              report.toString()
            });

    assertThat(configuration.model()).isEqualTo(model);
    assertThat(configuration.adapter()).isEqualTo(adapter);
    assertThat(configuration.modelsRevision()).isEqualTo(REVISION);
    assertThat(configuration.maxTokens()).isEqualTo(40);
    assertThat(configuration.report()).isEqualTo(report);
  }

  @Test
  void rejectsAnUnpinnedRevisionAndOutOfRangeGeneration() throws Exception {
    Path model = Files.writeString(temporary.resolve("model.gguf"), "model");
    Path adapter = Files.createDirectory(temporary.resolve("adapter"));

    assertThatIllegalArgumentException()
        .isThrownBy(
            () ->
                ActivatedLongContextQualificationCli.parse(
                    new String[] {
                      "--model",
                      model.toString(),
                      "--adapter",
                      adapter.toString(),
                      "--models-revision",
                      "main",
                      "--max-tokens",
                      "0"
                    }))
        .withMessageContaining("models-revision");
  }

  @Test
  void loadsAFrozenEightCaseFourThousandTokenSuite() throws Exception {
    ActivatedLongContextQualificationCli.Suite suite =
        ActivatedLongContextQualificationCli.loadSuite();

    assertThat(suite.schemaVersion()).isEqualTo(1);
    assertThat(suite.workload()).isEqualTo("qwen3-activated-long-context-v1");
    assertThat(suite.targetPrefixTokens()).isEqualTo(4_096);
    assertThat(suite.cases()).hasSize(8);
    assertThat(suite.cases())
        .extracting(ActivatedLongContextQualificationCli.Case::id)
        .doesNotHaveDuplicates();
    assertThat(suite.cases())
        .extracting(ActivatedLongContextQualificationCli.Case::archiveCode)
        .doesNotHaveDuplicates();
  }

  @Test
  void passesOnlyWhenEveryNativeCorrectAnswerIsRetainedExactlyOverPhysicalSharing() {
    List<ActivatedLongContextQualificationCli.CaseResult> results =
        java.util.stream.IntStream.range(0, 8)
            .mapToObj(index -> result(index, true, true, true, true))
            .toList();

    ActivatedLongContextQualificationCli.Summary summary =
        ActivatedLongContextQualificationCli.summarize(results);

    assertThat(summary.nativeCorrect()).isEqualTo(8);
    assertThat(summary.retainedNativeCorrect()).isEqualTo(8);
    assertThat(summary.exactOutputMatches()).isEqualTo(8);
    assertThat(summary.qualified()).isTrue();
  }

  @Test
  void rejectsSemanticRetentionWithoutExactGenerationEquivalence() {
    List<ActivatedLongContextQualificationCli.CaseResult> results =
        java.util.stream.IntStream.range(0, 8)
            .mapToObj(index -> result(index, true, true, index != 7, true))
            .toList();

    assertThat(ActivatedLongContextQualificationCli.summarize(results).qualified()).isFalse();
  }

  @Test
  void rejectsAResultThatDidNotActuallyShareTheRequiredContextTier() {
    List<ActivatedLongContextQualificationCli.CaseResult> results =
        java.util.stream.IntStream.range(0, 8)
            .mapToObj(index -> result(index, true, true, true, true))
            .map(
                result ->
                    result.id().equals("case-7")
                        ? new ActivatedLongContextQualificationCli.CaseResult(
                            result.id(),
                            4_095,
                            result.nativeCorrect(),
                            result.retainedCorrect(),
                            result.exactOutputMatch(),
                            result.toolCallCorrect(),
                            result.physicallyShared(),
                            result.nativeOutput(),
                            result.sharedOutput(),
                            result.toolOutput(),
                            result.nativeMillis(),
                            result.sharedMillis(),
                            result.toolMillis())
                        : result)
            .toList();

    assertThat(ActivatedLongContextQualificationCli.summarize(results).qualified()).isFalse();
  }

  @Test
  void rejectsAWeakNativeBaselineOrAContextLengthToolRegression() {
    List<ActivatedLongContextQualificationCli.CaseResult> weakBaseline =
        java.util.stream.IntStream.range(0, 8)
            .mapToObj(index -> result(index, index < 5, index < 5, true, true))
            .toList();
    List<ActivatedLongContextQualificationCli.CaseResult> toolRegression =
        java.util.stream.IntStream.range(0, 8)
            .mapToObj(index -> result(index, true, true, true, index != 7))
            .toList();

    assertThat(ActivatedLongContextQualificationCli.summarize(weakBaseline).qualified()).isFalse();
    assertThat(ActivatedLongContextQualificationCli.summarize(toolRegression).qualified())
        .isFalse();
  }

  private static ActivatedLongContextQualificationCli.CaseResult result(
      int index,
      boolean nativeCorrect,
      boolean retainedCorrect,
      boolean exact,
      boolean toolCorrect) {
    return new ActivatedLongContextQualificationCli.CaseResult(
        "case-" + index,
        4_096,
        nativeCorrect,
        retainedCorrect,
        exact,
        toolCorrect,
        true,
        "native",
        "shared",
        "tool",
        1,
        1,
        1);
  }
}
