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
import static org.assertj.core.api.Assertions.within;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ActivatedPrefixSharingBenchmarkCliTest {
  private static final String REVISION = "a".repeat(40);

  @TempDir Path temporary;

  @Test
  void parsesPinnedArtifactsAndBoundedTrialControls() throws Exception {
    Path model = Files.writeString(temporary.resolve("model.gguf"), "model");
    Path adapter = Files.createDirectory(temporary.resolve("adapter"));

    ActivatedPrefixSharingBenchmarkCli.Configuration configuration =
        ActivatedPrefixSharingBenchmarkCli.parse(
            new String[] {
              "--model",
              model.toString(),
              "--adapter",
              adapter.toString(),
              "--models-revision",
              REVISION,
              "--warmups",
              "2",
              "--trials",
              "5",
              "--report",
              temporary.resolve("report.json").toString()
            });

    assertThat(configuration.model()).isEqualTo(model);
    assertThat(configuration.adapter()).isEqualTo(adapter);
    assertThat(configuration.modelsRevision()).isEqualTo(REVISION);
    assertThat(configuration.warmups()).isEqualTo(2);
    assertThat(configuration.trials()).isEqualTo(5);
  }

  @Test
  void rejectsMissingArtifactsAndUnboundedControls() throws Exception {
    Path model = Files.writeString(temporary.resolve("model.gguf"), "model");
    Path adapter = Files.createDirectory(temporary.resolve("adapter"));

    assertThatIllegalArgumentException()
        .isThrownBy(
            () ->
                ActivatedPrefixSharingBenchmarkCli.parse(
                    new String[] {
                      "--model",
                      model.toString(),
                      "--adapter",
                      adapter.toString(),
                      "--models-revision",
                      REVISION,
                      "--trials",
                      "0"
                    }))
        .withMessageContaining("--trials");
  }

  @Test
  void findsTheFirstWinningTierAndRequiresTwentyPercentAtFourThousandTokens() {
    List<ActivatedPrefixSharingBenchmarkCli.TierSummary> summaries =
        List.of(summary(256, 110, 100), summary(1_024, 90, 120), summary(4_096, 200, 300));

    ActivatedPrefixSharingBenchmarkCli.Verdict verdict =
        ActivatedPrefixSharingBenchmarkCli.verdict(summaries);

    assertThat(verdict.crossoverPrefixTokens()).hasValue(1_024);
    assertThat(verdict.fourKImprovement()).isCloseTo(1.0 / 3.0, within(0.000_001));
    assertThat(verdict.passed()).isTrue();
  }

  @Test
  void rejectsAWinningCrossoverThatMissesTheFourThousandTokenGate() {
    List<ActivatedPrefixSharingBenchmarkCli.TierSummary> summaries =
        List.of(summary(256, 90, 100), summary(1_024, 90, 100), summary(4_096, 85, 100));

    ActivatedPrefixSharingBenchmarkCli.Verdict verdict =
        ActivatedPrefixSharingBenchmarkCli.verdict(summaries);

    assertThat(verdict.crossoverPrefixTokens()).hasValue(256);
    assertThat(verdict.fourKImprovement()).isCloseTo(0.15, within(0.000_001));
    assertThat(verdict.passed()).isFalse();
  }

  @Test
  void rejectsTimingWinsThatDoNotReduceUniqueInferenceState() {
    List<ActivatedPrefixSharingBenchmarkCli.TierSummary> summaries =
        List.of(
            summary(256, 80, 100, 2_000, 2_000),
            summary(1_024, 70, 100, 2_000, 2_000),
            summary(4_096, 60, 100, 2_000, 2_000));

    assertThat(ActivatedPrefixSharingBenchmarkCli.verdict(summaries).passed()).isFalse();
  }

  @Test
  void rejectsTimingAndMemoryWinsWhenSharedAndRecomputedTokensDiffer() {
    List<ActivatedPrefixSharingBenchmarkCli.TierSummary> summaries =
        List.of(
            summary(256, 80, 100, 1_000, 2_000, true),
            summary(1_024, 70, 100, 1_000, 2_000, true),
            summary(4_096, 60, 100, 1_000, 2_000, false));

    assertThat(ActivatedPrefixSharingBenchmarkCli.verdict(summaries).passed()).isFalse();
  }

  @Test
  void distinguishesDifferentTokenSequencesEvenWhenDecodedTextCouldMatch() {
    assertThat(
            ActivatedPrefixSharingBenchmarkCli.tokenSequencesExact(
                List.of(List.of(151_644, 198), List.of(151_644, 198))))
        .isTrue();
    assertThat(
            ActivatedPrefixSharingBenchmarkCli.tokenSequencesExact(
                List.of(List.of(151_644, 198), List.of(151_645, 198))))
        .isFalse();
  }

  private static ActivatedPrefixSharingBenchmarkCli.TierSummary summary(
      int tokens, double sharedMillis, double recomputedMillis) {
    return summary(tokens, sharedMillis, recomputedMillis, 1_000, 2_000);
  }

  private static ActivatedPrefixSharingBenchmarkCli.TierSummary summary(
      int tokens,
      double sharedMillis,
      double recomputedMillis,
      long sharedBytes,
      long recomputedBytes) {
    return summary(tokens, sharedMillis, recomputedMillis, sharedBytes, recomputedBytes, true);
  }

  private static ActivatedPrefixSharingBenchmarkCli.TierSummary summary(
      int tokens,
      double sharedMillis,
      double recomputedMillis,
      long sharedBytes,
      long recomputedBytes,
      boolean tokenExactAcrossStrategies) {
    return new ActivatedPrefixSharingBenchmarkCli.TierSummary(
        tokens,
        sharedMillis,
        recomputedMillis,
        1.0 - sharedMillis / recomputedMillis,
        sharedBytes,
        recomputedBytes,
        true,
        true,
        tokenExactAcrossStrategies);
  }
}
