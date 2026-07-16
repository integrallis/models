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

import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class InferenceBenchmarkCliTest {

  @TempDir Path directory;

  @Test
  void resolvesPureJavaDefaultsAndArtifactIdentity() throws Exception {
    Path model = Files.write(directory.resolve("fixture.gguf"), new byte[] {1, 2, 3});

    BenchmarkConfiguration configuration =
        InferenceBenchmarkCli.parse(
            new String[] {
              "--backend", "pure-java", "--model", model.toString(), "--model-id", "fixture"
            });

    assertThat(configuration.backend()).isEqualTo("pure-java");
    assertThat(configuration.artifact()).isEqualTo(model);
    assertThat(configuration.iterations()).isEqualTo(10);
    assertThat(configuration.warmups()).isEqualTo(2);
    assertThat(configuration.maxTokens()).isEqualTo(64);
    assertThat(configuration.speculativeOptions().enabled()).isFalse();
  }

  @Test
  void resolvesExplicitNgramSpeculationControlsForPureJava() throws Exception {
    Path model = Files.write(directory.resolve("fixture.gguf"), new byte[] {1, 2, 3});

    BenchmarkConfiguration configuration =
        InferenceBenchmarkCli.parse(
            new String[] {
              "--backend",
              "pure-java",
              "--model",
              model.toString(),
              "--speculation",
              "ngram",
              "--ngram-size",
              "6",
              "--draft-min",
              "3",
              "--draft-max",
              "7",
              "--speculation-window",
              "4",
              "--speculation-min-acceptance",
              "0.85",
              "--speculation-cooldown",
              "24"
            });

    assertThat(configuration.speculativeOptions().enabled()).isTrue();
    assertThat(configuration.speculativeOptions().ngramSize()).isEqualTo(6);
    assertThat(configuration.speculativeOptions().minimumDraftTokens()).isEqualTo(3);
    assertThat(configuration.speculativeOptions().maximumDraftTokens()).isEqualTo(7);
    assertThat(configuration.speculativeOptions().adaptationWindow()).isEqualTo(4);
    assertThat(configuration.speculativeOptions().minimumAcceptanceRate()).isEqualTo(0.85f);
    assertThat(configuration.speculativeOptions().cooldownTokens()).isEqualTo(24);
  }

  @Test
  void rejectsUnknownOptionsInsteadOfIgnoringTypos() {
    assertThatThrownBy(
            () ->
                InferenceBenchmarkCli.parse(
                    new String[] {
                      "--backend", "ollama", "--model", "fixture", "--interations", "10"
                    }))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("unknown option");
  }

  @Test
  void capturesControlledExternalBackendIdentityAndProcessSettings() throws Exception {
    Path artifact = Files.write(directory.resolve("fixture.gguf"), new byte[] {1, 2, 3});

    BenchmarkConfiguration configuration =
        InferenceBenchmarkCli.parse(
            new String[] {
              "--backend",
              "ollama",
              "--model",
              "fixture:gguf",
              "--artifact",
              artifact.toString(),
              "--backend-version",
              "0.12.1",
              "--threads",
              "8",
              "--pid",
              "4242",
              "--load-ms",
              "912.5"
            });

    assertThat(configuration.backendVersion()).isEqualTo("0.12.1");
    assertThat(configuration.threads()).isEqualTo(8);
    assertThat(configuration.backendPid()).isEqualTo(4242);
    assertThat(configuration.loadMillis()).isEqualTo(912.5);
  }

  @Test
  void rejectsAThreadCountPureJavaCannotEnforce() throws Exception {
    Path model = Files.write(directory.resolve("fixture.gguf"), new byte[] {1, 2, 3});
    int unsupportedThreadCount = Runtime.getRuntime().availableProcessors() + 1;

    assertThatThrownBy(
            () ->
                InferenceBenchmarkCli.parse(
                    new String[] {
                      "--backend",
                      "pure-java",
                      "--model",
                      model.toString(),
                      "--threads",
                      Integer.toString(unsupportedThreadCount)
                    }))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("pure-java uses the JVM processor allocation")
        .hasMessageContaining(Integer.toString(Runtime.getRuntime().availableProcessors()));
  }

  @Test
  void createsADeterministicCacheResistantPromptSeries() {
    String basePrompt = "fixed benchmark prompt";

    String first = InferenceBenchmarkCli.benchmarkPrompt(basePrompt, "measurement", 0);
    String repeated = InferenceBenchmarkCli.benchmarkPrompt(basePrompt, "measurement", 0);
    String second = InferenceBenchmarkCli.benchmarkPrompt(basePrompt, "measurement", 1);

    assertThat(first).isEqualTo(repeated).endsWith("\n" + basePrompt);
    assertThat(second).endsWith("\n" + basePrompt).isNotEqualTo(first);
    assertThat(first.substring(0, first.indexOf('\n'))).hasSize(16);
    assertThat(second.substring(0, second.indexOf('\n'))).hasSize(16);
  }
}
