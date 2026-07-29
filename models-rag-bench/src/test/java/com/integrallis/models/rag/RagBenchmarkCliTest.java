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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class RagBenchmarkCliTest {

  @TempDir Path temporaryDirectory;

  @Test
  void parsesAReproduciblePureJavaRun() throws Exception {
    Path model = Files.writeString(temporaryDirectory.resolve("model.gguf"), "fixture");

    RagBenchmarkConfiguration configuration =
        RagBenchmarkCli.parse(
            new String[] {
              "--framework", "spring-ai",
              "--backend", "pure-java",
              "--model", model.toString(),
              "--model-id", "fixture-q4",
              "--workload", "coding",
              "--prompt-template", "chatml",
              "--case", "auto-glass-deadline,idempotency",
              "--temperature", "0.7",
              "--top-p", "0.95",
              "--sampling-top-k", "40",
              "--seed", "1729",
              "--repetition-penalty", "1.05",
              "--stop-sequence", "\\n\\n",
              "--max-tokens", "48",
              "--iterations", "2"
            });

    assertThat(configuration.framework()).isEqualTo("spring-ai");
    assertThat(configuration.artifact()).isEqualTo(model);
    assertThat(configuration.modelId()).isEqualTo("fixture-q4");
    assertThat(configuration.workload()).isEqualTo(RagWorkload.CODING);
    assertThat(configuration.promptTemplate()).isEqualTo(RagPromptTemplate.CHATML);
    assertThat(configuration.caseIds()).containsExactly("auto-glass-deadline", "idempotency");
    assertThat(configuration.sampling())
        .isEqualTo(new RagSamplingProfile(0.7, 0.95, 40, 1729L, 1.05, List.of("\n\n")));
    assertThat(configuration.maxTokens()).isEqualTo(48);
    assertThat(configuration.iterations()).isEqualTo(2);
  }

  @Test
  void rejectsInvalidSamplingControlsBeforeLoadingTheBackend() {
    assertThatThrownBy(
            () ->
                RagBenchmarkCli.parse(
                    new String[] {
                      "--framework", "plain-java",
                      "--backend", "ollama",
                      "--model", "qwen",
                      "--temperature", "-0.1"
                    }))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("--temperature");
  }

  @Test
  void parsesAReproducibleRustFfmRun() throws Exception {
    Path model = Files.writeString(temporaryDirectory.resolve("model-q8.gguf"), "fixture");

    RagBenchmarkConfiguration configuration =
        RagBenchmarkCli.parse(
            new String[] {
              "--framework", "plain-java",
              "--backend", "rust-ffm",
              "--model", model.toString(),
              "--model-id", "fixture-q8",
              "--prompt-template", "chatml"
            });

    assertThat(configuration.backend()).isEqualTo("rust-ffm");
    assertThat(configuration.model()).isEqualTo(model.toString());
    assertThat(configuration.artifact()).isEqualTo(model);
    assertThat(configuration.modelId()).isEqualTo("fixture-q8");
    assertThat(configuration.endpoint()).isNull();
    assertThat(configuration.promptTemplate()).isEqualTo(RagPromptTemplate.CHATML);
  }

  @Test
  void rejectsAnUnknownFrameworkBeforeLoadingAModel() {
    assertThatThrownBy(
            () ->
                RagBenchmarkCli.parse(
                    new String[] {
                      "--framework", "invented", "--backend", "ollama", "--model", "qwen"
                    }))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("framework");
  }

  @Test
  void rejectsAnUnknownWorkloadBeforeRunningInference() {
    assertThatThrownBy(
            () ->
                RagBenchmarkCli.parse(
                    new String[] {
                      "--framework", "plain-java",
                      "--backend", "ollama",
                      "--model", "qwen",
                      "--workload", "invented"
                    }))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("workload");
  }

  @Test
  void parsesAHostedOpenAiRunWithTheProviderEndpoint() {
    RagBenchmarkConfiguration configuration =
        RagBenchmarkCli.parse(
            new String[] {
              "--framework", "plain-java",
              "--backend", "openai",
              "--model", "gpt-5.4-nano-2026-03-17"
            });

    assertThat(configuration.endpoint())
        .isEqualTo(java.net.URI.create("https://api.openai.com/v1"));
    assertThat(configuration.artifact()).isNull();
    assertThat(configuration.promptTemplate()).isEqualTo(RagPromptTemplate.RAW);
  }

  @Test
  void rejectsArtifactsForHostedProviders() {
    assertThatThrownBy(
            () ->
                RagBenchmarkCli.parse(
                    new String[] {
                      "--framework", "plain-java",
                      "--backend", "anthropic",
                      "--model", "claude-haiku-4-5-20251001",
                      "--artifact", temporaryDirectory.resolve("model.gguf").toString()
                    }))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("--artifact");
  }
}
