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

  /**
   * The output cap defaults to room for a grounded answer, which 64 was not.
   *
   * <p>Pinned because the value is not cosmetic: at 64 the 2026-09-29 campaign truncated most of
   * its attempts, and a truncated answer cannot carry the citation the grounding policy screens
   * for, so it was replaced by an extractive one and scored exactly like a model with nothing to
   * say. The campaign never chose 64 -- it inherited this default.
   */
  @Test
  void theOutputTokenCapDefaultsToRoomForAGroundedAnswer() throws Exception {
    Path model = Files.writeString(temporaryDirectory.resolve("default-cap.gguf"), "fixture");

    RagBenchmarkConfiguration configuration =
        RagBenchmarkCli.parse(
            new String[] {
              "--framework", "plain-java",
              "--backend", "pure-java",
              "--model", model.toString(),
              "--model-id", "fixture",
              "--workload", "general"
            });

    assertThat(configuration.maxTokens()).isEqualTo(256);
  }

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
  void acceptsAHuggingFaceDirectoryAndBindsEvidenceToItsPrimaryWeights() throws Exception {
    Path modelDirectory = Files.createDirectories(temporaryDirectory.resolve("qwen-hf"));
    Path weights = Files.write(modelDirectory.resolve("model.safetensors"), new byte[] {1, 2, 3});
    Files.writeString(modelDirectory.resolve("config.json"), "{}");

    RagBenchmarkConfiguration configuration =
        RagBenchmarkCli.parse(
            new String[] {
              "--framework", "plain-java",
              "--backend", "pure-java",
              "--model", modelDirectory.toString()
            });

    assertThat(configuration.artifact()).isEqualTo(modelDirectory);
    assertThat(RagBenchmarkCli.artifactIdentity(configuration.artifact())).isEqualTo(weights);
  }

  @Test
  void aShardedHuggingFaceDirectoryIsIdentifiedByItsIndex() throws Exception {
    // gpt-oss-20b ships model-0000x-of-00002.safetensors plus an index and no single
    // model.safetensors. SafetensorsBundle already loads either shape; rejecting the sharded one
    // here was the only thing keeping such a model out of a qualification run.
    Path directory = Files.createTempDirectory("sharded-hf");
    Files.writeString(
        directory.resolve("model.safetensors.index.json"),
        "{\"weight_map\": {\"a\": \"model-00001-of-00002.safetensors\","
            + " \"b\": \"model-00002-of-00002.safetensors\"}}");
    Files.write(directory.resolve("model-00001-of-00002.safetensors"), new byte[] {1, 2, 3});
    Files.write(directory.resolve("model-00002-of-00002.safetensors"), new byte[] {4, 5});

    assertThat(RagBenchmarkCli.artifactIdentity(directory))
        .isEqualTo(directory.resolve("model.safetensors.index.json"));
  }

  @Test
  void aSingleFileBundleStillWinsOverAnIndex() throws Exception {
    Path directory = Files.createTempDirectory("single-hf");
    Files.write(directory.resolve("model.safetensors"), new byte[] {9});
    Files.writeString(directory.resolve("model.safetensors.index.json"), "{}");

    assertThat(RagBenchmarkCli.artifactIdentity(directory))
        .describedAs("a directory with both must use the single file, as it did before")
        .isEqualTo(directory.resolve("model.safetensors"));
  }

  @Test
  void aDirectoryWithNeitherShapeSaysSo() throws Exception {
    Path directory = Files.createTempDirectory("empty-hf");

    assertThatThrownBy(() -> RagBenchmarkCli.artifactIdentity(directory))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("model.safetensors.index.json");
  }

  @Test
  void rejectsAHuggingFaceDirectoryWithoutPrimaryWeights() throws Exception {
    Path modelDirectory = Files.createDirectories(temporaryDirectory.resolve("incomplete-hf"));
    Files.writeString(modelDirectory.resolve("config.json"), "{}");

    assertThatThrownBy(
            () ->
                RagBenchmarkCli.parse(
                    new String[] {
                      "--framework", "plain-java",
                      "--backend", "pure-java",
                      "--model", modelDirectory.toString()
                    }))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("model.safetensors");
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
