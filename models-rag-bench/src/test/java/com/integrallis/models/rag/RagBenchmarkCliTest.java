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
import java.util.Properties;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.modeljars.ModelJarRegistry;
import org.modeljars.PropertiesModelJarRegistry;

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
              "--prompt-template", "chatml",
              "--case", "auto-glass-deadline,idempotency",
              "--max-tokens", "48",
              "--iterations", "2"
            });

    assertThat(configuration.framework()).isEqualTo("spring-ai");
    assertThat(configuration.artifact()).isEqualTo(model);
    assertThat(configuration.modelId()).isEqualTo("fixture-q4");
    assertThat(configuration.promptTemplate()).isEqualTo(RagPromptTemplate.CHATML);
    assertThat(configuration.caseIds()).containsExactly("auto-glass-deadline", "idempotency");
    assertThat(configuration.maxTokens()).isEqualTo(48);
    assertThat(configuration.iterations()).isEqualTo(2);
  }

  @Test
  void resolvesAModelJarForTheProfiledPureJavaPath() throws Exception {
    Path model = Files.writeString(temporaryDirectory.resolve("model.gguf"), "fixture");

    RagBenchmarkConfiguration configuration =
        RagBenchmarkCli.parse(
            new String[] {
              "--framework", "plain-java",
              "--backend", "pure-java",
              "--modeljar", "fixture_q4",
              "--prompt-template", "chatml"
            },
            registry(model));

    assertThat(configuration.model()).isEqualTo("fixture_q4");
    assertThat(configuration.modelId()).isEqualTo("fixture_q4");
    assertThat(configuration.artifact()).isEqualTo(model);
    assertThat(configuration.modelJarDescriptor())
        .hasValueSatisfying(descriptor -> assertThat(descriptor.alias()).isEqualTo("fixture_q4"));
  }

  @Test
  void rejectsAmbiguousPureJavaSources() throws Exception {
    Path model = Files.writeString(temporaryDirectory.resolve("model.gguf"), "fixture");

    assertThatThrownBy(
            () ->
                RagBenchmarkCli.parse(
                    new String[] {
                      "--framework", "plain-java",
                      "--backend", "pure-java",
                      "--model", model.toString(),
                      "--modeljar", "fixture_q4"
                    },
                    registry(model)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("exactly one");
  }

  @Test
  void rejectsModelJarsForNativeBackends() throws Exception {
    Path model = Files.writeString(temporaryDirectory.resolve("model.gguf"), "fixture");

    assertThatThrownBy(
            () ->
                RagBenchmarkCli.parse(
                    new String[] {
                      "--framework", "plain-java",
                      "--backend", "llama.cpp",
                      "--modeljar", "fixture_q4"
                    },
                    registry(model)))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("only by pure-java");
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

  private static ModelJarRegistry registry(Path model) {
    Properties properties = new Properties();
    properties.setProperty("model.fixture_q4.sourceId", "hf://example/fixture");
    properties.setProperty(
        "model.fixture_q4.markerCoordinate", "org.modeljars.local:fixture.q4_0:1.0.0");
    properties.setProperty("model.fixture_q4.modelVersion", "1.0.0");
    properties.setProperty("model.fixture_q4.variant", "q4_0");
    properties.setProperty("model.fixture_q4.format", "gguf");
    properties.setProperty("model.fixture_q4.architecture", "llama");
    properties.setProperty("model.fixture_q4.quantization", "Q4_0");
    properties.setProperty("model.fixture_q4.path", model.toString());
    properties.setProperty("model.fixture_q4.backend.pure-java", "true");
    return PropertiesModelJarRegistry.fromProperties(properties);
  }
}
