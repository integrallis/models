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
              "--case", "auto-glass-deadline,idempotency",
              "--max-tokens", "48",
              "--iterations", "2"
            });

    assertThat(configuration.framework()).isEqualTo("spring-ai");
    assertThat(configuration.artifact()).isEqualTo(model);
    assertThat(configuration.modelId()).isEqualTo("fixture-q4");
    assertThat(configuration.caseIds()).containsExactly("auto-glass-deadline", "idempotency");
    assertThat(configuration.maxTokens()).isEqualTo(48);
    assertThat(configuration.iterations()).isEqualTo(2);
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
}
