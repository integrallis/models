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
}
