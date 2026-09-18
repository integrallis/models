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

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class AcceleratorProfileCliTest {

  @TempDir Path directory;

  @Test
  void parsesTheAcceleratorGateOptions() throws IOException {
    Path model = Files.writeString(directory.resolve("model.gguf"), "not a real model");

    AcceleratorProfileCli.Configuration configuration =
        AcceleratorProfileCli.parse(
            new String[] {
              "--model", model.toString(),
              "--prompt", "measure it",
              "--tokens", "16",
              "--warmups", "0",
              "--context", "512",
              "--batch", "8",
              "--decode", "false",
              "--eager", "false",
              "--require", "true",
              "--output", directory.resolve("gate.json").toString()
            });

    assertThat(configuration.model()).isEqualTo(model.toAbsolutePath().normalize());
    assertThat(configuration.prompt()).isEqualTo("measure it");
    assertThat(configuration.tokens()).isEqualTo(16);
    assertThat(configuration.warmups()).isZero();
    assertThat(configuration.contextLength()).isEqualTo(512);
    assertThat(configuration.batch()).isEqualTo(8);
    assertThat(configuration.decode()).isFalse();
    assertThat(configuration.eager()).isFalse();
    assertThat(configuration.require()).isTrue();
    assertThat(configuration.output()).hasFileName("gate.json");
  }

  @Test
  void defaultsToAnAcceleratedPrefillAndDecodeGate() throws IOException {
    Path model = Files.writeString(directory.resolve("model.gguf"), "not a real model");

    AcceleratorProfileCli.Configuration configuration =
        AcceleratorProfileCli.parse(new String[] {"--model", model.toString()});

    assertThat(configuration.decode()).isTrue();
    assertThat(configuration.eager()).isTrue();
    assertThat(configuration.require()).isFalse();
    assertThat(configuration.batch()).isEqualTo(32);
    assertThat(configuration.tokens()).isEqualTo(64);
    assertThat(configuration.prompt()).isNotBlank();
  }

  @Test
  void rejectsAModelPathThatIsNotAFile() {
    assertThatThrownBy(
            () ->
                AcceleratorProfileCli.parse(
                    new String[] {"--model", directory.resolve("missing.gguf").toString()}))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("regular file");
    assertThatThrownBy(() -> AcceleratorProfileCli.parse(new String[0]))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("--model");
  }

  @Test
  void rejectsNonBooleanFlagValues() throws IOException {
    Path model = Files.writeString(directory.resolve("model.gguf"), "not a real model");

    assertThatThrownBy(
            () ->
                AcceleratorProfileCli.parse(
                    new String[] {"--model", model.toString(), "--require", "yes"}))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("--require must be true or false");
  }
}
