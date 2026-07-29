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

class PureJavaModelSourceTest {

  @TempDir Path directory;

  @Test
  void resolvesAnExplicitModelFile() throws Exception {
    Path model = Files.write(directory.resolve("fixture.gguf"), new byte[] {1, 2, 3});

    PureJavaModelSource source = PureJavaModelSource.resolve(model.toString());

    assertThat(source.identity()).isEqualTo(model.toString());
    assertThat(source.artifact()).isEqualTo(model);
  }

  @Test
  void rejectsMissingAndUnusableSources() {
    assertThatThrownBy(() -> PureJavaModelSource.resolve(null))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("require --model");
    assertThatThrownBy(() -> PureJavaModelSource.resolve(" "))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("require --model");
    assertThatThrownBy(
            () -> PureJavaModelSource.resolve(directory.resolve("absent.gguf").toString()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("artifact does not exist");
  }

  @Test
  void cannotRepresentANonexistentArtifact() {
    Path absent = directory.resolve("absent.gguf");

    assertThatThrownBy(() -> new PureJavaModelSource("absent", absent))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("artifact does not exist");
  }
}
