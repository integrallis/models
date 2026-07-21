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
import java.util.List;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;
import org.modeljars.ModelJarDescriptor;
import org.modeljars.ModelJarRegistry;

class PureJavaModelSourceTest {

  @TempDir Path directory;

  @Test
  void resolvesAFileOrAnExactModelJarAlias() throws Exception {
    Path model = Files.write(directory.resolve("fixture.gguf"), new byte[] {1, 2, 3});
    ModelJarDescriptor descriptor = ModelJarTestFixtures.descriptor("fixture", model);
    ModelJarRegistry registry = ModelJarRegistry.of(List.of(descriptor));

    PureJavaModelSource pathSource = PureJavaModelSource.resolve(model.toString(), null, registry);
    PureJavaModelSource modelJarSource = PureJavaModelSource.resolve(null, "fixture", registry);

    assertThat(pathSource.identity()).isEqualTo(model.toString());
    assertThat(pathSource.artifact()).isEqualTo(model);
    assertThat(pathSource.descriptor()).isEmpty();
    assertThat(modelJarSource.identity()).isEqualTo("fixture");
    assertThat(modelJarSource.artifact()).isEqualTo(model);
    assertThat(modelJarSource.descriptor()).contains(descriptor);
  }

  @Test
  void rejectsAmbiguousMissingAndUnusableSources() throws Exception {
    Path model = Files.write(directory.resolve("fixture.gguf"), new byte[] {1});
    ModelJarDescriptor descriptor = ModelJarTestFixtures.descriptor("fixture", model);
    ModelJarRegistry registry = ModelJarRegistry.of(List.of(descriptor));

    assertThatThrownBy(() -> PureJavaModelSource.resolve(model.toString(), "fixture", registry))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("exactly one");
    assertThatThrownBy(() -> PureJavaModelSource.resolve(null, null, registry))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("exactly one");
    assertThatThrownBy(() -> PureJavaModelSource.resolve(null, "missing", registry))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("unknown ModelJar alias");
    assertThatThrownBy(
            () ->
                PureJavaModelSource.resolve(
                    null,
                    "fixture",
                    ModelJarRegistry.of(
                        List.of(
                            descriptor,
                            ModelJarTestFixtures.descriptor(
                                "fixture", directory.resolve("second.gguf"))))))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("ambiguous ModelJar alias");
    assertThatThrownBy(
            () ->
                PureJavaModelSource.resolve(
                    null,
                    "absent",
                    ModelJarRegistry.of(
                        List.of(
                            ModelJarTestFixtures.descriptor(
                                "absent", directory.resolve("absent.gguf"))))))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("artifact does not exist");
  }

  @Test
  void cannotRepresentANonexistentArtifact() {
    Path absent = directory.resolve("absent.gguf");

    assertThatThrownBy(() -> new PureJavaModelSource("absent", absent, java.util.Optional.empty()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("artifact does not exist");
  }
}
