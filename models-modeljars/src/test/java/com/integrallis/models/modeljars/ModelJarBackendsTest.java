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
package com.integrallis.models.modeljars;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.net.URI;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Properties;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.modeljars.ModelDimensions;
import org.modeljars.ModelJarCoordinate;
import org.modeljars.ModelJarDescriptor;
import org.modeljars.ModelJarException;
import org.modeljars.ModelPerformanceProfileRegistry;
import org.modeljars.ModelVersion;

@Tag("unit")
class ModelJarBackendsTest {

  @Test
  void producesAnEmptyRegistryNeutralConfigurationWithoutAMatchingProfile() {
    Properties properties = new Properties();
    properties.setProperty("modeljars.performance.schemaVersion", "1");

    var configuration =
        ModelJarBackends.configuration(
            descriptor(true),
            "pure-java",
            ModelPerformanceProfileRegistry.fromProperties(properties),
            Map.of("os", "test"),
            List.of());

    assertThat(configuration.environment())
        .containsEntry("modeljar-alias", "nano")
        .containsEntry("modeljar-coordinate", "org.modeljars.test:nano:1.0.0");
    assertThat(configuration.recommendations()).isEmpty();
    assertThat(configuration.optimizations()).isEmpty();
  }

  @Test
  void rejectsADescriptorThatDoesNotSupportTheRequestedBackend() {
    assertThatThrownBy(() -> ModelJarBackends.loadPureJava(descriptor(false)))
        .isInstanceOf(ModelJarException.class)
        .hasMessageContaining("does not support pure-java");
  }

  private static ModelJarDescriptor descriptor(boolean supportsPureJava) {
    return new ModelJarDescriptor(
        "nano",
        "hf://example/nano",
        ModelJarCoordinate.parse("org.modeljars.test:nano:1.0.0"),
        ModelVersion.parse("1.0.0"),
        "q4_0",
        "gguf",
        "llama",
        "Q4_0",
        Optional.of(Path.of("missing.gguf")),
        Optional.empty(),
        Optional.of(URI.create("https://example.invalid/nano")),
        Optional.empty(),
        Optional.of("0123456789abcdef0123456789abcdef01234567"),
        Optional.of("aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"),
        Optional.of(1L),
        Optional.of("Apache-2.0"),
        Set.of("text-generation"),
        Set.of(),
        Map.of("pure-java", supportsPureJava),
        Optional.of("Nano"),
        Optional.of("test fixture"),
        Optional.of(URI.create("https://www.apache.org/licenses/LICENSE-2.0")),
        Set.of("test"),
        ModelDimensions.unknown());
  }
}
