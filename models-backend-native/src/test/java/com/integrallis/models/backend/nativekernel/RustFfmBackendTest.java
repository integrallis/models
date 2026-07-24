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
package com.integrallis.models.backend.nativekernel;

import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.file.Path;
import java.util.Properties;
import org.junit.jupiter.api.Test;
import org.modeljars.ModelJarDescriptor;
import org.modeljars.ModelJarException;
import org.modeljars.PropertiesModelJarRegistry;

class RustFfmBackendTest {

  @Test
  void rejectsAModelJarWithoutRustSupportBeforeOpeningTheLibrary() {
    ModelJarDescriptor descriptor = descriptor(false);

    assertThatThrownBy(() -> RustFfmBackend.load(descriptor, Path.of("missing-native-library")))
        .isInstanceOf(ModelJarException.class)
        .hasMessageContaining("does not support rust-ffm")
        .hasMessageContaining(descriptor.alias());
  }

  private static ModelJarDescriptor descriptor(boolean supportsRust) {
    Properties properties = new Properties();
    properties.setProperty("model.fixture.sourceId", "hf://example/fixture");
    properties.setProperty(
        "model.fixture.markerCoordinate", "org.modeljars.local:fixture.q8_0:1.0.0");
    properties.setProperty("model.fixture.modelVersion", "1.0.0");
    properties.setProperty("model.fixture.variant", "q8_0");
    properties.setProperty("model.fixture.format", "gguf");
    properties.setProperty("model.fixture.architecture", "llama");
    properties.setProperty("model.fixture.quantization", "Q8_0");
    properties.setProperty("model.fixture.path", "missing-model.gguf");
    properties.setProperty("model.fixture.backend.rust-ffm", Boolean.toString(supportsRust));
    return PropertiesModelJarRegistry.fromProperties(properties).descriptors().getFirst();
  }
}
