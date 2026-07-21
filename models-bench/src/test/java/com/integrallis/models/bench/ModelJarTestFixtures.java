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

import java.net.URI;
import java.nio.file.Path;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.modeljars.ModelDimensions;
import org.modeljars.ModelJarCoordinate;
import org.modeljars.ModelJarDescriptor;
import org.modeljars.ModelVersion;

final class ModelJarTestFixtures {

  private ModelJarTestFixtures() {}

  static ModelJarDescriptor descriptor(String alias, Path localPath) {
    return new ModelJarDescriptor(
        alias,
        "hf://example/fixture",
        ModelJarCoordinate.parse("org.modeljars.huggingface:example.fixture.q4_0:1.0.0-q4_0.1"),
        ModelVersion.parse("1.0.0"),
        "q4_0",
        "gguf",
        "llama",
        "Q4_0",
        Optional.of(localPath),
        Optional.empty(),
        Optional.of(URI.create("https://huggingface.co/example/fixture")),
        Optional.empty(),
        Optional.of("main"),
        Optional.of("0".repeat(64)),
        Optional.of(3L),
        Optional.of("Apache-2.0"),
        Set.of("text-generation"),
        Set.of(),
        Map.of("pure-java", true),
        Optional.of("Fixture"),
        Optional.empty(),
        Optional.empty(),
        Set.of(),
        ModelDimensions.unknown());
  }
}
