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
package com.integrallis.models.backend.apple;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assumptions.assumeThat;

import java.nio.file.Files;
import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("integration")
class AppleFoundationModelsIntegrationTest {

  @Test
  void generatesTextWithRealAppleFoundationModels() {
    AppleFoundationModelsAvailability support = ApplePlatform.current().foundationModelsSupport();
    assumeThat(support.supported()).as(support.reason()).isTrue();

    String configuredLibrary = configuredNativeLibrary();
    assumeThat(configuredLibrary)
        .as(
            "%s or %s must point to libjavamodels_apple_foundation.dylib",
            NativeLibraryLocator.LIBRARY_PATH_PROPERTY, NativeLibraryLocator.LIBRARY_PATH_ENV)
        .isNotBlank();

    Path library = Path.of(configuredLibrary);
    assumeThat(Files.isRegularFile(library))
        .as("Apple Foundation Models bridge must exist at %s", library)
        .isTrue();

    try (AppleFoundationModelsClient client = AppleFoundationModels.create()) {
      AppleFoundationModelsAvailability availability = client.availability();
      assumeThat(availability.available()).as(availability.reason()).isTrue();

      AppleFoundationModelsResponse response =
          client.generate(
              AppleFoundationModelsRequest.builder("Reply with the single word hello.")
                  .instructions("Answer briefly.")
                  .maxOutputTokens(16)
                  .build());

      assertThat(response.text()).isNotBlank();
    }
  }

  private static String configuredNativeLibrary() {
    String configured = System.getProperty(NativeLibraryLocator.LIBRARY_PATH_PROPERTY);
    if (configured == null || configured.isBlank()) {
      configured = System.getenv(NativeLibraryLocator.LIBRARY_PATH_ENV);
    }
    return configured;
  }
}
