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

import java.nio.file.Path;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class NativeLibraryLocatorTest {

  @TempDir Path temporaryDirectory;

  @Test
  void systemLocatorPrefersAnExplicitLibraryPath() {
    String previous = System.getProperty(NativeLibraryLocator.LIBRARY_PATH_PROPERTY);
    Path configured = temporaryDirectory.resolve("explicit.dylib");
    try {
      System.setProperty(NativeLibraryLocator.LIBRARY_PATH_PROPERTY, configured.toString());

      assertThat(NativeLibraryLocator.system().locate()).contains(configured);
    } finally {
      restoreProperty(NativeLibraryLocator.LIBRARY_PATH_PROPERTY, previous);
    }
  }

  @Test
  void configuredLocatorUsesTheEnvironmentBeforeTheBundledBridge() {
    Path environment = temporaryDirectory.resolve("environment.dylib");
    Path bundled = temporaryDirectory.resolve("bundled.dylib");

    assertThat(
            NativeLibraryLocator.resolveConfigured(
                null, environment.toString(), NativeLibraryLocator.fixed(bundled)))
        .contains(environment);
  }

  @Test
  void configuredLocatorFallsBackToTheBundledBridge() {
    Path bundled = temporaryDirectory.resolve("bundled.dylib");

    assertThat(
            NativeLibraryLocator.resolveConfigured(" ", null, NativeLibraryLocator.fixed(bundled)))
        .contains(bundled);
  }

  private static void restoreProperty(String name, String previous) {
    if (previous == null) {
      System.clearProperty(name);
    } else {
      System.setProperty(name, previous);
    }
  }
}
