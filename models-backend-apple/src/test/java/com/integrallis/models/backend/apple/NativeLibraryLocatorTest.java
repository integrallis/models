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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.util.HexFormat;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

@Tag("unit")
class NativeLibraryLocatorTest {

  @TempDir Path temporaryDirectory;

  @Test
  void systemLocatorAcceptsAnIntegrityCheckedExplicitLibraryPath() throws Exception {
    String previousPath = System.getProperty(NativeLibraryLocator.LIBRARY_PATH_PROPERTY);
    String previousDigest = System.getProperty("models.apple.foundation.library.sha256");
    Path configured = temporaryDirectory.resolve("explicit.dylib");
    Files.writeString(configured, "verified native library", StandardCharsets.UTF_8);
    try {
      System.setProperty(NativeLibraryLocator.LIBRARY_PATH_PROPERTY, configured.toString());
      System.setProperty(
          "models.apple.foundation.library.sha256", sha256(Files.readAllBytes(configured)));

      assertThat(NativeLibraryLocator.system().locate())
          .contains(configured.toAbsolutePath().normalize());
    } finally {
      restoreProperty(NativeLibraryLocator.LIBRARY_PATH_PROPERTY, previousPath);
      restoreProperty("models.apple.foundation.library.sha256", previousDigest);
    }
  }

  @Test
  void configuredLocatorRejectsAnExplicitPathWithoutIntegrityMetadata() throws Exception {
    Path configured = temporaryDirectory.resolve("unchecked.dylib");
    Files.writeString(configured, "unchecked native library", StandardCharsets.UTF_8);

    assertThatThrownBy(
            () ->
                NativeLibraryLocator.resolveConfigured(
                    configured.toString(),
                    null,
                    null,
                    null,
                    null,
                    null,
                    NativeLibraryLocator.empty()))
        .isInstanceOf(SecurityException.class)
        .hasMessageContaining("SHA-256");
  }

  @Test
  void systemLocatorRejectsAnExplicitPathWithTheWrongDigest() throws Exception {
    String previousPath = System.getProperty(NativeLibraryLocator.LIBRARY_PATH_PROPERTY);
    String previousDigest = System.getProperty("models.apple.foundation.library.sha256");
    Path configured = temporaryDirectory.resolve("tampered.dylib");
    Files.writeString(configured, "tampered native library", StandardCharsets.UTF_8);
    try {
      System.setProperty(NativeLibraryLocator.LIBRARY_PATH_PROPERTY, configured.toString());
      System.setProperty("models.apple.foundation.library.sha256", "0".repeat(64));

      assertThatThrownBy(() -> NativeLibraryLocator.system().locate())
          .isInstanceOf(SecurityException.class)
          .hasMessageContaining("mismatch");
    } finally {
      restoreProperty(NativeLibraryLocator.LIBRARY_PATH_PROPERTY, previousPath);
      restoreProperty("models.apple.foundation.library.sha256", previousDigest);
    }
  }

  @Test
  void configuredLocatorRequiresAnExplicitOptInForUnverifiedDevelopmentLibraries()
      throws Exception {
    Path configured = temporaryDirectory.resolve("development.dylib");
    Files.writeString(configured, "development native library", StandardCharsets.UTF_8);

    assertThat(
            NativeLibraryLocator.resolveConfigured(
                configured.toString(),
                null,
                null,
                null,
                "true",
                null,
                NativeLibraryLocator.empty()))
        .contains(configured.toAbsolutePath().normalize());
  }

  @Test
  void configuredLocatorFallsBackToTheBundledBridge() {
    Path bundled = temporaryDirectory.resolve("bundled.dylib");

    assertThat(
            NativeLibraryLocator.resolveConfigured(
                " ", null, null, null, null, null, NativeLibraryLocator.fixed(bundled)))
        .contains(bundled);
  }

  private static String sha256(byte[] bytes) throws Exception {
    return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
  }

  private static void restoreProperty(String name, String previous) {
    if (previous == null) {
      System.clearProperty(name);
    } else {
      System.setProperty(name, previous);
    }
  }
}
