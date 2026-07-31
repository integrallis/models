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

import java.net.URLClassLoader;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.attribute.PosixFilePermission;
import java.security.MessageDigest;
import java.util.EnumSet;
import java.util.HexFormat;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class BundledAppleFoundationLibraryTest {
  private static final byte[] LIBRARY =
      "test-apple-foundation-bridge".getBytes(StandardCharsets.UTF_8);

  @TempDir Path temporaryDirectory;

  @Test
  void extractsAndVerifiesTheBundledBridge() throws Exception {
    Path resources = temporaryDirectory.resolve("resources");
    writeBundle(resources, sha256(LIBRARY));

    try (var loader = new URLClassLoader(new java.net.URL[] {resources.toUri().toURL()}, null)) {
      Path extracted =
          BundledAppleFoundationLibrary.resolve(loader, temporaryDirectory.resolve("cache"))
              .orElseThrow();

      assertThat(Files.readAllBytes(extracted)).isEqualTo(LIBRARY);
      assertThat(
              BundledAppleFoundationLibrary.resolve(loader, temporaryDirectory.resolve("cache"))
                  .orElseThrow())
          .isEqualTo(extracted);
    }
  }

  @Test
  void rejectsABridgeWhoseDigestDoesNotMatchMetadata() throws Exception {
    Path resources = temporaryDirectory.resolve("tampered");
    writeBundle(resources, "0".repeat(64));

    try (var loader = new URLClassLoader(new java.net.URL[] {resources.toUri().toURL()}, null)) {
      assertThatThrownBy(
              () ->
                  BundledAppleFoundationLibrary.resolve(
                      loader, temporaryDirectory.resolve("cache")))
          .isInstanceOf(SecurityException.class)
          .hasMessageContaining("SHA-256 mismatch");
    }
  }

  @Test
  void rejectsMalformedOrIncompatibleMetadata() throws Exception {
    Path invalidDigest = temporaryDirectory.resolve("invalid-digest");
    writeBundle(invalidDigest, "not-a-digest");
    assertBundleFailure(invalidDigest, "invalid SHA-256");

    Path incompatibleAbi = temporaryDirectory.resolve("incompatible-abi");
    Path abiDirectory = writeBundle(incompatibleAbi, sha256(LIBRARY));
    replaceMetadata(abiDirectory, "abi=2", "abi=1");
    assertBundleFailure(incompatibleAbi, "abi must be 2");

    Path missingDigest = temporaryDirectory.resolve("missing-digest");
    Path digestDirectory = writeBundle(missingDigest, sha256(LIBRARY));
    replaceMetadata(digestDirectory, "sha256=" + sha256(LIBRARY), "");
    assertBundleFailure(missingDigest, "missing sha256");
  }

  @Test
  void rejectsMetadataWithoutTheBridgeLibrary() throws Exception {
    Path resources = temporaryDirectory.resolve("missing-library");
    Path directory = writeBundle(resources, sha256(LIBRARY));
    Files.delete(directory.resolve(BundledAppleFoundationLibrary.LIBRARY_FILE_NAME));

    assertBundleFailure(resources, "metadata exists without");
  }

  @Test
  void rejectsDuplicateBridgeResources() throws Exception {
    Path first = temporaryDirectory.resolve("first");
    Path second = temporaryDirectory.resolve("second");
    writeBundle(first, sha256(LIBRARY));
    writeBundle(second, sha256(LIBRARY));

    try (var loader =
        new URLClassLoader(
            new java.net.URL[] {first.toUri().toURL(), second.toUri().toURL()}, null)) {
      assertThatThrownBy(
              () ->
                  BundledAppleFoundationLibrary.resolve(
                      loader, temporaryDirectory.resolve("cache")))
          .isInstanceOf(IllegalStateException.class)
          .hasMessageContaining("Multiple Apple Foundation Models bridges");
    }
  }

  @Test
  void rejectsTamperingInTheExtractedCache() throws Exception {
    Path resources = temporaryDirectory.resolve("cache-tampering");
    writeBundle(resources, sha256(LIBRARY));

    try (var loader = new URLClassLoader(new java.net.URL[] {resources.toUri().toURL()}, null)) {
      Path cache = temporaryDirectory.resolve("cache");
      Path extracted = BundledAppleFoundationLibrary.resolve(loader, cache).orElseThrow();
      Files.writeString(extracted, "tampered", StandardCharsets.UTF_8);

      assertThatThrownBy(() -> BundledAppleFoundationLibrary.resolve(loader, cache))
          .isInstanceOf(SecurityException.class)
          .hasMessageContaining("Cached Apple Foundation Models bridge SHA-256 mismatch");
    }
  }

  @Test
  void rejectsAGroupOrWorldWritableCacheRoot() throws Exception {
    Path resources = temporaryDirectory.resolve("unsafe-cache-resources");
    writeBundle(resources, sha256(LIBRARY));
    Path cache = temporaryDirectory.resolve("unsafe-cache");
    Files.createDirectories(cache);
    try {
      Files.setPosixFilePermissions(
          cache,
          EnumSet.of(
              PosixFilePermission.OWNER_READ,
              PosixFilePermission.OWNER_WRITE,
              PosixFilePermission.OWNER_EXECUTE,
              PosixFilePermission.GROUP_WRITE,
              PosixFilePermission.OTHERS_WRITE));
    } catch (UnsupportedOperationException failure) {
      return;
    }

    try (var loader = new URLClassLoader(new java.net.URL[] {resources.toUri().toURL()}, null)) {
      assertThatThrownBy(() -> BundledAppleFoundationLibrary.resolve(loader, cache))
          .isInstanceOf(SecurityException.class)
          .hasMessageContaining("group or world writable");
    }
  }

  @Test
  void returnsEmptyWhenNoBundledBridgeIsPresent() throws Exception {
    try (var loader = new URLClassLoader(new java.net.URL[0], null)) {
      assertThat(BundledAppleFoundationLibrary.resolve(loader, temporaryDirectory.resolve("cache")))
          .isEmpty();
    }
  }

  @Test
  void configuredResolutionUsesTheConfiguredCacheDirectory() throws Exception {
    Path resources = temporaryDirectory.resolve("configured-cache-resources");
    writeBundle(resources, sha256(LIBRARY));
    Path configuredCache = temporaryDirectory.resolve("configured-cache");
    String previous = System.getProperty(BundledAppleFoundationLibrary.CACHE_DIRECTORY_PROPERTY);
    try (var loader = new URLClassLoader(new java.net.URL[] {resources.toUri().toURL()}, null)) {
      System.setProperty(
          BundledAppleFoundationLibrary.CACHE_DIRECTORY_PROPERTY, configuredCache.toString());

      assertThat(BundledAppleFoundationLibrary.resolve(loader).orElseThrow())
          .startsWith(configuredCache.toAbsolutePath().normalize());
    } finally {
      restoreProperty(BundledAppleFoundationLibrary.CACHE_DIRECTORY_PROPERTY, previous);
    }
  }

  private void assertBundleFailure(Path resources, String message) throws Exception {
    try (var loader = new URLClassLoader(new java.net.URL[] {resources.toUri().toURL()}, null)) {
      assertThatThrownBy(
              () ->
                  BundledAppleFoundationLibrary.resolve(
                      loader, temporaryDirectory.resolve("cache")))
          .isInstanceOf(IllegalStateException.class)
          .hasMessageContaining(message);
    }
  }

  private static Path writeBundle(Path resources, String digest) throws Exception {
    Path directory = resources.resolve("META-INF/models/apple-foundation/macos-aarch64");
    Files.createDirectories(directory);
    Files.write(directory.resolve(BundledAppleFoundationLibrary.LIBRARY_FILE_NAME), LIBRARY);
    Files.writeString(
        directory.resolve(BundledAppleFoundationLibrary.METADATA_FILE_NAME),
        """
        abi=2
        platform=macos-aarch64
        library=libjavamodels_apple_foundation.dylib
        sha256=%s
        """
            .formatted(digest),
        StandardCharsets.UTF_8);
    return directory;
  }

  private static void replaceMetadata(Path directory, String target, String replacement)
      throws Exception {
    Path metadata = directory.resolve(BundledAppleFoundationLibrary.METADATA_FILE_NAME);
    Files.writeString(
        metadata,
        Files.readString(metadata, StandardCharsets.UTF_8).replace(target, replacement),
        StandardCharsets.UTF_8);
  }

  private static void restoreProperty(String name, String previous) {
    if (previous == null) {
      System.clearProperty(name);
    } else {
      System.setProperty(name, previous);
    }
  }

  private static String sha256(byte[] bytes) throws Exception {
    return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
  }
}
