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

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.net.URL;
import java.net.URLConnection;
import java.nio.charset.StandardCharsets;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.FileAlreadyExistsException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.attribute.PosixFilePermission;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.EnumSet;
import java.util.HexFormat;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Properties;
import java.util.Set;

/** Extracts the integrity-checked Apple Foundation Models bridge bundled in the backend JAR. */
final class BundledAppleFoundationLibrary {
  static final String METADATA_FILE_NAME = "native.properties";
  static final String LIBRARY_FILE_NAME = "libjavamodels_apple_foundation.dylib";
  static final String CACHE_DIRECTORY_PROPERTY = "models.apple.foundation.cache";

  private static final int ABI_VERSION = 1;
  private static final String PLATFORM = "macos-aarch64";
  private static final String RESOURCE_DIRECTORY =
      "META-INF/models/apple-foundation/" + PLATFORM + "/";
  private static final Set<PosixFilePermission> OWNER_PERMISSIONS =
      EnumSet.of(
          PosixFilePermission.OWNER_READ,
          PosixFilePermission.OWNER_WRITE,
          PosixFilePermission.OWNER_EXECUTE);

  private BundledAppleFoundationLibrary() {}

  static Optional<Path> resolve() {
    String configuredCache = System.getProperty(CACHE_DIRECTORY_PROPERTY);
    Path cacheRoot =
        configuredCache == null || configuredCache.isBlank()
            ? Path.of(System.getProperty("user.home"), ".models", "apple-foundation")
            : Path.of(configuredCache);
    return resolve(AppleFoundationModels.class.getClassLoader(), cacheRoot);
  }

  static Optional<Path> resolve(ClassLoader classLoader, Path cacheRoot) {
    Objects.requireNonNull(classLoader, "classLoader");
    Objects.requireNonNull(cacheRoot, "cacheRoot");

    Optional<URL> metadataUrl =
        uniqueResource(classLoader, RESOURCE_DIRECTORY + METADATA_FILE_NAME);
    if (metadataUrl.isEmpty()) {
      return Optional.empty();
    }

    Properties metadata = loadMetadata(metadataUrl.orElseThrow());
    requireMetadata(metadata, "abi", Integer.toString(ABI_VERSION));
    requireMetadata(metadata, "platform", PLATFORM);
    requireMetadata(metadata, "library", LIBRARY_FILE_NAME);
    String expectedDigest = required(metadata, "sha256");
    if (!expectedDigest.matches("[0-9a-f]{64}")) {
      throw new IllegalStateException("Apple bridge metadata contains an invalid SHA-256");
    }

    URL libraryUrl =
        uniqueResource(classLoader, RESOURCE_DIRECTORY + LIBRARY_FILE_NAME)
            .orElseThrow(
                () ->
                    new IllegalStateException(
                        "Apple bridge metadata exists without " + LIBRARY_FILE_NAME));
    byte[] libraryBytes = readBytes(libraryUrl);
    String actualDigest = sha256(libraryBytes);
    if (!MessageDigest.isEqual(
        expectedDigest.getBytes(StandardCharsets.US_ASCII),
        actualDigest.getBytes(StandardCharsets.US_ASCII))) {
      throw new SecurityException(
          "Apple Foundation Models bridge SHA-256 mismatch: expected "
              + expectedDigest
              + " but found "
              + actualDigest);
    }

    Path target =
        cacheRoot
            .toAbsolutePath()
            .normalize()
            .resolve("abi-" + ABI_VERSION)
            .resolve(PLATFORM)
            .resolve(expectedDigest)
            .resolve(LIBRARY_FILE_NAME);
    extractOnce(target, libraryBytes, expectedDigest);
    return Optional.of(target);
  }

  private static Optional<URL> uniqueResource(ClassLoader classLoader, String resourceName) {
    try {
      List<URL> resources = new ArrayList<>(2);
      var matches = classLoader.getResources(resourceName);
      while (matches.hasMoreElements()) {
        resources.add(matches.nextElement());
      }
      if (resources.size() > 1) {
        throw new IllegalStateException(
            "Multiple Apple Foundation Models bridges provide " + resourceName + ": " + resources);
      }
      return resources.stream().findFirst();
    } catch (IOException failure) {
      throw new IllegalStateException(
          "Unable to locate Apple bridge resource " + resourceName, failure);
    }
  }

  private static Properties loadMetadata(URL metadataUrl) {
    Properties metadata = new Properties();
    try (InputStream input = openUncached(metadataUrl);
        var reader = new InputStreamReader(input, StandardCharsets.UTF_8)) {
      metadata.load(reader);
      return metadata;
    } catch (IOException failure) {
      throw new IllegalStateException("Unable to read Apple bridge metadata", failure);
    }
  }

  private static byte[] readBytes(URL resource) {
    try (InputStream input = openUncached(resource)) {
      return input.readAllBytes();
    } catch (IOException failure) {
      throw new IllegalStateException("Unable to read bundled Apple bridge", failure);
    }
  }

  private static InputStream openUncached(URL resource) throws IOException {
    URLConnection connection = resource.openConnection();
    connection.setUseCaches(false);
    return connection.getInputStream();
  }

  private static void requireMetadata(Properties metadata, String name, String expected) {
    String actual = required(metadata, name);
    if (!expected.equals(actual)) {
      throw new IllegalStateException(
          "Apple bridge metadata " + name + " must be " + expected + " but was " + actual);
    }
  }

  private static String required(Properties metadata, String name) {
    String value = metadata.getProperty(name);
    if (value == null || value.isBlank()) {
      throw new IllegalStateException("Apple bridge metadata is missing " + name);
    }
    return value.strip();
  }

  private static void extractOnce(Path target, byte[] bytes, String expectedDigest) {
    try {
      if (Files.isRegularFile(target)) {
        verifyDigest(target, expectedDigest);
        return;
      }
      Path parent = Objects.requireNonNull(target.getParent(), "target parent");
      Files.createDirectories(parent);
      setOwnerPermissions(parent);
      Path temporary = Files.createTempFile(parent, LIBRARY_FILE_NAME + ".", ".tmp");
      try {
        Files.write(temporary, bytes);
        setOwnerPermissions(temporary);
        moveIntoPlace(temporary, target);
      } finally {
        Files.deleteIfExists(temporary);
      }
      verifyDigest(target, expectedDigest);
    } catch (IOException failure) {
      throw new IllegalStateException("Unable to extract Apple bridge to " + target, failure);
    }
  }

  private static void moveIntoPlace(Path source, Path target) throws IOException {
    try {
      Files.move(source, target, StandardCopyOption.ATOMIC_MOVE);
    } catch (AtomicMoveNotSupportedException failure) {
      try {
        Files.move(source, target);
      } catch (FileAlreadyExistsException ignored) {
        // A concurrent process populated the content-addressed cache.
      }
    } catch (FileAlreadyExistsException ignored) {
      // A concurrent process populated the content-addressed cache.
    }
  }

  private static void verifyDigest(Path path, String expectedDigest) throws IOException {
    String actualDigest = sha256(Files.readAllBytes(path));
    if (!MessageDigest.isEqual(
        expectedDigest.getBytes(StandardCharsets.US_ASCII),
        actualDigest.getBytes(StandardCharsets.US_ASCII))) {
      throw new SecurityException("Cached Apple Foundation Models bridge SHA-256 mismatch");
    }
  }

  private static void setOwnerPermissions(Path path) throws IOException {
    try {
      Files.setPosixFilePermissions(path, OWNER_PERMISSIONS);
    } catch (UnsupportedOperationException ignored) {
      // Non-POSIX filesystems do not expose executable permissions.
    }
  }

  private static String sha256(byte[] bytes) {
    try {
      return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
    } catch (NoSuchAlgorithmException failure) {
      throw new IllegalStateException("SHA-256 is unavailable", failure);
    }
  }
}
