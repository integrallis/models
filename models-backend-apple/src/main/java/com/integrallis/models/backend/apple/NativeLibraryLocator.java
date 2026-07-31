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
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.Locale;
import java.util.Objects;
import java.util.Optional;

@FunctionalInterface
interface NativeLibraryLocator {

  String LIBRARY_PATH_PROPERTY = "models.apple.foundation.library";
  String LIBRARY_PATH_ENV = "MODELS_APPLE_FOUNDATION_LIBRARY";
  String LIBRARY_SHA256_PROPERTY = "models.apple.foundation.library.sha256";
  String LIBRARY_SHA256_ENV = "MODELS_APPLE_FOUNDATION_LIBRARY_SHA256";
  String ALLOW_UNVERIFIED_PROPERTY = "models.apple.foundation.library.allow-unverified";
  String ALLOW_UNVERIFIED_ENV = "MODELS_APPLE_FOUNDATION_LIBRARY_ALLOW_UNVERIFIED";

  Optional<Path> locate();

  static NativeLibraryLocator system() {
    return () ->
        resolveConfigured(
            System.getProperty(LIBRARY_PATH_PROPERTY),
            System.getenv(LIBRARY_PATH_ENV),
            System.getProperty(LIBRARY_SHA256_PROPERTY),
            System.getenv(LIBRARY_SHA256_ENV),
            System.getProperty(ALLOW_UNVERIFIED_PROPERTY),
            System.getenv(ALLOW_UNVERIFIED_ENV),
            BundledAppleFoundationLibrary::resolve);
  }

  static Optional<Path> resolveConfigured(
      String propertyValue,
      String environmentValue,
      String propertyDigest,
      String environmentDigest,
      String propertyAllowUnverified,
      String environmentAllowUnverified,
      NativeLibraryLocator fallback) {
    String configured = firstNonBlank(propertyValue, environmentValue);
    if (configured == null || configured.isBlank()) {
      return Objects.requireNonNull(fallback, "fallback").locate();
    }

    Path configuredPath = Path.of(configured).toAbsolutePath().normalize();
    String expectedDigest = firstNonBlank(propertyDigest, environmentDigest);
    if (expectedDigest == null) {
      if (isTrue(firstNonBlank(propertyAllowUnverified, environmentAllowUnverified))) {
        return Optional.of(configuredPath);
      }
      throw new SecurityException(
          "An explicit Apple Foundation Models bridge requires SHA-256 verification. Set "
              + LIBRARY_SHA256_PROPERTY
              + " or "
              + LIBRARY_SHA256_ENV
              + "; use "
              + ALLOW_UNVERIFIED_PROPERTY
              + "=true only for local bridge development.");
    }

    String normalizedDigest = expectedDigest.strip().toLowerCase(Locale.ROOT);
    if (!normalizedDigest.matches("[0-9a-f]{64}")) {
      throw new IllegalArgumentException("Apple bridge SHA-256 must contain 64 hexadecimal digits");
    }
    String actualDigest = sha256(configuredPath);
    if (!MessageDigest.isEqual(
        normalizedDigest.getBytes(StandardCharsets.US_ASCII),
        actualDigest.getBytes(StandardCharsets.US_ASCII))) {
      throw new SecurityException(
          "Apple Foundation Models bridge SHA-256 mismatch: expected "
              + normalizedDigest
              + " but found "
              + actualDigest);
    }
    return Optional.of(configuredPath);
  }

  static NativeLibraryLocator fixed(Path path) {
    return () -> Optional.of(path);
  }

  static NativeLibraryLocator empty() {
    return Optional::empty;
  }

  private static String firstNonBlank(String preferred, String fallback) {
    return preferred != null && !preferred.isBlank()
        ? preferred
        : fallback != null && !fallback.isBlank() ? fallback : null;
  }

  private static boolean isTrue(String value) {
    return value != null && Boolean.parseBoolean(value.strip());
  }

  private static String sha256(Path path) {
    try {
      byte[] bytes = Files.readAllBytes(path);
      return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
    } catch (IOException failure) {
      throw new IllegalStateException("Unable to read configured Apple bridge " + path, failure);
    } catch (NoSuchAlgorithmException failure) {
      throw new IllegalStateException("SHA-256 is unavailable", failure);
    }
  }
}
