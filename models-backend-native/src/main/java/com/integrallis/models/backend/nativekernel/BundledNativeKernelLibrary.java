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

import java.io.IOException;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.io.Reader;
import java.net.URL;
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
import java.util.Enumeration;
import java.util.HexFormat;
import java.util.List;
import java.util.Objects;
import java.util.Properties;
import java.util.Set;

/**
 * Resolves an integrity-checked native library supplied by a platform artifact on the classpath.
 */
final class BundledNativeKernelLibrary {
  static final String METADATA_FILE_NAME = "native.properties";
  static final String CACHE_DIRECTORY_PROPERTY = "models.native.kernels.cache";

  private static final String ABI_PROPERTY = "abi";
  private static final String PLATFORM_PROPERTY = "platform";
  private static final String LIBRARY_PROPERTY = "library";
  private static final String SHA_256_PROPERTY = "sha256";
  private static final Set<PosixFilePermission> LIBRARY_PERMISSIONS =
      EnumSet.of(
          PosixFilePermission.OWNER_READ,
          PosixFilePermission.OWNER_WRITE,
          PosixFilePermission.OWNER_EXECUTE);
  private static final Set<PosixFilePermission> DIRECTORY_PERMISSIONS =
      EnumSet.of(
          PosixFilePermission.OWNER_READ,
          PosixFilePermission.OWNER_WRITE,
          PosixFilePermission.OWNER_EXECUTE);

  private BundledNativeKernelLibrary() {}

  static Path resolve() {
    return resolve(
        RustFfmBackend.class.getClassLoader(), NativeKernelPlatform.current(), defaultCacheRoot());
  }

  static Path resolve(
      ClassLoader classLoader, NativeKernelPlatform platform, Path nativeCacheRoot) {
    String metadataResource = platform.resourceDirectory() + METADATA_FILE_NAME;
    URL metadataUrl = uniqueResource(classLoader, metadataResource, platform);
    Properties metadata = loadMetadata(metadataUrl);
    validateMetadata(metadata, platform);

    String libraryName = metadata.getProperty(LIBRARY_PROPERTY);
    String expectedDigest = metadata.getProperty(SHA_256_PROPERTY);
    URL libraryUrl =
        uniqueResource(classLoader, platform.resourceDirectory() + libraryName, platform);
    byte[] libraryBytes = readBytes(libraryUrl, "native library");
    String actualDigest = sha256(libraryBytes);
    if (!MessageDigest.isEqual(
        expectedDigest.getBytes(StandardCharsets.US_ASCII),
        actualDigest.getBytes(StandardCharsets.US_ASCII))) {
      throw new SecurityException(
          "native kernel SHA-256 mismatch for "
              + platform.id()
              + ": expected "
              + expectedDigest
              + " but found "
              + actualDigest);
    }

    Path cacheDirectory =
        nativeCacheRoot
            .toAbsolutePath()
            .normalize()
            .resolve("abi-" + NativeKernelLibrary.ABI_VERSION)
            .resolve(platform.id())
            .resolve(expectedDigest);
    Path extractedLibrary = cacheDirectory.resolve(libraryName);
    extractOnce(extractedLibrary, libraryBytes, expectedDigest);
    return extractedLibrary;
  }

  private static Path defaultCacheRoot() {
    String configured = System.getProperty(CACHE_DIRECTORY_PROPERTY);
    if (configured != null && !configured.isBlank()) {
      return Path.of(configured);
    }
    return Path.of(System.getProperty("user.home"), ".models", "native-kernels");
  }

  private static URL uniqueResource(
      ClassLoader classLoader, String resourceName, NativeKernelPlatform platform) {
    try {
      Enumeration<URL> resources = classLoader.getResources(resourceName);
      List<URL> matches = new ArrayList<>(2);
      while (resources.hasMoreElements()) {
        matches.add(resources.nextElement());
      }
      if (matches.isEmpty()) {
        throw new IllegalStateException(
            "no Models native-kernel artifact for "
                + platform.id()
                + "; expected classpath resource "
                + resourceName);
      }
      if (matches.size() != 1) {
        throw new IllegalStateException(
            "multiple Models native-kernel artifacts provide " + resourceName + ": " + matches);
      }
      return matches.getFirst();
    } catch (IOException failure) {
      throw new IllegalStateException(
          "failed to locate classpath resource " + resourceName, failure);
    }
  }

  private static Properties loadMetadata(URL metadataUrl) {
    Properties metadata = new Properties();
    try (InputStream input = metadataUrl.openStream();
        Reader reader = new InputStreamReader(input, StandardCharsets.UTF_8)) {
      metadata.load(reader);
      return metadata;
    } catch (IOException failure) {
      throw new IllegalStateException(
          "failed to read native kernel metadata " + metadataUrl, failure);
    }
  }

  private static void validateMetadata(Properties metadata, NativeKernelPlatform expectedPlatform) {
    String abi = requireMetadata(metadata, ABI_PROPERTY);
    if (!Integer.toString(NativeKernelLibrary.ABI_VERSION).equals(abi)) {
      throw new IllegalStateException(
          "native kernel artifact ABI "
              + abi
              + " does not match Java ABI "
              + NativeKernelLibrary.ABI_VERSION);
    }
    String platform = requireMetadata(metadata, PLATFORM_PROPERTY);
    if (!expectedPlatform.id().equals(platform)) {
      throw new IllegalStateException(
          "native kernel artifact platform "
              + platform
              + " does not match runtime "
              + expectedPlatform.id());
    }
    String library = requireMetadata(metadata, LIBRARY_PROPERTY);
    if (!expectedPlatform.libraryFileName().equals(library)) {
      throw new IllegalStateException(
          "native kernel artifact library "
              + library
              + " does not match platform filename "
              + expectedPlatform.libraryFileName());
    }
    String digest = requireMetadata(metadata, SHA_256_PROPERTY);
    if (!digest.matches("[0-9a-f]{64}")) {
      throw new IllegalStateException("native kernel artifact has an invalid SHA-256: " + digest);
    }
  }

  private static String requireMetadata(Properties metadata, String name) {
    String value = metadata.getProperty(name);
    if (value == null || value.isBlank()) {
      throw new IllegalStateException("native kernel metadata is missing " + name);
    }
    return value.strip();
  }

  private static byte[] readBytes(URL resource, String description) {
    try (InputStream input = resource.openStream()) {
      return input.readAllBytes();
    } catch (IOException failure) {
      throw new IllegalStateException("failed to read " + description + " " + resource, failure);
    }
  }

  private static void extractOnce(Path target, byte[] libraryBytes, String expectedDigest) {
    try {
      if (Files.isRegularFile(target)) {
        verifyCachedLibrary(target, expectedDigest);
        return;
      }
      Path parent =
          Objects.requireNonNull(target.getParent(), "native cache target must have a parent");
      Path fileName =
          Objects.requireNonNull(target.getFileName(), "native cache target must have a filename");
      Files.createDirectories(parent);
      setPosixPermissions(parent, DIRECTORY_PERMISSIONS);
      Path temporary = Files.createTempFile(parent, fileName.toString() + ".", ".tmp");
      try {
        Files.write(temporary, libraryBytes);
        setLibraryPermissions(temporary);
        moveIntoCache(temporary, target);
      } finally {
        Files.deleteIfExists(temporary);
      }
      verifyCachedLibrary(target, expectedDigest);
    } catch (IOException failure) {
      throw new IllegalStateException(
          "failed to extract native kernel library to " + target, failure);
    }
  }

  private static void moveIntoCache(Path source, Path target) throws IOException {
    try {
      Files.move(source, target, StandardCopyOption.ATOMIC_MOVE);
    } catch (AtomicMoveNotSupportedException failure) {
      try {
        Files.move(source, target);
      } catch (FileAlreadyExistsException concurrentExtraction) {
        // Another process populated the content-addressed cache first.
      }
    } catch (FileAlreadyExistsException concurrentExtraction) {
      // Another process populated the content-addressed cache first.
    }
  }

  private static void setLibraryPermissions(Path library) throws IOException {
    setPosixPermissions(library, LIBRARY_PERMISSIONS);
  }

  private static void setPosixPermissions(Path path, Set<PosixFilePermission> permissions)
      throws IOException {
    try {
      Files.setPosixFilePermissions(path, permissions);
    } catch (UnsupportedOperationException ignored) {
      // Windows and non-POSIX filesystems do not expose POSIX permissions.
    }
  }

  private static void verifyCachedLibrary(Path library, String expectedDigest) throws IOException {
    String actualDigest = sha256(Files.readAllBytes(library));
    if (!MessageDigest.isEqual(
        expectedDigest.getBytes(StandardCharsets.US_ASCII),
        actualDigest.getBytes(StandardCharsets.US_ASCII))) {
      throw new SecurityException(
          "cached native kernel SHA-256 mismatch at "
              + library
              + ": expected "
              + expectedDigest
              + " but found "
              + actualDigest);
    }
  }

  private static String sha256(byte[] bytes) {
    try {
      return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
    } catch (NoSuchAlgorithmException impossible) {
      throw new IllegalStateException("SHA-256 is required by the Java platform", impossible);
    }
  }
}
