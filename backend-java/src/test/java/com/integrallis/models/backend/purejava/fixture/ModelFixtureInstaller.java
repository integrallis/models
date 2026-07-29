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
package com.integrallis.models.backend.purejava.fixture;

import java.io.IOException;
import java.io.InputStream;
import java.net.URLConnection;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;

/** Downloads one pinned fixture and installs it only after size and SHA-256 verification. */
final class ModelFixtureInstaller {
  private static final int BUFFER_SIZE = 1024 * 1024;

  private ModelFixtureInstaller() {}

  static Path install(ModelFixtureDescriptor descriptor) throws IOException {
    Path target = descriptor.localPath().orElseThrow();
    if (matches(target, descriptor)) {
      return target;
    }

    Files.createDirectories(target.toAbsolutePath().getParent());
    Path temporary = Files.createTempFile(target.toAbsolutePath().getParent(), ".model-", ".part");
    try {
      URLConnection connection = descriptor.downloadUri().toURL().openConnection();
      connection.setRequestProperty("User-Agent", "integrallis-models-test-fixtures/0.1");
      connection.setConnectTimeout(30_000);
      connection.setReadTimeout(120_000);
      try (InputStream input = connection.getInputStream()) {
        Files.copy(input, temporary, StandardCopyOption.REPLACE_EXISTING);
      }
      requireMatch(temporary, descriptor);
      try {
        Files.move(
            temporary, target, StandardCopyOption.ATOMIC_MOVE, StandardCopyOption.REPLACE_EXISTING);
      } catch (AtomicMoveNotSupportedException unsupported) {
        Files.move(temporary, target, StandardCopyOption.REPLACE_EXISTING);
      }
      return target;
    } finally {
      Files.deleteIfExists(temporary);
    }
  }

  private static boolean matches(Path artifact, ModelFixtureDescriptor descriptor)
      throws IOException {
    return Files.isRegularFile(artifact)
        && Files.size(artifact) == descriptor.sizeBytes().orElseThrow()
        && sha256(artifact).equals(descriptor.sha256().orElseThrow());
  }

  private static void requireMatch(Path artifact, ModelFixtureDescriptor descriptor)
      throws IOException {
    long actualSize = Files.size(artifact);
    long expectedSize = descriptor.sizeBytes().orElseThrow();
    if (actualSize != expectedSize) {
      throw new IOException(
          "Fixture size mismatch for "
              + descriptor.id()
              + ": expected "
              + expectedSize
              + ", got "
              + actualSize);
    }
    String actualSha256 = sha256(artifact);
    String expectedSha256 = descriptor.sha256().orElseThrow();
    if (!actualSha256.equals(expectedSha256)) {
      throw new IOException(
          "Fixture SHA-256 mismatch for "
              + descriptor.id()
              + ": expected "
              + expectedSha256
              + ", got "
              + actualSha256);
    }
  }

  private static String sha256(Path artifact) throws IOException {
    MessageDigest digest;
    try {
      digest = MessageDigest.getInstance("SHA-256");
    } catch (NoSuchAlgorithmException failure) {
      throw new IllegalStateException("SHA-256 is unavailable", failure);
    }
    byte[] buffer = new byte[BUFFER_SIZE];
    try (InputStream input = Files.newInputStream(artifact)) {
      int read;
      while ((read = input.read(buffer)) >= 0) {
        if (read > 0) {
          digest.update(buffer, 0, read);
        }
      }
    }
    return HexFormat.of().formatHex(digest.digest());
  }
}
