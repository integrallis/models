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
import java.io.OutputStream;
import java.net.HttpURLConnection;
import java.net.URLConnection;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.nio.file.StandardOpenOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** Downloads one pinned fixture and installs it only after size and SHA-256 verification. */
final class ModelFixtureInstaller {
  private static final int BUFFER_SIZE = 1024 * 1024;
  private static final int MAX_DOWNLOAD_ATTEMPTS = 4;
  private static final long INITIAL_RETRY_DELAY_MILLIS = 1_000;
  private static final Pattern CONTENT_RANGE = Pattern.compile("bytes\\s+(\\d+)-(\\d+)/(\\d+)");

  private ModelFixtureInstaller() {}

  static Path install(ModelFixtureDescriptor descriptor) throws IOException {
    Path target = descriptor.localPath().orElseThrow();
    if (matches(target, descriptor)) {
      return target;
    }

    Files.createDirectories(target.toAbsolutePath().getParent());
    Path temporary = Files.createTempFile(target.toAbsolutePath().getParent(), ".model-", ".part");
    try {
      download(descriptor, temporary);
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

  private static void download(ModelFixtureDescriptor descriptor, Path temporary)
      throws IOException {
    long expectedSize = descriptor.sizeBytes().orElseThrow();
    IOException lastFailure = null;
    for (int attempt = 1; attempt <= MAX_DOWNLOAD_ATTEMPTS; attempt++) {
      long downloaded = Files.size(temporary);
      if (downloaded >= expectedSize) {
        return;
      }
      try {
        downloadAttempt(descriptor, temporary, expectedSize);
        downloaded = Files.size(temporary);
        if (downloaded >= expectedSize) {
          return;
        }
        lastFailure =
            new IOException(
                "Incomplete fixture download: expected "
                    + expectedSize
                    + " bytes, got "
                    + downloaded);
      } catch (IOException failure) {
        lastFailure = failure;
      }
      if (attempt < MAX_DOWNLOAD_ATTEMPTS) {
        pauseBeforeRetry(attempt);
      }
    }
    throw new IOException(
        "Unable to download fixture "
            + descriptor.id()
            + " after "
            + MAX_DOWNLOAD_ATTEMPTS
            + " attempts; received "
            + Files.size(temporary)
            + " of "
            + expectedSize
            + " bytes",
        lastFailure);
  }

  private static void downloadAttempt(
      ModelFixtureDescriptor descriptor, Path temporary, long expectedSize) throws IOException {
    long offset = Files.size(temporary);
    URLConnection connection = descriptor.downloadUri().toURL().openConnection();
    connection.setRequestProperty("User-Agent", "integrallis-models-test-fixtures/0.1");
    connection.setRequestProperty("Accept-Encoding", "identity");
    connection.setConnectTimeout(30_000);
    connection.setReadTimeout(120_000);
    HttpURLConnection http =
        connection instanceof HttpURLConnection httpConnection ? httpConnection : null;
    try {
      if (http != null && offset > 0) {
        http.setRequestProperty("Range", "bytes=" + offset + "-");
      }

      boolean append = false;
      if (http != null) {
        int status = http.getResponseCode();
        if (offset > 0 && status == HttpURLConnection.HTTP_PARTIAL) {
          verifyContentRange(http.getHeaderField("Content-Range"), offset, expectedSize);
          append = true;
        } else if (status != HttpURLConnection.HTTP_OK) {
          throw new IOException(
              "Fixture download returned HTTP " + status + " for " + descriptor.id());
        }
      }

      try (InputStream input = connection.getInputStream();
          OutputStream output =
              append
                  ? Files.newOutputStream(
                      temporary, StandardOpenOption.WRITE, StandardOpenOption.APPEND)
                  : Files.newOutputStream(
                      temporary, StandardOpenOption.WRITE, StandardOpenOption.TRUNCATE_EXISTING)) {
        input.transferTo(output);
      }
    } finally {
      if (http != null) {
        http.disconnect();
      }
    }
  }

  private static void verifyContentRange(String contentRange, long offset, long expectedSize)
      throws IOException {
    Matcher matcher = contentRange == null ? null : CONTENT_RANGE.matcher(contentRange);
    if (matcher == null || !matcher.matches()) {
      throw new IOException(
          "Invalid Content-Range for fixture download at byte " + offset + ": " + contentRange);
    }
    try {
      long start = Long.parseLong(matcher.group(1));
      long end = Long.parseLong(matcher.group(2));
      long total = Long.parseLong(matcher.group(3));
      if (start != offset || end < start || end >= expectedSize || total != expectedSize) {
        throw new IOException(
            "Invalid Content-Range for fixture download at byte " + offset + ": " + contentRange);
      }
    } catch (NumberFormatException malformed) {
      throw new IOException("Invalid numeric Content-Range: " + contentRange, malformed);
    }
  }

  private static void pauseBeforeRetry(int failedAttempt) throws IOException {
    try {
      Thread.sleep(INITIAL_RETRY_DELAY_MILLIS << (failedAttempt - 1));
    } catch (InterruptedException interrupted) {
      Thread.currentThread().interrupt();
      throw new IOException("Interrupted while waiting to retry fixture download", interrupted);
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
