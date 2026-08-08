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
package com.integrallis.models.router;

import java.io.IOException;
import java.io.InputStream;
import java.io.UncheckedIOException;
import java.nio.file.AtomicMoveNotSupportedException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.Comparator;
import java.util.HexFormat;
import java.util.Objects;
import java.util.zip.ZipEntry;
import java.util.zip.ZipInputStream;

/**
 * Unpacks the task index shipped inside the jar so it can be opened.
 *
 * <p>The index is a directory of memory-mapped files, and a jar entry cannot be mapped, so it
 * travels as a zip and is expanded once into a cache directory. The cache path carries a digest of
 * the archive, so a jar upgrade that changes the index lands in a new directory rather than
 * silently reusing the old one.
 */
public final class TaskIndexResource {

  /** Classpath location of the packaged index. */
  public static final String RESOURCE = "/com/integrallis/models/router/task-index.zip";

  /**
   * Guards against an archive that expands to an implausible size.
   *
   * <p>Set against the real artifact rather than guessed: the shipped index is under a megabyte of
   * 4-bit codes, so 64 MiB leaves room for a corpus many times larger or a return to full-precision
   * vectors, while still refusing anything pathological. The original 512 MiB was picked before
   * anything had been built and would have let a corrupt archive fill a cache directory.
   */
  private static final long MAX_TOTAL_BYTES = 64L * 1024 * 1024;

  private TaskIndexResource() {}

  /**
   * Opens the index bundled with this library, expanding it on first use.
   *
   * @return the opened index; the caller closes it
   * @throws IllegalStateException if no index is packaged in this jar
   */
  public static TaskIndex openBundled() {
    return TaskIndex.open(extractTo(defaultCacheRoot()));
  }

  /**
   * Expands the packaged index beneath {@code cacheRoot}, reusing a previous expansion.
   *
   * @param cacheRoot directory to hold expanded indexes
   * @return the directory holding the expanded index
   * @throws IllegalStateException if no index is packaged in this jar
   */
  public static Path extractTo(Path cacheRoot) {
    Objects.requireNonNull(cacheRoot, "cacheRoot");
    return expand(readResource(), cacheRoot);
  }

  /**
   * Expands archive bytes beneath {@code cacheRoot}, reusing a previous expansion.
   *
   * <p>Separate from {@link #extractTo} so the expansion can be exercised without packaging an
   * archive into the test classpath first.
   *
   * @param archive the zipped index
   * @param cacheRoot directory to hold expanded indexes
   * @return the directory holding the expanded index
   */
  static Path expand(byte[] archive, Path cacheRoot) {
    Path target = cacheRoot.resolve("task-index-" + sha256(archive).substring(0, 16));
    if (Files.isReadable(target.resolve(TaskIndexBuilder.MANIFEST))) {
      return target;
    }
    try {
      Files.createDirectories(cacheRoot);
      // Expand into a scratch directory and move it into place, so a reader never observes a
      // half-written index and two processes racing cannot interleave their writes.
      Path scratch = Files.createTempDirectory(cacheRoot, "task-index-partial-");
      try {
        unzip(archive, scratch);
        try {
          Files.move(scratch, target, StandardCopyOption.ATOMIC_MOVE);
        } catch (AtomicMoveNotSupportedException | java.nio.file.FileAlreadyExistsException e) {
          // Another process won the race, or the filesystem cannot move atomically. Either way an
          // index that already exists at the digest path is the one we would have written.
          if (!Files.isReadable(target.resolve(TaskIndexBuilder.MANIFEST))) {
            Files.move(scratch, target, StandardCopyOption.REPLACE_EXISTING);
          }
        }
      } finally {
        deleteIfPresent(scratch);
      }
    } catch (IOException e) {
      throw new UncheckedIOException("cannot expand " + RESOURCE + " into " + cacheRoot, e);
    }
    return target;
  }

  private static byte[] readResource() {
    try (InputStream in = TaskIndexResource.class.getResourceAsStream(RESOURCE)) {
      if (in == null) {
        throw new IllegalStateException(
            "no task index packaged at "
                + RESOURCE
                + "; build one with TaskIndexBuilder and open it with TaskIndex.open");
      }
      return in.readAllBytes();
    } catch (IOException e) {
      throw new UncheckedIOException("cannot read " + RESOURCE, e);
    }
  }

  private static void unzip(byte[] archive, Path destination) throws IOException {
    Path root = destination.toRealPath();
    long written = 0;
    try (ZipInputStream zip = new ZipInputStream(new java.io.ByteArrayInputStream(archive))) {
      ZipEntry entry;
      while ((entry = zip.getNextEntry()) != null) {
        // Reject an entry whose name escapes the destination: a crafted archive could otherwise
        // write anywhere the process can reach.
        Path resolved = root.resolve(entry.getName()).normalize();
        if (!resolved.startsWith(root)) {
          throw new IOException("zip entry escapes the destination: " + entry.getName());
        }
        if (entry.isDirectory()) {
          Files.createDirectories(resolved);
          continue;
        }
        Path parent = resolved.getParent();
        if (parent != null) {
          Files.createDirectories(parent);
        }
        written += Files.copy(zip, resolved, StandardCopyOption.REPLACE_EXISTING);
        if (written > MAX_TOTAL_BYTES) {
          throw new IOException("archive expands beyond " + MAX_TOTAL_BYTES + " bytes");
        }
      }
    }
  }

  private static Path defaultCacheRoot() {
    String override = System.getProperty("models.router.indexCache");
    if (override != null && !override.isBlank()) {
      return Path.of(override);
    }
    String xdg = System.getenv("XDG_CACHE_HOME");
    Path base =
        xdg != null && !xdg.isBlank()
            ? Path.of(xdg)
            : Path.of(System.getProperty("user.home"), ".cache");
    return base.resolve("models-router");
  }

  private static String sha256(byte[] bytes) {
    try {
      return HexFormat.of().formatHex(MessageDigest.getInstance("SHA-256").digest(bytes));
    } catch (NoSuchAlgorithmException e) {
      throw new IllegalStateException("SHA-256 is required by the platform", e);
    }
  }

  private static void deleteIfPresent(Path directory) throws IOException {
    if (!Files.exists(directory)) {
      return;
    }
    try (var walk = Files.walk(directory)) {
      for (Path path : walk.sorted(Comparator.reverseOrder()).toList()) {
        Files.deleteIfExists(path);
      }
    }
  }
}
