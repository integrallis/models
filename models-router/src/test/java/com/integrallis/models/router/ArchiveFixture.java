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
import java.io.OutputStream;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.zip.ZipEntry;
import java.util.zip.ZipOutputStream;

/** Builds zip archives for {@link TaskIndexResourceTest} and drives extraction on one. */
final class ArchiveFixture {

  private ArchiveFixture() {}

  static Path write(Path directory) throws IOException {
    Files.createDirectories(directory);
    Path archive = directory.resolve("index.zip");
    try (OutputStream out = Files.newOutputStream(archive);
        ZipOutputStream zip = new ZipOutputStream(out)) {
      zip.putNextEntry(new ZipEntry(TaskIndexBuilder.MANIFEST));
      zip.write(
          "embeddingModelId=fake\ndimension=4\ntasks=code\n".getBytes(StandardCharsets.UTF_8));
      zip.closeEntry();
    }
    return archive;
  }

  static Path writeEscaping(Path directory) throws IOException {
    Files.createDirectories(directory);
    Path archive = directory.resolve("evil.zip");
    try (OutputStream out = Files.newOutputStream(archive);
        ZipOutputStream zip = new ZipOutputStream(out)) {
      zip.putNextEntry(new ZipEntry("../../escaped.txt"));
      zip.write("nope".getBytes(StandardCharsets.UTF_8));
      zip.closeEntry();
    }
    return archive;
  }

  /** Expands an archive read from disk through the package-private entry point. */
  static Path extract(Path archive, Path cacheRoot) throws IOException {
    return TaskIndexResource.expand(Files.readAllBytes(archive), cacheRoot);
  }
}
