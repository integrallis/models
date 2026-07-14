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
package com.integrallis.models.bench;

import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

/** Immutable identity of the exact model artifact used by a benchmark. */
public record BenchmarkArtifact(String path, String sha256, long sizeBytes) {

  public BenchmarkArtifact {
    if (path == null || path.isBlank()) {
      throw new IllegalArgumentException("path must not be blank");
    }
    if (sha256 == null || !sha256.matches("[0-9a-f]{64}")) {
      throw new IllegalArgumentException("sha256 must be a lowercase SHA-256 digest");
    }
    if (sizeBytes < 0) {
      throw new IllegalArgumentException("sizeBytes must not be negative");
    }
  }

  static BenchmarkArtifact inspect(Path artifact) throws IOException {
    Objects.requireNonNull(artifact, "artifact");
    Path realPath = artifact.toRealPath();
    if (!Files.isRegularFile(realPath)) {
      throw new IllegalArgumentException("artifact is not a regular file: " + artifact);
    }
    return new BenchmarkArtifact(
        realPath.toString(), Hashing.sha256(realPath), Files.size(realPath));
  }
}
