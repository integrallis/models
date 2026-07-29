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

import com.integrallis.models.backend.purejava.PureJavaBackend;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Objects;

/** Exact filesystem source used by an in-process benchmark. */
record PureJavaModelSource(String identity, Path artifact) {

  PureJavaModelSource {
    if (identity == null || identity.isBlank()) {
      throw new IllegalArgumentException("identity must not be blank");
    }
    String resolvedIdentity = identity.trim();
    identity = resolvedIdentity;
    Path resolvedArtifact = requireArtifact(Objects.requireNonNull(artifact, "artifact"));
    artifact = resolvedArtifact;
  }

  static PureJavaModelSource resolve(String pathValue) {
    if (pathValue == null || pathValue.isBlank()) {
      throw new IllegalArgumentException("in-process backends require --model");
    }
    String identity = pathValue.trim();
    return new PureJavaModelSource(identity, Path.of(identity));
  }

  PureJavaBackend load() {
    return PureJavaBackend.load(artifact);
  }

  private static Path requireArtifact(Path artifact) {
    if (!Files.isRegularFile(artifact)) {
      throw new IllegalArgumentException("artifact does not exist: " + artifact);
    }
    return artifact;
  }
}
