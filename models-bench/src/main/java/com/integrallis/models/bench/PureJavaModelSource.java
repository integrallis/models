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
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import org.modeljars.ModelJarDescriptor;
import org.modeljars.ModelJarRegistry;

/** Exact filesystem or ModelJar source used by a pure-Java benchmark process. */
record PureJavaModelSource(
    String identity, Path artifact, Optional<ModelJarDescriptor> descriptor) {

  PureJavaModelSource {
    if (identity == null || identity.isBlank()) {
      throw new IllegalArgumentException("identity must not be blank");
    }
    String resolvedIdentity = identity.trim();
    identity = resolvedIdentity;
    Path resolvedArtifact = requireArtifact(Objects.requireNonNull(artifact, "artifact"));
    artifact = resolvedArtifact;
    descriptor = Objects.requireNonNull(descriptor, "descriptor");
    descriptor.ifPresent(
        value -> {
          if (!resolvedIdentity.equals(value.alias())) {
            throw new IllegalArgumentException(
                "ModelJar descriptor alias does not match benchmark identity");
          }
          if (!value.supportsBackend("pure-java")) {
            throw new IllegalArgumentException(
                "ModelJar does not support pure-java: " + resolvedIdentity);
          }
          if (!value.localPath().filter(resolvedArtifact::equals).isPresent()) {
            throw new IllegalArgumentException(
                "ModelJar descriptor local path does not match benchmark artifact");
          }
        });
  }

  static PureJavaModelSource resolve(
      String pathValue, String modelJarAlias, ModelJarRegistry registry) {
    Objects.requireNonNull(registry, "registry");
    boolean hasPath = pathValue != null && !pathValue.isBlank();
    boolean hasAlias = modelJarAlias != null && !modelJarAlias.isBlank();
    if (hasPath == hasAlias) {
      throw new IllegalArgumentException("pure-java requires exactly one of --model or --modeljar");
    }
    if (hasPath) {
      Path artifact = Path.of(pathValue);
      return new PureJavaModelSource(pathValue, artifact, Optional.empty());
    }

    String alias = modelJarAlias.trim();
    List<ModelJarDescriptor> matches =
        registry.descriptors().stream().filter(value -> alias.equals(value.alias())).toList();
    if (matches.isEmpty()) {
      throw new IllegalArgumentException("unknown ModelJar alias: " + alias);
    }
    if (matches.size() > 1) {
      throw new IllegalArgumentException("ambiguous ModelJar alias: " + alias);
    }
    ModelJarDescriptor descriptor = matches.getFirst();
    Path artifact =
        descriptor
            .localPath()
            .orElseThrow(
                () -> new IllegalArgumentException("ModelJar has no local artifact: " + alias));
    return new PureJavaModelSource(alias, artifact, Optional.of(descriptor));
  }

  PureJavaBackend load() {
    return descriptor.map(PureJavaBackend::load).orElseGet(() -> PureJavaBackend.load(artifact));
  }

  private static Path requireArtifact(Path artifact) {
    if (!Files.isRegularFile(artifact)) {
      throw new IllegalArgumentException("artifact does not exist: " + artifact);
    }
    return artifact;
  }
}
