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

import java.net.URI;
import java.nio.file.Path;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

/** Pinned artifact metadata used only by Models real-model tests. */
public final class ModelFixtureDescriptor {
  private final String id;
  private final String displayName;
  private final String modelVersion;
  private final String sourceId;
  private final String variant;
  private final String backend;
  private final URI downloadUri;
  private final Path artifact;
  private final String sha256;
  private final long sizeBytes;
  private final String architecture;
  private final String quantization;
  private final Set<String> capabilities;
  private final Set<String> features;
  private final boolean slow;
  private final String format;

  ModelFixtureDescriptor(
      String id,
      String displayName,
      String modelVersion,
      String sourceId,
      String variant,
      String backend,
      URI downloadUri,
      Path artifact,
      String sha256,
      long sizeBytes,
      String architecture,
      String quantization,
      Set<String> capabilities,
      Set<String> features,
      boolean slow,
      String format) {
    this.id = Objects.requireNonNull(id, "id");
    this.displayName = Objects.requireNonNull(displayName, "displayName");
    this.modelVersion = Objects.requireNonNull(modelVersion, "modelVersion");
    this.sourceId = Objects.requireNonNull(sourceId, "sourceId");
    this.variant = Objects.requireNonNull(variant, "variant");
    this.backend = Objects.requireNonNull(backend, "backend");
    this.downloadUri = Objects.requireNonNull(downloadUri, "downloadUri");
    this.artifact = Objects.requireNonNull(artifact, "artifact");
    this.sha256 = Objects.requireNonNull(sha256, "sha256");
    this.sizeBytes = sizeBytes;
    this.architecture = Objects.requireNonNull(architecture, "architecture");
    this.quantization = Objects.requireNonNull(quantization, "quantization");
    this.capabilities = Set.copyOf(capabilities);
    this.features = Set.copyOf(features);
    this.slow = slow;
    this.format = Objects.requireNonNull(format, "format");
  }

  public String id() {
    return id;
  }

  public String displayName() {
    return displayName;
  }

  public String modelVersion() {
    return modelVersion;
  }

  public String sourceId() {
    return sourceId;
  }

  public String variant() {
    return variant;
  }

  public String backend() {
    return backend;
  }

  public URI downloadUri() {
    return downloadUri;
  }

  public Optional<Path> localPath() {
    return Optional.of(artifact);
  }

  public Optional<String> sha256() {
    return Optional.of(sha256);
  }

  public Optional<Long> sizeBytes() {
    return Optional.of(sizeBytes);
  }

  public String architecture() {
    return architecture;
  }

  public String quantization() {
    return quantization;
  }

  public String format() {
    return format;
  }

  public Set<String> capabilities() {
    return capabilities;
  }

  public Set<String> features() {
    return features;
  }

  public boolean slow() {
    return slow;
  }
}
