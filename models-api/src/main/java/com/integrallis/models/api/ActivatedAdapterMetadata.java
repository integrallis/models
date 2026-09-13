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
package com.integrallis.models.api;

import java.util.Collections;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.TreeMap;

/** Immutable provenance and activation contract for a loaded Activated-LoRA adapter. */
public record ActivatedAdapterMetadata(
    String baseModel,
    String baseRevision,
    String baseArtifactSha256,
    Map<String, String> tokenizerFileSha256,
    String adapterSha256,
    int rank,
    int alpha,
    List<Integer> invocationTokens,
    TrainingProvenance trainingProvenance) {

  public ActivatedAdapterMetadata {
    baseModel = requireText(baseModel, "baseModel");
    baseRevision = requireHex(baseRevision, 40, "baseRevision");
    baseArtifactSha256 = requireHex(baseArtifactSha256, 64, "baseArtifactSha256");
    Objects.requireNonNull(tokenizerFileSha256, "tokenizerFileSha256");
    if (tokenizerFileSha256.isEmpty()) {
      throw new IllegalArgumentException("tokenizerFileSha256 must not be empty");
    }
    TreeMap<String, String> normalizedTokenizerHashes = new TreeMap<>();
    tokenizerFileSha256.forEach(
        (name, hash) ->
            normalizedTokenizerHashes.put(
                requireRelativeFile(name, "tokenizer file"),
                requireHex(hash, 64, "tokenizer file SHA-256")));
    tokenizerFileSha256 = Collections.unmodifiableMap(normalizedTokenizerHashes);
    adapterSha256 = requireHex(adapterSha256, 64, "adapterSha256");
    if (rank <= 0) {
      throw new IllegalArgumentException("rank must be > 0: " + rank);
    }
    if (alpha <= 0) {
      throw new IllegalArgumentException("alpha must be > 0: " + alpha);
    }
    Objects.requireNonNull(invocationTokens, "invocationTokens");
    if (invocationTokens.isEmpty()
        || invocationTokens.stream().anyMatch(token -> token == null || token < 0)) {
      throw new IllegalArgumentException("invocationTokens must contain nonnegative token IDs");
    }
    invocationTokens = List.copyOf(invocationTokens);
    trainingProvenance = Objects.requireNonNull(trainingProvenance, "trainingProvenance");
  }

  /** One exact, immutable training input and its role in the prepared corpus. */
  public record TrainingSource(
      String role, String dataset, String datasetRevision, String sourceFile, String sourceSha256) {

    public TrainingSource {
      role = requireText(role, "role");
      dataset = requireText(dataset, "dataset");
      datasetRevision = requireHex(datasetRevision, 40, "datasetRevision");
      sourceFile = requireRelativeFile(sourceFile, "sourceFile");
      sourceSha256 = requireHex(sourceSha256, 64, "sourceSha256");
    }
  }

  /** Exact row selection consumed from one prepared training or validation split. */
  public record TrainingSelection(
      int usable,
      int skippedOverMaxLength,
      int noCall,
      int multipleCall,
      String sourceLinesSha256) {

    public TrainingSelection {
      if (usable <= 0) {
        throw new IllegalArgumentException("usable must be > 0: " + usable);
      }
      if (skippedOverMaxLength < 0) {
        throw new IllegalArgumentException(
            "skippedOverMaxLength must be >= 0: " + skippedOverMaxLength);
      }
      if (noCall < 0 || noCall > usable) {
        throw new IllegalArgumentException("noCall must be in [0, usable]: " + noCall);
      }
      if (multipleCall < 0 || multipleCall > usable) {
        throw new IllegalArgumentException("multipleCall must be in [0, usable]: " + multipleCall);
      }
      sourceLinesSha256 = requireHex(sourceLinesSha256, 64, "sourceLinesSha256");
    }
  }

  /** Exact, immutable identity of every data source and formatter that produced the adapter. */
  public record TrainingProvenance(
      List<TrainingSource> sources,
      int preparedSchemaVersion,
      String preparedManifestSha256,
      String trainSha256,
      String validationSha256,
      Optional<TrainingSelection> trainSelection,
      Optional<TrainingSelection> validationSelection,
      String formatter,
      String formatterSha256) {

    public TrainingProvenance {
      Objects.requireNonNull(sources, "sources");
      if (sources.isEmpty() || sources.stream().anyMatch(Objects::isNull)) {
        throw new IllegalArgumentException("sources must contain at least one training source");
      }
      sources = List.copyOf(sources);
      if (new HashSet<>(sources).size() != sources.size()) {
        throw new IllegalArgumentException("sources must not contain duplicate training sources");
      }
      if (preparedSchemaVersion <= 0) {
        throw new IllegalArgumentException("preparedSchemaVersion must be > 0");
      }
      preparedManifestSha256 = requireHex(preparedManifestSha256, 64, "preparedManifestSha256");
      trainSha256 = requireHex(trainSha256, 64, "trainSha256");
      validationSha256 = requireHex(validationSha256, 64, "validationSha256");
      trainSelection = Objects.requireNonNull(trainSelection, "trainSelection");
      validationSelection = Objects.requireNonNull(validationSelection, "validationSelection");
      formatter = requireRelativeFile(formatter, "formatter");
      formatterSha256 = requireHex(formatterSha256, 64, "formatterSha256");
    }

    /** Compatibility constructor for metadata schemas that predate exact subset selection. */
    public TrainingProvenance(
        List<TrainingSource> sources,
        int preparedSchemaVersion,
        String preparedManifestSha256,
        String trainSha256,
        String validationSha256,
        String formatter,
        String formatterSha256) {
      this(
          sources,
          preparedSchemaVersion,
          preparedManifestSha256,
          trainSha256,
          validationSha256,
          Optional.empty(),
          Optional.empty(),
          formatter,
          formatterSha256);
    }

    /** Compatibility constructor for a corpus with one primary training source. */
    public TrainingProvenance(
        String dataset,
        String datasetRevision,
        String sourceFile,
        String sourceSha256,
        String preparedManifestSha256,
        String trainSha256,
        String validationSha256,
        String formatter,
        String formatterSha256) {
      this(
          List.of(
              new TrainingSource("primary", dataset, datasetRevision, sourceFile, sourceSha256)),
          1,
          preparedManifestSha256,
          trainSha256,
          validationSha256,
          Optional.empty(),
          Optional.empty(),
          formatter,
          formatterSha256);
    }

    public String dataset() {
      return sources.getFirst().dataset();
    }

    public String datasetRevision() {
      return sources.getFirst().datasetRevision();
    }

    public String sourceFile() {
      return sources.getFirst().sourceFile();
    }

    public String sourceSha256() {
      return sources.getFirst().sourceSha256();
    }
  }

  private static String requireText(String value, String name) {
    Objects.requireNonNull(value, name);
    if (value.isBlank()) {
      throw new IllegalArgumentException(name + " must not be blank");
    }
    return value;
  }

  private static String requireHex(String value, int length, String name) {
    Objects.requireNonNull(value, name);
    String normalized = value.toLowerCase(java.util.Locale.ROOT);
    if (!normalized.matches("[0-9a-f]{" + length + "}")) {
      throw new IllegalArgumentException(name + " must be " + length + " hexadecimal characters");
    }
    return normalized;
  }

  private static String requireRelativeFile(String value, String name) {
    String normalized = requireText(value, name);
    if (normalized.startsWith("/")
        || normalized.contains("\\")
        || normalized.contains("//")
        || normalized.matches("^[A-Za-z]:.*")) {
      throw new IllegalArgumentException(name + " must be a normalized relative path: " + value);
    }
    for (String element : normalized.split("/")) {
      if (element.isBlank() || ".".equals(element) || "..".equals(element)) {
        throw new IllegalArgumentException(name + " must be a normalized relative path: " + value);
      }
    }
    return normalized;
  }
}
