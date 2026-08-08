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

import com.integrallis.vectors.core.MetadataValue;
import com.integrallis.vectors.core.SimilarityFunction;
import com.integrallis.vectors.db.IndexType;
import com.integrallis.vectors.db.QuantizerKind;
import com.integrallis.vectors.db.SearchRequest;
import com.integrallis.vectors.db.SearchResult;
import com.integrallis.vectors.db.VectorCollection;
import java.io.IOException;
import java.io.UncheckedIOException;
import java.nio.charset.StandardCharsets;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Objects;
import java.util.Optional;
import java.util.Properties;

/**
 * A built task index, opened from the directory {@link TaskIndexBuilder} wrote.
 *
 * <p>Holds the labelled embeddings and answers nearest-neighbour queries against them. It does no
 * embedding of its own: the vectors it stores are only comparable with the model named in its
 * manifest, so producing the query vector is the caller's job and {@link #embeddingModelId()} says
 * which model that has to be.
 */
public final class TaskIndex implements AutoCloseable {

  private final VectorCollection collection;
  private final String embeddingModelId;
  private final int dimension;
  private final List<String> taskNames;

  private TaskIndex(
      VectorCollection collection, String embeddingModelId, int dimension, List<String> taskNames) {
    this.collection = collection;
    this.embeddingModelId = embeddingModelId;
    this.dimension = dimension;
    this.taskNames = taskNames;
  }

  /**
   * Opens an index directory.
   *
   * @param directory a directory previously written by {@link TaskIndexBuilder#build}
   * @return the opened index
   * @throws IllegalArgumentException if the directory holds no readable manifest
   */
  public static TaskIndex open(Path directory) {
    Objects.requireNonNull(directory, "directory");
    Path manifestFile = directory.resolve(TaskIndexBuilder.MANIFEST);
    if (!Files.isReadable(manifestFile)) {
      throw new IllegalArgumentException("no " + TaskIndexBuilder.MANIFEST + " in " + directory);
    }
    Properties manifest = new Properties();
    try (var reader = Files.newBufferedReader(manifestFile, StandardCharsets.UTF_8)) {
      manifest.load(reader);
    } catch (IOException e) {
      throw new UncheckedIOException("cannot read " + manifestFile, e);
    }
    String modelId = require(manifest, "embeddingModelId", manifestFile);
    int dimension = Integer.parseInt(require(manifest, "dimension", manifestFile));
    List<String> tasks = List.of(require(manifest, "tasks", manifestFile).split(","));
    // Absent in indexes written before quantization was an option, which were all full precision.
    QuantizerKind quantizer =
        QuantizerKind.valueOf(manifest.getProperty("quantizer", QuantizerKind.NONE.name()));

    VectorCollection collection =
        VectorCollection.builder()
            .dimension(dimension)
            .metric(SimilarityFunction.COSINE)
            .indexType(IndexType.FLAT)
            .quantizer(quantizer)
            .storagePath(directory.toAbsolutePath())
            .build();
    return new TaskIndex(collection, modelId, dimension, tasks);
  }

  private static String require(Properties manifest, String key, Path file) {
    String value = manifest.getProperty(key);
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException(file + " has no " + key);
    }
    return value;
  }

  /**
   * The model whose embeddings this index holds.
   *
   * @return the embedding model identifier recorded at build time
   */
  public String embeddingModelId() {
    return embeddingModelId;
  }

  /**
   * The width of the stored vectors.
   *
   * @return the embedding dimension
   */
  public int dimension() {
    return dimension;
  }

  /**
   * Task names present in this index.
   *
   * @return an immutable list of task names
   */
  public List<String> taskNames() {
    return List.copyOf(taskNames);
  }

  /**
   * Finds the task of the nearest indexed prompt.
   *
   * @param queryEmbedding the query vector, from the model named by {@link #embeddingModelId()}
   * @return the nearest match, or empty when the index is empty
   * @throws IllegalArgumentException if the query width does not match the index
   */
  public Optional<Match> nearest(float[] queryEmbedding) {
    Objects.requireNonNull(queryEmbedding, "queryEmbedding");
    if (queryEmbedding.length != dimension) {
      throw new IllegalArgumentException(
          "query has "
              + queryEmbedding.length
              + " dimensions but this index holds "
              + dimension
              + " from "
              + embeddingModelId
              + "; the query must come from the same embedding model");
    }
    SearchResult result =
        collection.search(
            SearchRequest.builder(queryEmbedding, 1)
                .includeMetadata(true)
                .includeVector(false)
                .build());
    if (result.hits().isEmpty()) {
      return Optional.empty();
    }
    SearchResult.Hit hit = result.hits().get(0);
    MetadataValue task = hit.document().metadata().get(TaskIndexBuilder.TASK_FIELD);
    if (!(task instanceof MetadataValue.Str label)) {
      return Optional.empty();
    }
    // SimilarityFunction.COSINE scores as (1 + cosine) / 2 over [0, 1]. Reported here as plain
    // cosine so a threshold reads the way cosine thresholds normally do: on that scale an
    // unrelated query scores near 0, where on the collection's it scores near 0.5.
    return Optional.of(new Match(label.value(), 2.0 * hit.score() - 1.0));
  }

  @Override
  public void close() {
    collection.close();
  }

  /**
   * The nearest indexed prompt's label and how close it was.
   *
   * @param task the task the nearest prompt belongs to
   * @param similarity plain cosine similarity to that prompt in [-1, 1], 1.0 being identical
   *     direction and 0.0 unrelated
   */
  public record Match(String task, double similarity) {

    /** Validates the label. */
    public Match {
      Objects.requireNonNull(task, "task");
    }
  }
}
