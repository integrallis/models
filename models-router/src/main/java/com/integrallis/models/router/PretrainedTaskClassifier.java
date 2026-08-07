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

import java.util.List;
import java.util.Objects;

/**
 * Classifies a query into a task type by similarity to a pre-built index of labelled prompts.
 *
 * <p>A query takes the label of the single nearest indexed prompt. Nearest-neighbour rather than a
 * per-task centroid because a task is not one region of embedding space — "write a Dockerfile" and
 * "fix this segfault" are both {@code code} and sit far apart, and their midpoint is neither.
 *
 * <p>Pre-trained means the labelled corpus, the index built from it, and the decision threshold are
 * fixed and measured, not that weights ship as a binary. The corpus and its provenance live under
 * {@code models-router/corpus}; what ships is the index.
 *
 * <p>A query whose nearest neighbour is farther than the threshold classifies as {@code null}, and
 * {@link ModelRouter} then routes on cost, latency and reliability with quality averaged across
 * tasks. That is the right outcome for an unfamiliar request: guessing a task would silently apply
 * the wrong quality column.
 *
 * <pre>{@code
 * try (TaskIndex index = TaskIndex.open(indexDirectory)) {
 *   TaskClassifier classifier =
 *       PretrainedTaskClassifier.using(index, embeddingBackend::embed, minimumSimilarity);
 *   ModelRouter router = ModelRouter.builder().classifier(classifier).candidates(models).build();
 * }
 * }</pre>
 */
public final class PretrainedTaskClassifier implements TaskClassifier {

  private final TaskIndex index;
  private final TaskIndexBuilder.TaskEmbedder embedder;
  private final double minimumSimilarity;

  private PretrainedTaskClassifier(
      TaskIndex index, TaskIndexBuilder.TaskEmbedder embedder, double minimumSimilarity) {
    this.index = index;
    this.embedder = embedder;
    this.minimumSimilarity = minimumSimilarity;
  }

  /**
   * Builds a classifier over an opened index.
   *
   * <p>The embedder must be the model named by {@link TaskIndex#embeddingModelId()}. Embeddings are
   * only comparable within the model that produced them; a mismatch is not detectable from the
   * vectors themselves, only from their width, so supplying the wrong model of the same width
   * yields confident nonsense.
   *
   * @param index the index to search
   * @param embedder embeds queries with the index's embedding model
   * @param minimumSimilarity cosine similarity below which a query is left unclassified, in [-1, 1]
   * @return a classifier over that index
   * @throws IllegalArgumentException if the threshold is outside [-1, 1]
   */
  public static PretrainedTaskClassifier using(
      TaskIndex index, TaskIndexBuilder.TaskEmbedder embedder, double minimumSimilarity) {
    Objects.requireNonNull(index, "index");
    Objects.requireNonNull(embedder, "embedder");
    if (!(minimumSimilarity >= -1.0) || minimumSimilarity > 1.0) {
      throw new IllegalArgumentException("minimumSimilarity must be within [-1, 1]");
    }
    return new PretrainedTaskClassifier(index, embedder, minimumSimilarity);
  }

  @Override
  public String classify(String query) {
    if (query == null || query.isBlank()) {
      return null;
    }
    float[] embedding = embedder.embed(query);
    if (embedding == null || embedding.length == 0) {
      return null;
    }
    return index
        .nearest(embedding)
        .filter(match -> match.similarity() >= minimumSimilarity)
        .map(TaskIndex.Match::task)
        .orElse(null);
  }

  /**
   * Task names this classifier can return.
   *
   * @return an immutable list of task names
   */
  public List<String> taskNames() {
    return index.taskNames();
  }

  /**
   * The similarity a nearest neighbour must reach to be accepted.
   *
   * @return the configured floor
   */
  public double minimumSimilarity() {
    return minimumSimilarity;
  }
}
