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
package com.integrallis.models.runtime.chat;

import com.integrallis.models.api.EmbeddingBackend;
import com.integrallis.models.api.ToolSpec;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;

/**
 * Embedding-backed retrieval over declared tool schemas.
 *
 * <p>This is the portable part of Needle's tool-routing path: keep a compact set of relevant tools
 * in the prompt instead of rendering every schema. The retriever owns no native resources; callers
 * own the supplied embedding backend lifecycle.
 */
public final class ToolSpecRetriever {

  private final EmbeddingBackend embeddings;
  private final List<Entry> entries;

  /** A ranked tool match. */
  public record Match(ToolSpec tool, float score) {
    public Match {
      Objects.requireNonNull(tool, "tool");
      if (!Float.isFinite(score)) {
        throw new IllegalArgumentException("score must be finite: " + score);
      }
    }
  }

  /** Embeds and indexes {@code tools} for later query-time selection. */
  public ToolSpecRetriever(EmbeddingBackend embeddings, List<ToolSpec> tools) {
    this.embeddings = Objects.requireNonNull(embeddings, "embeddings");
    List<ToolSpec> declared = List.copyOf(Objects.requireNonNull(tools, "tools"));
    if (declared.isEmpty()) {
      throw new IllegalArgumentException("tools must not be empty");
    }

    List<String> documents = declared.stream().map(ToolSpecRetriever::document).toList();
    float[][] vectors = embeddings.embedAll(documents);
    if (vectors.length != declared.size()) {
      throw new IllegalArgumentException(
          "embedding backend returned "
              + vectors.length
              + " rows for "
              + declared.size()
              + " tools");
    }
    List<Entry> indexed = new ArrayList<>(declared.size());
    for (int index = 0; index < declared.size(); index++) {
      indexed.add(
          new Entry(index, declared.get(index), normalize(vectors[index], "tool " + index)));
    }
    entries = List.copyOf(indexed);
  }

  /** Returns up to {@code limit} tools most similar to {@code query}. */
  public List<Match> select(String query, int limit) {
    Objects.requireNonNull(query, "query");
    if (limit <= 0) {
      throw new IllegalArgumentException("limit must be > 0: " + limit);
    }
    float[] queryVector = normalize(embeddings.embed(query), "query");
    return entries.stream()
        .map(entry -> new ScoredEntry(entry, dot(queryVector, entry.vector())))
        .sorted(
            Comparator.comparingDouble(ScoredEntry::score)
                .reversed()
                .thenComparingInt(scored -> scored.entry().index()))
        .limit(limit)
        .map(scored -> new Match(scored.entry().tool(), scored.score()))
        .toList();
  }

  private static String document(ToolSpec tool) {
    return tool.name() + "\n" + tool.description() + "\n" + tool.inputSchema();
  }

  private static float[] normalize(float[] vector, String label) {
    Objects.requireNonNull(vector, label);
    if (vector.length == 0) {
      throw new IllegalArgumentException(label + " embedding is empty");
    }
    double squaredNorm = 0.0d;
    for (float value : vector) {
      if (!Float.isFinite(value)) {
        throw new IllegalArgumentException(label + " embedding contains non-finite value");
      }
      squaredNorm += (double) value * value;
    }
    if (squaredNorm == 0.0d) {
      throw new IllegalArgumentException(label + " embedding has zero norm");
    }
    double norm = Math.sqrt(squaredNorm);
    float[] normalized = new float[vector.length];
    for (int index = 0; index < vector.length; index++) {
      normalized[index] = (float) (vector[index] / norm);
    }
    return normalized;
  }

  private static float dot(float[] left, float[] right) {
    if (left.length != right.length) {
      throw new IllegalArgumentException(
          "embedding dimensions differ: " + left.length + " != " + right.length);
    }
    float score = 0.0f;
    for (int index = 0; index < left.length; index++) {
      score += left[index] * right[index];
    }
    return score;
  }

  private record Entry(int index, ToolSpec tool, float[] vector) {}

  private record ScoredEntry(Entry entry, float score) {}
}
