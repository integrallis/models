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

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Objects;

/**
 * Cross-encoder model that scores a document in the context of a query.
 *
 * <p>Unlike an embedding model, a reranker sees the query and document together. Its scores are
 * meaningful for ordering documents for the same query; callers should not assume that scores from
 * different queries share a calibrated scale.
 */
@FunctionalInterface
public interface RerankingModel extends AutoCloseable {

  /** Returns this model's relevance score for one query/document pair. */
  double score(String query, String document);

  /** Scores documents in caller order. */
  default List<Double> scoreAll(String query, List<String> documents) {
    Objects.requireNonNull(query, "query");
    Objects.requireNonNull(documents, "documents");
    List<Double> scores = new ArrayList<>(documents.size());
    for (String document : documents) {
      double score =
          score(query, Objects.requireNonNull(document, "documents must not contain null"));
      if (!Double.isFinite(score)) {
        throw new IllegalStateException("reranking model returned a non-finite score: " + score);
      }
      scores.add(score);
    }
    return List.copyOf(scores);
  }

  /** Ranks every document by descending score, retaining original positions for correlation. */
  default List<RerankResult> rerank(String query, List<String> documents) {
    return rerank(query, documents, Objects.requireNonNull(documents, "documents").size());
  }

  /**
   * Returns at most {@code limit} documents by descending score. Equal scores retain caller order.
   */
  default List<RerankResult> rerank(String query, List<String> documents, int limit) {
    Objects.requireNonNull(query, "query");
    Objects.requireNonNull(documents, "documents");
    if (limit < 0) {
      throw new IllegalArgumentException("limit must be >= 0: " + limit);
    }
    List<Double> scores = scoreAll(query, documents);
    List<RerankResult> ranked = new ArrayList<>(documents.size());
    for (int index = 0; index < documents.size(); index++) {
      ranked.add(new RerankResult(index, documents.get(index), scores.get(index)));
    }
    ranked.sort(
        Comparator.comparingDouble(RerankResult::score)
            .reversed()
            .thenComparingInt(RerankResult::originalIndex));
    return List.copyOf(ranked.subList(0, Math.min(limit, ranked.size())));
  }

  /** Default lifecycle is a no-op for models without owned mapped or native state. */
  @Override
  default void close() {}
}
