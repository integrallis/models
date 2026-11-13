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
package com.integrallis.models.tinystruct;

import com.integrallis.models.api.RerankResult;
import com.integrallis.models.api.RerankingModel;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import org.tinystruct.data.component.Builder;
import org.tinystruct.data.component.Builders;

/**
 * In-process cross-encoder document reranker for tinystruct applications backed by a {@link
 * RerankingModel}.
 */
public final class ModelsTinystructReranker implements AutoCloseable {

  private final RerankingModel model;
  private final int limit;
  private final Object modelMonitor = new Object();

  /** Keeps every document after reranking. */
  public ModelsTinystructReranker(RerankingModel model) {
    this(model, Integer.MAX_VALUE);
  }

  /** Keeps at most {@code limit} documents after reranking. */
  public ModelsTinystructReranker(RerankingModel model, int limit) {
    this.model = Objects.requireNonNull(model, "model");
    if (limit < 0) {
      throw new IllegalArgumentException("limit must be >= 0: " + limit);
    }
    this.limit = limit;
  }

  /**
   * Reranks the candidate document strings against {@code query} and returns a {@link Builders}
   * list ordered by descending relevance score.
   *
   * @param query the search query
   * @param documents candidate document texts
   * @return ranked {@link Builders} entries with {@code original_index}, {@code text}, and {@code
   *     score}
   */
  public Builders rerank(String query, List<String> documents) {
    Objects.requireNonNull(query, "query");
    Objects.requireNonNull(documents, "documents");
    for (String document : documents) {
      Objects.requireNonNull(document, "documents must not contain null");
    }
    List<RerankResult> ranked;
    synchronized (modelMonitor) {
      ranked = model.rerank(query, documents, Math.min(limit, documents.size()));
    }
    Builders results = new Builders();
    for (RerankResult item : ranked) {
      Builder row = new Builder();
      row.put("original_index", item.originalIndex());
      row.put("text", item.document());
      row.put("score", item.score());
      results.add(row);
    }
    return results;
  }

  /**
   * Reranks a tinystruct {@link Builders} collection of document {@link Builder} objects using the
   * text stored under {@code textField} and returns copies annotated with {@code score} and {@code
   * original_index}.
   *
   * @param query the search query
   * @param documents the input document builders
   * @param textField the key holding each document's text
   * @return ranked {@link Builders} of copied {@link Builder} entries
   */
  public Builders rerankDocuments(String query, Builders documents, String textField) {
    Objects.requireNonNull(query, "query");
    Objects.requireNonNull(documents, "documents");
    Objects.requireNonNull(textField, "textField");
    if (textField.isBlank()) {
      throw new IllegalArgumentException("textField must not be blank");
    }
    List<String> texts = new ArrayList<>(documents.size());
    for (Builder document : documents) {
      Objects.requireNonNull(document, "documents must not contain null");
      Object value = document.get(textField);
      if (value == null) {
        throw new IllegalArgumentException(
            "document is missing required text field '" + textField + "'");
      }
      texts.add(value.toString());
    }
    List<RerankResult> ranked;
    synchronized (modelMonitor) {
      ranked = model.rerank(query, texts, Math.min(limit, texts.size()));
    }
    Builders results = new Builders();
    for (RerankResult item : ranked) {
      Builder original = documents.get(item.originalIndex());
      Builder copy = new Builder();
      for (Map.Entry<String, Object> entry : original.entrySet()) {
        copy.put(entry.getKey(), entry.getValue());
      }
      copy.put("original_index", item.originalIndex());
      copy.put("score", item.score());
      results.add(copy);
    }
    return results;
  }

  @Override
  public void close() {
    synchronized (modelMonitor) {
      try {
        model.close();
      } catch (RuntimeException failure) {
        throw failure;
      } catch (Exception failure) {
        throw new IllegalStateException("failed to close reranking model", failure);
      }
    }
  }
}
