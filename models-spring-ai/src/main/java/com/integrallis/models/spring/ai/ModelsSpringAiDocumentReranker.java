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
package com.integrallis.models.spring.ai;

import com.integrallis.models.api.RerankResult;
import com.integrallis.models.api.RerankingModel;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;
import org.springframework.ai.rag.postretrieval.document.DocumentPostProcessor;

/** Spring AI post-retrieval reranker backed by an in-process Models cross-encoder. */
public final class ModelsSpringAiDocumentReranker implements DocumentPostProcessor, AutoCloseable {

  private final RerankingModel model;
  private final int limit;
  private final Object modelMonitor = new Object();

  /** Keeps every document after reranking. */
  public ModelsSpringAiDocumentReranker(RerankingModel model) {
    this(model, Integer.MAX_VALUE);
  }

  /** Keeps at most {@code limit} documents after reranking. */
  public ModelsSpringAiDocumentReranker(RerankingModel model, int limit) {
    this.model = Objects.requireNonNull(model, "model");
    if (limit < 0) {
      throw new IllegalArgumentException("limit must be >= 0: " + limit);
    }
    this.limit = limit;
  }

  @Override
  public List<Document> process(Query query, List<Document> documents) {
    Objects.requireNonNull(query, "query");
    Objects.requireNonNull(documents, "documents");
    List<String> texts =
        documents.stream()
            .map(
                document -> {
                  Document value =
                      Objects.requireNonNull(document, "documents must not contain null");
                  if (!value.isText()) {
                    throw new IllegalArgumentException("reranking requires text documents");
                  }
                  return value.getText();
                })
            .toList();
    List<RerankResult> ranked;
    synchronized (modelMonitor) {
      ranked = model.rerank(query.text(), texts, Math.min(limit, texts.size()));
    }
    List<Document> result = new ArrayList<>(ranked.size());
    for (RerankResult item : ranked) {
      result.add(documents.get(item.originalIndex()).mutate().score(item.score()).build());
    }
    return List.copyOf(result);
  }

  /** Releases the mapped reranking model. */
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
