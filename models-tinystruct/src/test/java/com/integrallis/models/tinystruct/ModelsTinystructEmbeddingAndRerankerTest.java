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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.api.EmbeddingBackend;
import com.integrallis.models.api.RerankResult;
import com.integrallis.models.api.RerankingModel;
import java.util.List;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.tinystruct.data.component.Builder;
import org.tinystruct.data.component.Builders;

@Tag("unit")
class ModelsTinystructEmbeddingAndRerankerTest {

  @Test
  void embedsSingleAndBatchTextsIntoBuildersAndClosesBackend() {
    AtomicBoolean closed = new AtomicBoolean();
    EmbeddingBackend backend =
        new EmbeddingBackend() {
          @Override
          public float[] embed(String text) {
            return new float[] {text.length(), 1.0f, -0.5f};
          }

          @Override
          public int dimension() {
            return 3;
          }

          @Override
          public void close() {
            closed.set(true);
          }
        };

    try (ModelsTinystructEmbeddingModel model =
        new ModelsTinystructEmbeddingModel(backend, "minilm-local")) {
      assertThat(model.modelName()).isEqualTo("minilm-local");
      assertThat(model.dimensions()).isEqualTo(3);
      assertThat(model.embed("abc")).containsExactly(3.0f, 1.0f, -0.5f);

      Builders batch = model.embedAll(List.of("hi", "world"));
      assertThat(batch).hasSize(2);
      assertThat(batch.get(0).get("index")).isEqualTo(0);
      assertThat(batch.get(0).get("model")).isEqualTo("minilm-local");
      assertThat(batch.get(0).get("dimensions")).isEqualTo(3);
      assertThat((double[]) batch.get(1).get("embedding")).containsExactly(5.0, 1.0, -0.5);
    }
    assertThat(closed).isTrue();

    assertThatThrownBy(() -> new ModelsTinystructEmbeddingModel(backend, " "))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("modelName must not be blank");
  }

  @Test
  void reranksPlainStringsAndDocumentBuildersAndHandlesCloseFailures() {
    AtomicBoolean closed = new AtomicBoolean();
    RerankingModel rerankingModel =
        new RerankingModel() {
          @Override
          public double score(String query, String document) {
            return document.contains(query) ? 0.95 : 0.10;
          }

          @Override
          public List<RerankResult> rerank(String query, List<String> documents, int topK) {
            return List.of(
                    new RerankResult(1, documents.get(1), 0.95),
                    new RerankResult(0, documents.get(0), 0.10))
                .subList(0, Math.min(topK, documents.size()));
          }

          @Override
          public void close() {
            closed.set(true);
          }
        };

    try (ModelsTinystructReranker reranker = new ModelsTinystructReranker(rerankingModel, 1)) {
      Builders rankedStrings = reranker.rerank("java", List.of("python guide", "java 25 vector"));
      assertThat(rankedStrings).hasSize(1);
      assertThat(rankedStrings.get(0).get("original_index")).isEqualTo(1);
      assertThat(rankedStrings.get(0).get("text")).isEqualTo("java 25 vector");
      assertThat(rankedStrings.get(0).get("score")).isEqualTo(0.95);

      Builders docs = new Builders();
      Builder doc0 = new Builder();
      doc0.put("id", "d0");
      doc0.put("body", "python guide");
      Builder doc1 = new Builder();
      doc1.put("id", "d1");
      doc1.put("body", "java 25 vector");
      docs.add(doc0);
      docs.add(doc1);

      Builders rankedDocs = reranker.rerankDocuments("java", docs, "body");
      assertThat(rankedDocs).hasSize(1);
      assertThat(rankedDocs.get(0).get("id")).isEqualTo("d1");
      assertThat(rankedDocs.get(0).get("original_index")).isEqualTo(1);
      assertThat(rankedDocs.get(0).get("score")).isEqualTo(0.95);

      assertThatThrownBy(() -> reranker.rerankDocuments("java", docs, "missing_field"))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("missing_field");
      assertThatThrownBy(() -> reranker.rerankDocuments("java", docs, "  "))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("textField must not be blank");
    }
    assertThat(closed).isTrue();

    assertThatThrownBy(() -> new ModelsTinystructReranker(rerankingModel, -1))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("limit must be >= 0");
  }
}
