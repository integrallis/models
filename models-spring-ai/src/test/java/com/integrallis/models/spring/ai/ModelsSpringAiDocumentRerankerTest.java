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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.api.RerankingModel;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;

@Tag("unit")
class ModelsSpringAiDocumentRerankerTest {

  @Test
  void reranksDocumentsAndPreservesTheirIdentityAndMetadata() {
    ScriptedModel backend = new ScriptedModel();
    var reranker = new ModelsSpringAiDocumentReranker(backend, 2);
    Document first =
        Document.builder().id("first").text("short").metadata(Map.of("line", "red")).build();
    Document second =
        Document.builder()
            .id("second")
            .text("the longest document")
            .metadata("line", "blue")
            .build();
    Document third = Document.builder().id("third").text("medium length").build();

    List<Document> ranked =
        reranker.process(new Query("transit query"), List.of(first, second, third));

    assertThat(ranked).extracting(Document::getId).containsExactly("second", "third");
    assertThat(ranked).extracting(Document::getScore).containsExactly(20.0, 13.0);
    assertThat(ranked.get(0).getMetadata()).containsEntry("line", "blue");
    assertThat(backend.seen)
        .containsExactly(
            "transit query|short",
            "transit query|the longest document",
            "transit query|medium length");
  }

  @Test
  void defaultsToReturningEveryDocumentAndKeepsStableTies() {
    var reranker = new ModelsSpringAiDocumentReranker(new ScriptedModel());
    Document first = Document.builder().id("first").text("same").build();
    Document second = Document.builder().id("second").text("size").build();

    assertThat(reranker.process(new Query("query"), List.of(first, second)))
        .extracting(Document::getId)
        .containsExactly("first", "second");
  }

  @Test
  void closesTheOwnedModelAndRejectsInvalidConfiguration() {
    ScriptedModel backend = new ScriptedModel();
    var reranker = new ModelsSpringAiDocumentReranker(backend);

    reranker.close();

    assertThat(backend.closed).isTrue();
    assertThatThrownBy(() -> new ModelsSpringAiDocumentReranker(null))
        .isInstanceOf(NullPointerException.class);
    assertThatThrownBy(() -> new ModelsSpringAiDocumentReranker(new ScriptedModel(), -1))
        .isInstanceOf(IllegalArgumentException.class);
  }

  private static final class ScriptedModel implements RerankingModel {
    private final List<String> seen = new ArrayList<>();
    private boolean closed;

    @Override
    public double score(String query, String document) {
      seen.add(query + "|" + document);
      return document.length();
    }

    @Override
    public void close() {
      closed = true;
    }
  }
}
