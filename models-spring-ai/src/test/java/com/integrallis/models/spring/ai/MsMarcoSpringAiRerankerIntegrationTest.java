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
import static org.assertj.core.api.Assertions.within;

import com.integrallis.models.backend.purejava.GgufRerankingModel;
import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.document.Document;
import org.springframework.ai.rag.Query;

@Tag("integration")
class MsMarcoSpringAiRerankerIntegrationTest {

  @Test
  void documentPostProcessorReturnsTheHighestScoringRealDocument() {
    try (var reranker =
        new ModelsSpringAiDocumentReranker(
            GgufRerankingModel.load(Path.of(System.getProperty("models.fixtures.msMarcoReranker"))),
            1)) {
      Document relevant =
          Document.builder()
              .id("berlin")
              .text(
                  "Berlin has a population of 3,520,031 registered inhabitants in an area of 891.82 square kilometers.")
              .build();
      Document irrelevant =
          Document.builder()
              .id("cats")
              .text("Domestic cats sleep for a large part of the day.")
              .build();

      List<Document> result =
          reranker.process(
              new Query("How many people live in Berlin?"), List.of(irrelevant, relevant));

      assertThat(result).singleElement().extracting(Document::getId).isEqualTo("berlin");
      assertThat(result.get(0).getScore()).isCloseTo(8.846, within(0.15));
    }
  }
}
