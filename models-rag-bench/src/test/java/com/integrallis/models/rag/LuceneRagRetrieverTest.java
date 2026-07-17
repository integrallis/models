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
package com.integrallis.models.rag;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.charset.StandardCharsets;
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.HexFormat;
import org.junit.jupiter.api.Test;

class LuceneRagRetrieverTest {

  @Test
  void bm25RanksEveryAnswerableCaseSourceFirst() throws Exception {
    RagCorpus corpus = RagCorpus.loadDefault();

    try (LuceneRagRetriever retriever = new LuceneRagRetriever(corpus.documents())) {
      for (RagCase testCase : corpus.cases().stream().filter(RagCase::answerable).toList()) {
        assertThat(retriever.retrieve(testCase.question(), 3))
            .as(testCase.id())
            .extracting(hit -> hit.document().id())
            .first()
            .isEqualTo(testCase.relevantDocumentIds().getFirst());
      }
    }
  }

  @Test
  void everyTopThreePromptMatchesThePythonContract() throws Exception {
    RagCorpus corpus = RagCorpus.loadDefault();

    try (LuceneRagRetriever retriever = new LuceneRagRetriever(corpus.documents())) {
      String promptHashes =
          String.join(
              "\n",
              corpus.cases().stream()
                  .map(
                      testCase -> {
                        try {
                          return sha256(
                              RagPromptRenderer.render(
                                  testCase.question(), retriever.retrieve(testCase.question(), 1)));
                        } catch (java.io.IOException failure) {
                          throw new IllegalStateException(failure);
                        }
                      })
                  .toList());

      assertThat(sha256(promptHashes))
          .isEqualTo("21da412524218914dc0ac680a8d92ee16c77d4254e2a18cd372f5134c6ab4709");
    }
  }

  private static String sha256(String value) {
    try {
      return HexFormat.of()
          .formatHex(
              MessageDigest.getInstance("SHA-256").digest(value.getBytes(StandardCharsets.UTF_8)));
    } catch (NoSuchAlgorithmException impossible) {
      throw new IllegalStateException(impossible);
    }
  }
}
