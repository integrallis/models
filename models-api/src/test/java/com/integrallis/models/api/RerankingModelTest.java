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

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import org.junit.jupiter.api.Test;

/** Unit coverage for {@link RerankingModel} default batch/sequential routing (#440). */
class RerankingModelTest {

  private static final RerankingModel DUMMY =
      new RerankingModel() {
        @Override
        public double score(String query, String document) {
          return query.length() * 1.0 + document.length();
        }

        @Override
        public List<Double> scoreBatch(String query, List<String> documents) {
          return documents.stream().map(d -> score(query, d)).toList();
        }
      };

  @Test
  void scoreAllRoutesToScoreBatch() {
    List<Double> scores = DUMMY.scoreAll("q", List.of("a", "bb", "ccc"));
    assertThat(scores).containsExactly(2.0, 3.0, 4.0);
  }

  @Test
  void scoreBatchMatchesIndividualScores() {
    List<Double> batch = DUMMY.scoreBatch("hello", List.of("a", "bb"));
    assertThat(batch).containsExactly(DUMMY.score("hello", "a"), DUMMY.score("hello", "bb"));
  }

  @Test
  void emptyDocumentsReturnsEmptyList() {
    assertThat(DUMMY.scoreAll("q", List.of())).isEmpty();
    assertThat(DUMMY.scoreBatch("q", List.of())).isEmpty();
  }

  @Test
  void rerankRetainsOriginalIndices() {
    var ranked = DUMMY.rerank("q", List.of("ccc", "a", "bb"));
    assertThat(ranked).hasSize(3);
    assertThat(ranked.get(0).originalIndex()).isEqualTo(0); // "ccc" (longest)
    assertThat(ranked.get(1).originalIndex()).isEqualTo(2); // "bb"
    assertThat(ranked.get(2).originalIndex()).isEqualTo(1); // "a"
  }
}
