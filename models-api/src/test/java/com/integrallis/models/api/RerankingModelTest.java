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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class RerankingModelTest {

  private final RerankingModel model = (query, document) -> document.length();

  @Test
  void scoresAllDocumentsInInputOrder() {
    assertThat(model.scoreAll("query", List.of("aaa", "b", "cc"))).containsExactly(3.0, 1.0, 2.0);
  }

  @Test
  void reranksByDescendingScoreWithStableTies() {
    assertThat(model.rerank("query", List.of("bbb", "a", "ccc", "dd")))
        .containsExactly(
            new RerankResult(0, "bbb", 3.0),
            new RerankResult(2, "ccc", 3.0),
            new RerankResult(3, "dd", 2.0),
            new RerankResult(1, "a", 1.0));
  }

  @Test
  void limitsTheRankedResultWithoutChangingOriginalIndexes() {
    assertThat(model.rerank("query", List.of("a", "longest", "medium"), 2))
        .containsExactly(new RerankResult(1, "longest", 7.0), new RerankResult(2, "medium", 6.0));
  }

  @Test
  void rejectsInvalidInputsBeforeEnteringTheModel() {
    assertThatThrownBy(() -> model.scoreAll(null, List.of("document")))
        .isInstanceOf(NullPointerException.class);
    assertThatThrownBy(() -> model.scoreAll("query", null))
        .isInstanceOf(NullPointerException.class);
    assertThatThrownBy(() -> model.scoreAll("query", java.util.Arrays.asList("document", null)))
        .isInstanceOf(NullPointerException.class);
    assertThatThrownBy(() -> model.rerank("query", List.of("document"), -1))
        .isInstanceOf(IllegalArgumentException.class);
  }
}
