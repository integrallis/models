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
package com.integrallis.models.semanticorder;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Test;
import org.modeljars.ModelJarException;
import org.modeljars.ModelJarRequirement;

class WordTourModelJarsTest {
  private static final ModelJarRequirement WORD_TOUR =
      ModelJarRequirement.forSource("github://joisino/wordtour")
          .versionRange("[1.0.0,2.0.0)")
          .variant("optimal")
          .backend("semantic-order")
          .capability("semantic-neighbors")
          .build();

  @Test
  void resolvesVerifiesAndLoadsCanonicalWordTourFromModelJars() {
    WordTour tour = WordTour.load(WORD_TOUR);

    assertThat(tour.size()).isEqualTo(40_000);
    assertThat(tour.rank("the")).hasValue(0);
    assertThat(tour.rank("of")).hasValue(39_999);
    assertThat(tour.neighbors("the", 1))
        .containsExactly(new SemanticNeighbor("of", 39_999, -1), new SemanticNeighbor("its", 1, 1));
  }

  @Test
  void reportsAnUnresolvedModelJarRequirement() {
    ModelJarRequirement missing =
        ModelJarRequirement.forSource("github://example/missing").backend("semantic-order").build();

    assertThatThrownBy(() -> WordTour.load(missing))
        .isInstanceOf(ModelJarException.class)
        .hasMessageContaining("No ModelJars descriptor matched");
  }
}
