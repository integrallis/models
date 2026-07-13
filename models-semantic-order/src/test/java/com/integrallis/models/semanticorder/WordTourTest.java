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

import java.io.ByteArrayInputStream;
import java.nio.charset.StandardCharsets;
import java.util.List;
import org.junit.jupiter.api.Test;

class WordTourTest {
  @Test
  void loadsRanksAndTermsFromUtf8Order() {
    WordTour tour = load("alpha\nbeta\ngamma\ndelta\n");

    assertThat(tour.size()).isEqualTo(4);
    assertThat(tour.rank("alpha")).hasValue(0);
    assertThat(tour.rank("gamma")).hasValue(2);
    assertThat(tour.rank("missing")).isEmpty();
    assertThat(tour.termAt(3)).isEqualTo("delta");
  }

  @Test
  void returnsNeighborsByIncreasingCyclicDistance() {
    WordTour tour = load("alpha\nbeta\ngamma\ndelta\nepsilon\n");

    assertThat(tour.neighbors("alpha", 2))
        .containsExactly(
            new SemanticNeighbor("epsilon", 4, -1),
            new SemanticNeighbor("beta", 1, 1),
            new SemanticNeighbor("delta", 3, -2),
            new SemanticNeighbor("gamma", 2, 2));
  }

  @Test
  void doesNotReturnDuplicateNeighborsWhenRadiusSpansTheCycle() {
    WordTour tour = load("alpha\nbeta\ngamma\ndelta\n");

    assertThat(tour.neighbors("alpha", 20))
        .extracting(SemanticNeighbor::term)
        .containsExactly("delta", "beta", "gamma");
  }

  @Test
  void computesShortestDistanceAcrossTheCycleBoundary() {
    WordTour tour = load("alpha\nbeta\ngamma\ndelta\nepsilon\n");

    assertThat(tour.cyclicDistance("alpha", "epsilon")).isEqualTo(1);
    assertThat(tour.cyclicDistance("beta", "epsilon")).isEqualTo(2);
    assertThat(tour.cyclicDistance("gamma", "gamma")).isZero();
  }

  @Test
  void validatesTermsRanksRadiiAndVocabulary() {
    WordTour tour = load("alpha\nbeta\ngamma\n");

    assertThatThrownBy(() -> tour.termAt(-1)).isInstanceOf(IndexOutOfBoundsException.class);
    assertThatThrownBy(() -> tour.termAt(3)).isInstanceOf(IndexOutOfBoundsException.class);
    assertThatThrownBy(() -> tour.neighbors("missing", 1))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("missing");
    assertThatThrownBy(() -> tour.neighbors("alpha", -1))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("radius");
    assertThatThrownBy(() -> tour.cyclicDistance("alpha", "missing"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("missing");
  }

  @Test
  void rejectsEmptyBlankOrDuplicateTours() {
    assertThatThrownBy(() -> load(""))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("empty");
    assertThatThrownBy(() -> load("alpha\n\nbeta\n"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("blank")
        .hasMessageContaining("2");
    assertThatThrownBy(() -> load("alpha\nbeta\nalpha\n"))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("Duplicate")
        .hasMessageContaining("alpha");
  }

  @Test
  void exposesAnImmutableTermSnapshot() {
    WordTour tour = WordTour.fromTerms(List.of("alpha", "beta", "gamma"));

    assertThat(tour.terms()).containsExactly("alpha", "beta", "gamma");
    assertThatThrownBy(() -> tour.terms().add("delta"))
        .isInstanceOf(UnsupportedOperationException.class);
  }

  private static WordTour load(String value) {
    return WordTour.load(new ByteArrayInputStream(value.getBytes(StandardCharsets.UTF_8)));
  }
}
