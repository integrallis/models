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
package com.integrallis.models.audio;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class SopranoTextChunkerTest {

  @Test
  void preservesAShortTrimmedUtterance() {
    assertThat(SopranoTextChunker.split(" \tHello   JVM!\n", 200)).containsExactly("Hello   JVM!");
  }

  @Test
  void prefersTheLastSentenceBoundaryInsideTheBudget() {
    assertThat(SopranoTextChunker.split("One two. Three four five", 19))
        .containsExactly("One two.", "Three four five");
  }

  @Test
  void countsUnicodeCodePointsRatherThanUtf16CodeUnits() {
    assertThat(SopranoTextChunker.split("A 😀 B C", 5)).containsExactly("A 😀 B", "C");
  }

  @Test
  void keepsAnOverBudgetWordWholeLikeTheReferenceChunker() {
    assertThat(SopranoTextChunker.split("extraordinary ok", 5))
        .containsExactly("extraordinary", "ok");
  }

  @Test
  void rejectsANonPositiveBudget() {
    assertThatThrownBy(() -> SopranoTextChunker.split("hello", 0))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("positive");
  }
}
