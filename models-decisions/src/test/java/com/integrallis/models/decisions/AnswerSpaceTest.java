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
package com.integrallis.models.decisions;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.stream.IntStream;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The answer space is the whole type-safety guarantee: whatever a head computes, the outcome has to
 * be one of these labels. These tests pin the bounds that make that guarantee cheap to trust.
 */
@Tag("unit")
class AnswerSpaceTest {

  @Test
  void aNoulOffersFalseThenTrue() {
    Noul noul = new Noul("the passage answers the question");

    assertThat(noul.size()).isEqualTo(2);
    assertThat(noul.labels()).containsExactly("false", "true");
    assertThat(noul.trueIndex()).isEqualTo(1);
  }

  @Test
  void aNoulRequiresAProposition() {
    assertThatThrownBy(() -> new Noul("  ")).isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void aChoiceHoldsBetweenTwoAndTwoHundredFiftyFiveOptions() {
    assertThat(new Choice("route", List.of("chat", "tools")).size()).isEqualTo(2);
    assertThat(new Choice("route", labels(255)).size()).isEqualTo(255);

    assertThatThrownBy(() -> new Choice("route", List.of("only")))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> new Choice("route", labels(256)))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void aChoiceRejectsRepeatedOrBlankOptions() {
    assertThatThrownBy(() -> new Choice("route", List.of("chat", "chat")))
        .isInstanceOf(IllegalArgumentException.class);
    assertThatThrownBy(() -> new Choice("route", List.of("chat", " ")))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void aScoreHoldsBetweenTwoAndTenOrderedLevels() {
    assertThat(new Score("severity", List.of("low", "high")).size()).isEqualTo(2);
    assertThat(new Score("severity", labels(10)).size()).isEqualTo(10);

    assertThatThrownBy(() -> new Score("severity", labels(11)))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void anAnswerSpaceCannotBeMutatedThroughTheListItHandsBack() {
    Choice choice = new Choice("route", List.of("chat", "tools"));
    List<String> labels = choice.labels();

    assertThatThrownBy(() -> labels.add("smuggled"))
        .isInstanceOf(UnsupportedOperationException.class);
  }

  @Test
  void anAnswerSpaceCopiesTheOptionsItWasGiven() {
    List<String> mutable = new ArrayList<>(List.of("chat", "tools"));
    Choice choice = new Choice("route", mutable);

    Collections.reverse(mutable);

    assertThat(choice.labels()).containsExactly("chat", "tools");
  }

  private static List<String> labels(int count) {
    return IntStream.range(0, count).mapToObj(index -> "option-" + index).toList();
  }
}
