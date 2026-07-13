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

import java.util.List;
import org.junit.jupiter.api.Test;

class BlurredBagOfWordsTest {
  private static final WordTour TOUR =
      WordTour.fromTerms(List.of("alpha", "beta", "gamma", "delta", "epsilon"));

  @Test
  void appliesTheReferenceGaussianBlurAndL1Normalization() {
    BlurredBagOfWords bow = BlurredBagOfWords.encode(TOUR, List.of("alpha"), 1, 1.0);
    double neighborWeight = Math.exp(-1.0);
    double total = 1.0 + 2.0 * neighborWeight;

    assertThat(bow.nonZeroCount()).isEqualTo(3);
    assertThat(bow.weightAtRank(0)).isCloseTo(1.0 / total, within(1.0e-12));
    assertThat(bow.weightAtRank(1)).isCloseTo(neighborWeight / total, within(1.0e-12));
    assertThat(bow.weightAtRank(4)).isCloseTo(neighborWeight / total, within(1.0e-12));
    assertThat(bow.l1Norm()).isCloseTo(1.0, within(1.0e-12));
  }

  @Test
  void accumulatesRepeatedTermsAndCollidingCyclicBlurPositions() {
    WordTour smallTour = WordTour.fromTerms(List.of("alpha", "beta", "gamma"));

    BlurredBagOfWords bow = BlurredBagOfWords.encode(smallTour, List.of("alpha", "alpha"), 2, 1.0);

    assertThat(bow.nonZeroCount()).isEqualTo(3);
    assertThat(bow.l1Norm()).isCloseTo(1.0, within(1.0e-12));
    assertThat(bow.weightAtRank(0)).isGreaterThan(bow.weightAtRank(1));
    assertThat(bow.weightAtRank(1)).isEqualTo(bow.weightAtRank(2));
  }

  @Test
  void ignoresOutOfVocabularyTermsWithoutInventingDimensions() {
    BlurredBagOfWords bow =
        BlurredBagOfWords.encode(TOUR, List.of("missing", "alpha", "unknown"), 0, 10.0);
    BlurredBagOfWords empty = BlurredBagOfWords.encode(TOUR, List.of("missing"), 10, 10.0);

    assertThat(bow.nonZeroCount()).isOne();
    assertThat(bow.weightAtRank(0)).isEqualTo(1.0);
    assertThat(empty.nonZeroCount()).isZero();
    assertThat(empty.l1Norm()).isZero();
  }

  @Test
  void computesSparseL1Distance() {
    BlurredBagOfWords alpha = BlurredBagOfWords.encode(TOUR, List.of("alpha"), 0, 10.0);
    BlurredBagOfWords beta = BlurredBagOfWords.encode(TOUR, List.of("beta"), 0, 10.0);
    BlurredBagOfWords empty = BlurredBagOfWords.encode(TOUR, List.of(), 0, 10.0);

    assertThat(alpha.l1Distance(alpha)).isZero();
    assertThat(alpha.l1Distance(beta)).isEqualTo(2.0);
    assertThat(alpha.l1Distance(empty)).isEqualTo(1.0);
    assertThat(empty.l1Distance(empty)).isZero();
  }

  @Test
  void rejectsIncompatibleOrdersAndInvalidHyperparameters() {
    BlurredBagOfWords alpha = BlurredBagOfWords.encode(TOUR, List.of("alpha"), 0, 10.0);
    WordTour otherTour = WordTour.fromTerms(List.of("alpha", "beta", "delta"));
    BlurredBagOfWords other = BlurredBagOfWords.encode(otherTour, List.of("alpha"), 0, 10.0);

    assertThatThrownBy(() -> BlurredBagOfWords.encode(TOUR, List.of("alpha"), -1, 1.0))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("radius");
    assertThatThrownBy(() -> BlurredBagOfWords.encode(TOUR, List.of("alpha"), 1, 0.0))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("sigma");
    assertThatThrownBy(() -> alpha.l1Distance(other))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("semantic order");
  }

  private static org.assertj.core.data.Offset<Double> within(double value) {
    return org.assertj.core.data.Offset.offset(value);
  }
}
