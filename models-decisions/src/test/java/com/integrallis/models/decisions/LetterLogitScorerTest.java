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
import static org.assertj.core.api.Assertions.within;

import java.util.List;
import org.junit.jupiter.api.Test;

/** Reading the answer off the letter logits, and refusing to read it off anything else. */
final class LetterLogitScorerTest {

  private static final Choice ROUTE =
      new Choice("which desk", List.of("billing", "shipping", "technical"));

  @Test
  void rendersOptionsAsALetteredListEndingInTheAnswerCue() {
    String rendered = LetterLogitScorer.renderOptions(List.of("billing", "shipping"));
    assertThat(rendered).isEqualTo("A: billing\nB: shipping\nAnswer:");
  }

  @Test
  void normalisesOverTheLettersAlone() {
    // The vocabulary is dominated by a token that is not an answer. A scorer that softmaxed over
    // everything would return near-zero for every option; conditioning on answering must not.
    float[] logits = new float[100];
    logits[42] = 50f; // some unrelated token the model would rather emit
    logits[10] = 1f;
    logits[11] = 2f;
    logits[12] = 3f;

    Verdict verdict = new LetterLogitScorer(1.0).score(ROUTE, logits, new int[] {10, 11, 12});
    double[] p = verdict.probabilities();

    assertThat(p[0] + p[1] + p[2]).isCloseTo(1.0, within(1e-9));
    assertThat(p[2]).isGreaterThan(p[1]).isGreaterThan(0.0);
    assertThat(p[1]).isGreaterThan(p[0]);
  }

  @Test
  void permutingTheOptionsPermutesTheAnswer() {
    // Order equivariance: the mechanism must not prefer a position.
    float[] logits = new float[100];
    logits[10] = 1f;
    logits[11] = 2f;
    logits[12] = 3f;
    LetterLogitScorer scorer = new LetterLogitScorer(1.0);

    double[] straight = scorer.score(ROUTE, logits, new int[] {10, 11, 12}).probabilities();
    double[] reversed = scorer.score(ROUTE, logits, new int[] {12, 11, 10}).probabilities();

    assertThat(reversed[0]).isEqualTo(straight[2]);
    assertThat(reversed[2]).isEqualTo(straight[0]);
  }

  @Test
  void temperatureFlattensWithoutMovingTheWinner() {
    float[] logits = new float[100];
    logits[10] = 1f;
    logits[11] = 5f;
    logits[12] = 2f;
    double[] sharp =
        new LetterLogitScorer(1.0).score(ROUTE, logits, new int[] {10, 11, 12}).probabilities();
    double[] flat =
        new LetterLogitScorer(4.0).score(ROUTE, logits, new int[] {10, 11, 12}).probabilities();

    assertThat(sharp[1]).isGreaterThan(flat[1]);
    // Monotonic: temperature cannot change which option wins, only how sure the answer is.
    assertThat(flat[1]).isGreaterThan(flat[0]).isGreaterThan(0.0);
    assertThat(flat[1]).isGreaterThan(flat[2]);
  }

  @Test
  void refusesAMismatchBetweenLabelsAndLetters() {
    assertThatThrownBy(
            () -> new LetterLogitScorer(1.0).score(ROUTE, new float[100], new int[] {10, 11}))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("one letter token per label");
  }

  @Test
  void refusesATokenOutsideTheVocabulary() {
    assertThatThrownBy(
            () -> new LetterLogitScorer(1.0).score(ROUTE, new float[20], new int[] {10, 11, 99}))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("outside the vocabulary");
  }

  @Test
  void refusesMoreOptionsThanThereAreLetters() {
    assertThatThrownBy(
            () ->
                LetterLogitScorer.renderOptions(
                    java.util.stream.IntStream.range(0, 30).mapToObj(Integer::toString).toList()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("26");
  }
}
