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

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Scoring candidates independently against one shared state, then normalising within the question.
 *
 * <p>This is the mechanism that covers all three primitives with one code path: a Noul is two
 * candidates, a Choice is up to 255, a Score is its ordered levels. It is also what buys
 * order-equivariance structurally rather than by sorting inputs — each candidate is scored without
 * reference to the others, so permuting them permutes the output and changes nothing else.
 *
 * <p>The open rebuild that established this design reports 100% consistency under candidate
 * reordering against 68.75% for a letter-probability readout, and zero probability difference
 * between asking questions singly and together. These tests hold us to the same properties.
 */
@Tag("unit")
class CandidateScorerTest {

  /** Scores a candidate by how strongly the state's hidden state aligns with the candidate's. */
  private static CandidateScorer.Alignment alignment() {
    return (state, candidate) -> {
      double dot = 0.0;
      for (int i = 0; i < state.length; i++) {
        dot += state[i] * candidate[i];
      }
      return dot;
    };
  }

  private static CandidateScorer scorer() {
    return new CandidateScorer(alignment(), 1.0);
  }

  @Test
  void permutingTheCandidatesPermutesTheAnswerAndNothingElse() {
    float[] state = {1.0f, 0.0f};
    List<String> labels = List.of("alpha", "beta", "gamma");
    List<float[]> vectors = List.of(v(3, 0), v(1, 0), v(2, 0));

    Verdict direct = scorer().score(new Choice("q", labels), state, vectors);

    List<String> reordered = List.of("gamma", "alpha", "beta");
    List<float[]> reorderedVectors = List.of(v(2, 0), v(3, 0), v(1, 0));
    Verdict permuted = scorer().score(new Choice("q", reordered), state, reorderedVectors);

    assertThat(direct.winner()).isEqualTo("alpha");
    assertThat(permuted.winner()).isEqualTo("alpha");
    assertThat(permuted.probabilityOf("alpha")).isEqualTo(direct.probabilityOf("alpha"));
    assertThat(permuted.probabilityOf("beta")).isEqualTo(direct.probabilityOf("beta"));
  }

  @Test
  void aCandidateIsScoredWithoutReferenceToTheOthers() {
    // Removing a losing candidate must not change the relative standing of those that remain.
    float[] state = {1.0f, 0.0f};
    Verdict three =
        scorer()
            .score(
                new Choice("q", List.of("a", "b", "c")), state, List.of(v(3, 0), v(2, 0), v(1, 0)));
    Verdict two =
        scorer().score(new Choice("q", List.of("a", "b")), state, List.of(v(3, 0), v(2, 0)));

    double ratioThree = three.probabilityOf("a") / three.probabilityOf("b");
    double ratioTwo = two.probabilityOf("a") / two.probabilityOf("b");

    assertThat(ratioTwo).isCloseTo(ratioThree, within(1e-9));
  }

  @Test
  void theDistributionIsProperAcrossTwoHundredAndFiftyFiveCandidates() {
    List<String> labels = new ArrayList<>();
    List<float[]> vectors = new ArrayList<>();
    for (int i = 0; i < 255; i++) {
      labels.add("option-" + i);
      vectors.add(v(i / 100.0, 0));
    }
    Verdict verdict = scorer().score(new Choice("wide", labels), new float[] {1.0f, 0.0f}, vectors);

    double total = 0.0;
    for (double p : verdict.probabilities()) {
      total += p;
      assertThat(p).isBetween(0.0, 1.0);
    }
    assertThat(total).isCloseTo(1.0, within(1e-9));
    assertThat(labels).contains(verdict.winner());
  }

  @Test
  void aNoulIsJustTwoCandidates() {
    Noul space = new Noul("is it so");
    Verdict verdict = scorer().score(space, new float[] {1.0f, 0.0f}, List.of(v(0, 0), v(2, 0)));

    assertThat(verdict.probabilityOfTrue()).isGreaterThan(0.5);
    assertThat(verdict.winner()).isEqualTo("true");
  }

  @Test
  void aScoreIsJustItsOrderedLevels() {
    Score space = new Score("severity", List.of("0", "1", "2", "3"));
    Verdict verdict =
        scorer()
            .score(space, new float[] {1.0f, 0.0f}, List.of(v(0, 0), v(1, 0), v(4, 0), v(1, 0)));

    assertThat(verdict.winner()).isEqualTo("2");
    assertThat(verdict.probabilities()).hasSize(4);
  }

  @Test
  void oneCandidateVectorPerLabelIsRequired() {
    assertThatThrownBy(
            () ->
                scorer()
                    .score(
                        new Choice("q", List.of("a", "b")), new float[] {1.0f}, List.of(v(1, 0))))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void theTemperatureFlattensWithoutMovingTheWinner() {
    Choice space = new Choice("q", List.of("a", "b"));
    float[] state = {1.0f, 0.0f};
    List<float[]> vectors = List.of(v(3, 0), v(1, 0));

    Verdict sharp = new CandidateScorer(alignment(), 0.5).score(space, state, vectors);
    Verdict flat = new CandidateScorer(alignment(), 8.0).score(space, state, vectors);

    assertThat(sharp.winner()).isEqualTo("a");
    assertThat(flat.winner()).isEqualTo("a");
    assertThat(flat.probabilityOf("a")).isLessThan(sharp.probabilityOf("a"));
  }

  @Test
  void scoringTheSameInputTwiceIsBitIdentical() {
    Choice space = new Choice("q", List.of("a", "b", "c"));
    float[] state = {0.3f, 0.7f};
    List<float[]> vectors = List.of(v(1, 2), v(0.5, 0.1), v(2, 0.3));

    assertThat(scorer().score(space, state, vectors).probabilities())
        .isEqualTo(scorer().score(space, state, vectors).probabilities());
  }

  private static float[] v(double a, double b) {
    return new float[] {(float) a, (float) b};
  }
}
