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
import java.util.Random;
import org.junit.jupiter.api.Test;

/** A typed head has to earn its accuracy against its own floor, and say so once. */
final class TypedEvaluationTest {

  private static final Choice ROUTE =
      new Choice("which desk", List.of("billing", "shipping", "technical", "other"));
  private static final Score SEVERITY = new Score("how severe", List.of("0", "1", "2", "3", "4"));

  @Test
  void learnsASeparableChoiceAndBeatsItsFloor() {
    TypedReport report = evaluate(ROUTE, separable(ROUTE.size(), 600, 0.0));

    assertThat(report.outcomes()).isEqualTo(4);
    assertThat(report.sealedSize()).isGreaterThan(0);
    assertThat(report.accuracy()).isGreaterThan(0.9);
    assertThat(report.beatsFloor(0.05)).isTrue();
    // An unordered space has no notion of a near miss, and reporting one would invent a metric.
    assertThat(report.ordinalMeanAbsoluteError()).isNaN();
  }

  @Test
  void reportsTheOrdinalErrorForAScore() {
    TypedReport report = evaluate(SEVERITY, separable(SEVERITY.size(), 750, 0.0));

    assertThat(report.ordinalMeanAbsoluteError()).isNotNaN();
    assertThat(report.ordinalMeanAbsoluteError()).isLessThan(0.5);
  }

  @Test
  void aStateThatCarriesNoSignalDoesNotBeatTheFloor() {
    // The condition that separates a weak head from no head at all. Without this, a head that has
    // learned the base rate and nothing else reads as a modest success.
    TypedReport report = evaluate(ROUTE, noise(ROUTE.size(), 600));

    assertThat(report.beatsFloor(0.05)).isFalse();
    assertThat(report.accuracy()).isCloseTo(report.majorityFloor(), within(0.15));
  }

  @Test
  void theSealedSplitIsReadOnlyOnce() {
    TypedEvaluation evaluation = evaluation(ROUTE, separable(ROUTE.size(), 400, 0.0));
    evaluation.scoreSealedOnce();

    assertThatThrownBy(evaluation::scoreSealedOnce)
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("read once");
  }

  @Test
  void theArtifactIsTheThingThatWasScored() {
    TypedEvaluation evaluation = evaluation(ROUTE, separable(ROUTE.size(), 400, 0.0));
    evaluation.scoreSealedOnce();
    DecisionArtifact artifact = evaluation.artifact();

    assertThat(artifact.space()).isEqualTo(ROUTE);
    assertThat(artifact.temperature()).isPositive();
  }

  @Test
  void refusesAnArtifactBeforeTheSealedReading() {
    assertThatThrownBy(() -> evaluation(ROUTE, separable(ROUTE.size(), 400, 0.0)).artifact())
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("score the sealed split");
  }

  @Test
  void refusesAnOutcomeOutsideTheSpace() {
    List<TypedRecord> records = new ArrayList<>(separable(ROUTE.size(), 400, 0.0));
    records.add(new TypedRecord("stray", "train", ROUTE.size(), false, new float[] {1f, 2f, 3f}));

    assertThatThrownBy(() -> evaluation(ROUTE, records))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("outside the space");
  }

  @Test
  void refusesAnItemInMoreThanOneSplit() {
    List<TypedRecord> records = new ArrayList<>(separable(ROUTE.size(), 400, 0.0));
    records.add(new TypedRecord(records.get(0).id(), "sealed", 0, false, new float[] {1f, 2f, 3f}));

    assertThatThrownBy(() -> evaluation(ROUTE, records))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("disjoint");
  }

  private static TypedReport evaluate(AnswerSpace space, List<TypedRecord> records) {
    return evaluation(space, records).scoreSealedOnce();
  }

  private static TypedEvaluation evaluation(AnswerSpace space, List<TypedRecord> records) {
    return new TypedEvaluation(
        "synthetic", space, records, new LogisticHeadTrainer(400, 0.2, 0.01), "synthetic-base", "");
  }

  /** One dimension per outcome, hot for the true one, so a linear head can separate them. */
  private static List<TypedRecord> separable(int outcomes, int count, double noiseScale) {
    Random random = new Random(20260921L);
    List<TypedRecord> records = new ArrayList<>(count);
    for (int index = 0; index < count; index++) {
      int outcome = index % outcomes;
      float[] state = new float[outcomes + 2];
      for (int dimension = 0; dimension < state.length; dimension++) {
        state[dimension] = (float) (random.nextGaussian() * noiseScale);
      }
      state[outcome] += 3.0f;
      records.add(new TypedRecord("item-" + index, splitFor(index), outcome, false, state));
    }
    return records;
  }

  /** Labels independent of the state, so nothing above the base rate is learnable. */
  private static List<TypedRecord> noise(int outcomes, int count) {
    Random random = new Random(4242L);
    List<TypedRecord> records = new ArrayList<>(count);
    for (int index = 0; index < count; index++) {
      float[] state = new float[outcomes + 2];
      for (int dimension = 0; dimension < state.length; dimension++) {
        state[dimension] = (float) random.nextGaussian();
      }
      records.add(
          new TypedRecord(
              "item-" + index, splitFor(index), random.nextInt(outcomes), false, state));
    }
    return records;
  }

  private static String splitFor(int index) {
    int bucket = index % 10;
    if (bucket < 6) {
      return "train";
    }
    return bucket < 8 ? "calibration" : "sealed";
  }
}
