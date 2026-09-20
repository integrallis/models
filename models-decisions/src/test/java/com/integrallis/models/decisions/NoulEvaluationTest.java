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
 * Scoring a harvested corpus: fit on train, calibrate on calibration, read the sealed split once.
 *
 * <p>The once-only read is enforced by the type rather than by discipline. A sealed split that can
 * be scored twice is a sealed split that will eventually be scored twice, with the second reading
 * chosen because the first was disappointing.
 */
@Tag("unit")
class NoulEvaluationTest {

  private static final Noul SPACE = new Noul("the state answers the question");

  @Test
  void theSealedSplitRefusesASecondReading() {
    NoulEvaluation evaluation = evaluation(separable(400));

    NoulReport first = evaluation.scoreSealedOnce();

    assertThat(first).isNotNull();
    assertThatThrownBy(evaluation::scoreSealedOnce)
        .isInstanceOf(IllegalStateException.class)
        .hasMessageContaining("once");
  }

  @Test
  void aReportCarriesItsFloorAndItsSampleSize() {
    NoulReport report = evaluation(separable(400)).scoreSealedOnce();

    assertThat(report.sealedSize()).isEqualTo(80);
    assertThat(report.majorityFloor()).isBetween(0.5, 1.0);
    assertThat(report.headAccuracy()).isBetween(0.0, 1.0);
    assertThat(report.decodeAccuracy()).isBetween(0.0, 1.0);
    assertThat(report.agreementWithDecode()).isBetween(0.0, 1.0);
    assertThat(report.expectedCalibrationError()).isBetween(0.0, 1.0);
    assertThat(report.brier()).isBetween(0.0, 1.0);
  }

  @Test
  void aSeparableProblemIsLearnedAndBeatsItsFloor() {
    NoulReport report = evaluation(separable(400)).scoreSealedOnce();

    assertThat(report.headAccuracy()).isGreaterThan(0.9);
    assertThat(report.headAccuracy()).isGreaterThan(report.majorityFloor());
  }

  @Test
  void anUnlearnableProblemDoesNotPretendToBeatItsFloor() {
    // Labels independent of the features: nothing to learn. The head should land near the floor,
    // and the report must not dress that up.
    NoulReport report = evaluation(noise(400)).scoreSealedOnce();

    assertThat(report.headAccuracy()).isLessThan(report.majorityFloor() + 0.15);
  }

  @Test
  void agreementWithDecodeIsReportedSeparatelyFromAccuracy() {
    // Decode is deliberately wrong everywhere, labels are learnable. A head that learns the labels
    // must therefore show high accuracy and low agreement. Conflating them would hide that.
    List<HarvestRecord> records = new ArrayList<>();
    for (HarvestRecord r : separable(400)) {
      records.add(
          new HarvestRecord(
              r.id(), r.split(), r.label(), !r.label(), r.truncated(), r.cpuMicros(), r.hidden()));
    }
    NoulReport report = evaluation(records).scoreSealedOnce();

    assertThat(report.headAccuracy()).isGreaterThan(0.9);
    assertThat(report.agreementWithDecode()).isLessThan(0.1);
    assertThat(report.decodeAccuracy()).isLessThan(0.1);
  }

  @Test
  void anOverlapBetweenSplitsIsRefused() {
    List<HarvestRecord> records = new ArrayList<>(separable(400));
    HarvestRecord leaked = records.get(0);
    records.add(
        new HarvestRecord(
            leaked.id(),
            "sealed",
            leaked.label(),
            leaked.decode(),
            leaked.truncated(),
            leaked.cpuMicros(),
            leaked.hidden()));

    assertThatThrownBy(() -> evaluation(records))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining(leaked.id());
  }

  @Test
  void anEmptySealedSplitIsRefusedRatherThanScoredAsPerfect() {
    List<HarvestRecord> onlyTrain = new ArrayList<>();
    for (HarvestRecord r : separable(400)) {
      if (r.split().equals("train")) {
        onlyTrain.add(r);
      }
    }
    assertThatThrownBy(() -> evaluation(onlyTrain)).isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void theTruncationRateIsCarriedSoItCanBeReportedBesideTheNumbers() {
    List<HarvestRecord> records = new ArrayList<>();
    int index = 0;
    for (HarvestRecord r : separable(400)) {
      boolean truncated = (index++ % 4) == 0;
      records.add(
          new HarvestRecord(
              r.id(), r.split(), r.label(), r.decode(), truncated, r.cpuMicros(), r.hidden()));
    }
    NoulReport report = evaluation(records).scoreSealedOnce();

    assertThat(report.sealedTruncatedRate()).isCloseTo(0.25, within(0.08));
  }

  @Test
  void calibrationIsFittedOnTheCalibrationSplitNotTheSealedOne() {
    // Fitting on sealed would leak. The fitted temperature must be identical whether or not the
    // sealed split's labels are flipped, because it never sees them.
    List<HarvestRecord> base = separable(400);
    List<HarvestRecord> flipped = new ArrayList<>();
    for (HarvestRecord r : base) {
      boolean label = r.split().equals("sealed") ? !r.label() : r.label();
      flipped.add(
          new HarvestRecord(
              r.id(), r.split(), label, r.decode(), r.truncated(), r.cpuMicros(), r.hidden()));
    }

    assertThat(evaluation(flipped).scoreSealedOnce().temperature())
        .isEqualTo(evaluation(base).scoreSealedOnce().temperature());
  }

  private static NoulEvaluation evaluation(List<HarvestRecord> records) {
    return new NoulEvaluation("synthetic", SPACE, records, new LogisticHeadTrainer(300, 0.2, 0.01));
  }

  /** Label is a clean function of the first feature, so a linear head can learn it. */
  private static List<HarvestRecord> separable(int count) {
    List<HarvestRecord> records = new ArrayList<>(count);
    for (int index = 0; index < count; index++) {
      double x = (index % 100) / 50.0 - 1.0;
      boolean label = x > 0;
      String split = index < count / 2 ? "train" : index < count * 4 / 5 ? "calibration" : "sealed";
      records.add(
          new HarvestRecord(
              "i" + index, split, label, label, false, 1_000L, new float[] {(float) x, 0.5f}));
    }
    return records;
  }

  /** Label unrelated to the features: there is nothing here to learn. */
  private static List<HarvestRecord> noise(int count) {
    List<HarvestRecord> records = new ArrayList<>(count);
    long state = 7L;
    for (int index = 0; index < count; index++) {
      state = state * 6364136223846793005L + 1442695040888963407L;
      boolean label = ((state >>> 40) & 1L) == 1L;
      String split = index < count / 2 ? "train" : index < count * 4 / 5 ? "calibration" : "sealed";
      records.add(
          new HarvestRecord(
              "i" + index,
              split,
              label,
              label,
              false,
              1_000L,
              new float[] {(float) ((index % 17) / 17.0 - 0.5), 0.5f}));
    }
    return records;
  }
}
