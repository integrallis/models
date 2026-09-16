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
package com.integrallis.models.bench;

import static org.assertj.core.api.Assertions.assertThat;

import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Test;

class ActivatedQ4CalibrationCliTest {

  @Test
  void selectsTheFixedBoundaryFromCalibrationScoresOnly() {
    var observations =
        List.of(
            observation("calibration", true, 10),
            observation("calibration", true, 9),
            observation("calibration", false, 1),
            observation("calibration", false, 0));

    var selection = ActivatedQ4CalibrationCli.selectThreshold(observations);

    assertThat(selection.threshold()).isEqualTo(5.0);
    assertThat(selection.score().correctCalls()).isEqualTo(2);
    assertThat(selection.score().correctNoCalls()).isEqualTo(2);
  }

  @Test
  void calibrationGateRequiresCountsAccuracyAndPhysicalSharing() {
    var passing = score(20, 10, 19, 9);

    assertThat(ActivatedQ4CalibrationCli.calibrationPassed(passing, 30, 30)).isTrue();
    assertThat(ActivatedQ4CalibrationCli.calibrationPassed(passing, 29, 30)).isFalse();
    assertThat(ActivatedQ4CalibrationCli.calibrationPassed(score(20, 10, 18, 10), 30, 30))
        .isFalse();
    assertThat(ActivatedQ4CalibrationCli.calibrationPassed(score(20, 10, 20, 8), 30, 30)).isFalse();
  }

  @Test
  void screenGateRequiresTheScreenAndCombinedFloors() {
    var calibration = score(20, 10, 19, 10);
    var passingScreen = score(30, 15, 28, 14);

    assertThat(ActivatedQ4CalibrationCli.screenPassed(calibration, passingScreen, 45, 45)).isTrue();
    assertThat(ActivatedQ4CalibrationCli.screenPassed(score(20, 10, 18, 10), passingScreen, 45, 45))
        .isFalse();
    assertThat(ActivatedQ4CalibrationCli.screenPassed(calibration, score(30, 15, 28, 13), 45, 45))
        .isFalse();
  }

  @Test
  void scoreRejectsMixingTheFrozenPartitions() {
    List<ActivatedQ4CalibrationCli.Observation> observations = new ArrayList<>();
    observations.add(observation("calibration", true, 1));
    observations.add(observation("screen", false, -1));

    org.assertj.core.api.Assertions.assertThatIllegalArgumentException()
        .isThrownBy(() -> ActivatedQ4CalibrationCli.score(observations, 0, "calibration"))
        .withMessageContaining("partition");
  }

  private static ActivatedQ4CalibrationCli.Observation observation(
      String partition, boolean expected, double score) {
    return new ActivatedQ4CalibrationCli.Observation(
        "id", "kind", partition, expected, 1, 1, 1, true, score, score > 0, 1);
  }

  private static ActivatedQ4CalibrationCli.Score score(
      int calls, int noCalls, int correctCalls, int correctNoCalls) {
    double callAccuracy = (double) correctCalls / calls;
    double noCallAccuracy = (double) correctNoCalls / noCalls;
    return new ActivatedQ4CalibrationCli.Score(
        calls,
        noCalls,
        correctCalls,
        correctNoCalls,
        callAccuracy,
        noCallAccuracy,
        (callAccuracy + noCallAccuracy) / 2.0);
  }
}
