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
package com.integrallis.models.bench.fusion;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.within;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Random;
import org.junit.jupiter.api.Test;

class VotingTest {
  private final AnswerExtractor numbers = AnswerExtractor.forDataset(DatasetKind.GSM8K);
  private static final List<String> PRIORITY = List.of("B", "A", "C");

  @Test
  void majorityGroupsEquivalentAnswersAndBreaksTiesWithThePriorityMember() {
    List<Vote> votes =
        List.of(
            new Vote("A", "18", true, -0.1),
            new Vote("B", "18.0", true, -0.2),
            new Vote("C", "20", true, -0.01));
    assertThat(Voting.majority(votes, numbers, PRIORITY)).isEqualTo("18");

    List<Vote> tied =
        List.of(
            new Vote("A", "7", true, -0.1),
            new Vote("B", "9", true, -0.2),
            new Vote("C", null, true, Double.NaN));
    assertThat(Voting.majority(tied, numbers, PRIORITY)).isEqualTo("9");
    List<Vote> tieBreakAbstains =
        List.of(
            new Vote("A", "7", true, -0.1),
            new Vote("B", null, true, Double.NaN),
            new Vote("C", "5", true, -0.1));
    assertThat(Voting.majority(tieBreakAbstains, numbers, PRIORITY)).isEqualTo("7");
    assertThat(Voting.majority(List.of(new Vote("A", null, true, Double.NaN)), numbers, PRIORITY))
        .isNull();
  }

  @Test
  void consistencyFilterDropsMismatchedVotesButKeepsAllWhenEveryVoteIsDropped() {
    List<Vote> votes =
        List.of(
            new Vote("A", "3", false, -0.1),
            new Vote("B", "3", false, -0.1),
            new Vote("C", "4", true, -0.1));
    assertThat(Voting.majority(votes, numbers, PRIORITY)).isEqualTo("3");
    assertThat(Voting.consistencyFiltered(votes, numbers, PRIORITY)).isEqualTo("4");
    List<Vote> allInconsistent =
        List.of(
            new Vote("A", "3", false, -0.1),
            new Vote("B", "3", false, -0.1),
            new Vote("C", "4", false, -0.1));
    assertThat(Voting.consistencyFiltered(allInconsistent, numbers, PRIORITY)).isEqualTo("3");
  }

  @Test
  void confidenceWeightingUsesCalibratedProbabilities() {
    List<Vote> votes =
        List.of(
            new Vote("A", "3", true, -2.0),
            new Vote("B", "3", true, -2.0),
            new Vote("C", "4", true, -0.01));
    Map<String, Double> uncalibrated = Map.of("A", 1.0, "B", 1.0, "C", 1.0);
    assertThat(Voting.confidenceWeighted(votes, numbers, PRIORITY, uncalibrated)).isEqualTo("4");
    Map<String, Double> overconfidentC = Map.of("A", 1.0, "B", 1.0, "C", 0.001);
    assertThat(Voting.confidenceWeighted(votes, numbers, PRIORITY, overconfidentC)).isEqualTo("3");
  }

  @Test
  void selfConsistencyBreaksTiesWithTheEarliestSample() {
    assertThat(Voting.selfConsistency(java.util.Arrays.asList("5", "6", null, "6", "5"), numbers))
        .isEqualTo("5");
    assertThat(Voting.selfConsistency(java.util.Arrays.asList(null, "6", "6", "5"), numbers))
        .isEqualTo("6");
  }

  @Test
  void temperatureCalibrationRecoversTheGeneratingTemperatureAndReducesEce() {
    Random random = new Random(11);
    List<ConfidenceCalibration.Sample> samples = new ArrayList<>();
    for (int i = 0; i < 20000; i++) {
      double c = -3 * random.nextDouble();
      boolean correct = random.nextDouble() < Math.exp(c / 2.5);
      samples.add(new ConfidenceCalibration.Sample(c, correct));
    }
    ConfidenceCalibration.Fit fit = ConfidenceCalibration.fit(samples);
    assertThat(fit.temperature()).isCloseTo(2.5, within(0.25));
    assertThat(fit.eceAfter()).isLessThan(fit.eceBefore());
    assertThat(fit.samples()).isEqualTo(20000);
  }

  @Test
  void expectedCalibrationErrorIsZeroForPerfectlyCalibratedBins() {
    List<ConfidenceCalibration.Sample> samples = new ArrayList<>();
    for (int i = 0; i < 10; i++) {
      samples.add(new ConfidenceCalibration.Sample(Math.log(0.55), i < 5));
    }
    samples.add(new ConfidenceCalibration.Sample(Math.log(0.55), true));
    double ece = ConfidenceCalibration.expectedCalibrationError(samples, 1.0, 10);
    assertThat(ece).isCloseTo(Math.abs(6 / 11.0 - 0.55), within(1e-9));
  }
}
