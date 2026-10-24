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
package com.integrallis.models.backend.purejava.bert;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.backend.purejava.bert.DecisionCalibrationQualification.Outcome;
import com.integrallis.models.backend.purejava.bert.DecisionCalibrationQualification.Report;
import com.integrallis.models.backend.purejava.bert.ModernBertConfig.QuestionType;
import java.util.ArrayList;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * The gate itself, checked on constructed cohorts before it is pointed at a model.
 *
 * <p>A policy that has only ever been run on real output is a policy nobody has seen fail. Each
 * gate here is driven to both sides, because the whole value of a threshold is that it refuses
 * something.
 */
@Tag("unit")
class DecisionCalibrationQualificationTest {

  @Test
  void qualifiesACohortThatClearsEveryGate() {
    List<Outcome> cohort = new ArrayList<>();
    // 40 four-way choices at 0.85 confidence, 34 right: accuracy 0.85, confidence 0.85.
    cohort.addAll(choice(34, true, 0.85));
    cohort.addAll(choice(6, false, 0.85));
    // 20 binary noul at 0.80 confidence, 16 right.
    cohort.addAll(noul(16, true, 0.80));
    cohort.addAll(noul(4, false, 0.80));

    Report report = DecisionCalibrationQualification.assess(cohort);

    assertThat(report.qualified()).as("%s", report.failures()).isTrue();
    assertThat(report.policyId()).isEqualTo("decision-calibration-v2");
    assertThat(report.tier()).isEqualTo(DecisionUseCaseTier.CALIBRATED_DECISION);
    assertThat(report.tier().exportsConfidenceQuality()).isTrue();
    assertThat(report.accuracy()).isEqualTo(0.8333, org.assertj.core.data.Offset.offset(1e-4));
    // Chance from the option counts actually present: 40 at 1/4 and 20 at 1/2.
    assertThat(report.chanceAccuracy())
        .isEqualTo(0.3333, org.assertj.core.data.Offset.offset(1e-4));
    assertThat(report.accuracyByType()).containsEntry("choice", 0.85).containsEntry("noul", 0.80);
  }

  @Test
  void refusesAStrongMultiWaySubsetCarryingAWeakBinaryOne() {
    // The reason the per-type gates exist. Overall accuracy clears 0.70 while the binary subset
    // sits at 0.55 -- barely above its own 0.50 chance, and useless to a caller.
    List<Outcome> cohort = new ArrayList<>();
    cohort.addAll(choice(76, true, 0.8));
    cohort.addAll(choice(4, false, 0.8));
    cohort.addAll(noul(11, true, 0.6));
    cohort.addAll(noul(9, false, 0.6));

    Report report = DecisionCalibrationQualification.assess(cohort);

    assertThat(report.accuracy()).isGreaterThan(DecisionCalibrationQualification.MINIMUM_ACCURACY);
    // An accuracy miss is still exclusion; only calibration demotes.
    assertThat(report.qualified()).isFalse();
    assertThat(report.tier()).isEqualTo(DecisionUseCaseTier.UNQUALIFIED);
    assertThat(report.failures()).anyMatch(failure -> failure.startsWith("noul accuracy"));
    assertThat(report.chanceByType()).containsEntry("noul", 0.5);
  }

  @Test
  void refusesConfidenceThatDoesNotMatchAccuracy() {
    // Right 75% of the time while claiming 0.99 every time: the exact failure the gate exists for,
    // and one accuracy alone cannot see. Accuracy deliberately clears 0.70, because otherwise the
    // model would be excluded for being inaccurate and the calibration demotion would never be
    // reached -- which is what the first version of this fixture did.
    List<Outcome> cohort = new ArrayList<>();
    cohort.addAll(choice(75, true, 0.99));
    cohort.addAll(choice(25, false, 0.99));

    Report report = DecisionCalibrationQualification.assess(cohort);

    assertThat(report.calibrationError())
        .isEqualTo(0.24, org.assertj.core.data.Offset.offset(1e-9));
    // Publishable, but demoted: the ranking is fine and the probability is not. A caller is told
    // exactly that instead of the model being withheld over a bar no other catalog entry faces.
    assertThat(report.qualified()).isTrue();
    assertThat(report.tier()).isEqualTo(DecisionUseCaseTier.RANKED_DECISION);
    assertThat(report.tier().exportsConfidenceQuality()).isFalse();
    assertThat(report.failures()).isEmpty();
    assertThat(report.limitations()).anyMatch(l -> l.startsWith("calibration error"));
  }

  @Test
  void excludesAnswersThatDeclaredThemselvesUncalibrated() {
    // Honesty must not be scored as error. These answers say their probability is not one anybody
    // fitted, so including them in the calibration term would penalise exactly the behaviour the
    // policy asks for.
    List<Outcome> honest = new ArrayList<>();
    honest.addAll(choice(34, true, 0.85));
    honest.addAll(choice(6, false, 0.85));
    honest.add(new Outcome("wild", QuestionType.CHOICE, 12, false, 0.99, true));

    Report report = DecisionCalibrationQualification.assess(honest);

    assertThat(report.uncalibratedAnswers()).isEqualTo(1);
    assertThat(report.uncalibratedBucketsExercised()).isEqualTo(1);
    assertThat(report.failures()).isEmpty();
  }

  @Test
  void refusesAnUncalibratedBucketThatPublishedAProbabilityAnyway() {
    List<Outcome> cohort = new ArrayList<>(choice(32, true, 0.8));
    cohort.addAll(choice(8, false, 0.8));
    cohort.add(new Outcome("twelve-way", QuestionType.CHOICE, 12, true, 0.99, false));

    Report report = DecisionCalibrationQualification.assess(cohort);

    // Still a hard failure, not a demotion: this is the model lying about its own calibration
    // rather than being honestly uncalibrated, and no tier covers that.
    assertThat(report.uncalibratedBucketsExercised()).isEqualTo(1);
    assertThat(report.qualified()).isFalse();
    assertThat(report.tier()).isEqualTo(DecisionUseCaseTier.UNQUALIFIED);
    assertThat(report.failures()).anyMatch(failure -> failure.contains("declares uncalibrated"));
  }

  @Test
  void reportsZeroExercisedWhenTheCohortCannotReachAnUncalibratedBucket() {
    // JevBench tops out at six options, so this is the case that will actually occur. A pass has
    // to be legible as "never tested" rather than as evidence.
    //
    // The cohort is calibrated on purpose -- 32 of 40 right while claiming 0.80 -- because an
    // all-correct cohort at 0.80 is UNDERconfident and the calibration gate rejects it. It did
    // exactly that when this fixture was first written, which is the gate working.
    List<Outcome> cohort = new ArrayList<>(choice(32, true, 0.8));
    cohort.addAll(choice(8, false, 0.8));
    Report report = DecisionCalibrationQualification.assess(cohort);

    assertThat(report.uncalibratedBucketsExercised()).isZero();
    assertThat(report.qualified()).isTrue();
    assertThat(report.tier()).isEqualTo(DecisionUseCaseTier.CALIBRATED_DECISION);
  }

  @Test
  void reportsNotANumberWhenEveryAnswerWasUncalibrated() {
    // Zero would read as perfect calibration, which is the opposite of what no data means.
    // Accurate enough to be publishable, so the NaN demotion is observable rather than masked by
    // an accuracy exclusion.
    List<Outcome> cohort = new ArrayList<>();
    for (int index = 0; index < 10; index++) {
      cohort.add(
          new Outcome("all-uncalibrated-" + index, QuestionType.CHOICE, 12, index < 8, 0.9, true));
    }

    Report report = DecisionCalibrationQualification.assess(cohort);

    assertThat(report.calibrationError()).isNaN();
    // NaN cannot clear a threshold, so it demotes rather than silently qualifying as calibrated.
    assertThat(report.tier()).isEqualTo(DecisionUseCaseTier.RANKED_DECISION);
  }

  @Test
  void refusesAnEmptyCohort() {
    assertThatThrownBy(() -> DecisionCalibrationQualification.assess(List.of()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("empty cohort");
  }

  /**
   * THE EVIDENCE MUST SERIALISE THE SAME WAY TWICE.
   *
   * <p>MEASURED 2026-10-23: re-running Laya's JevBench qualification on released 0.5.4 reproduced
   * every metric exactly -- accuracy, chance floor, calibration error, every per-type cell -- and
   * still produced a different reportSha256, because the canonical constructor used Map.copyOf.
   * Java's immutable maps randomise iteration order per JVM with a salted hash, so `byType` came
   * out score/noul/choice on one run and noul/choice/score on the next.
   *
   * <p>ModelJars pins reportSha256 and matches evidence by SHA-256 rather than by filename, so an
   * unchanged measurement failed its own digest check -- which pushes toward not re-running, the
   * exact opposite of the rule that our own earlier results are re-derived when the library moves.
   *
   * <p>Asserting the key ORDER rather than just the content is the point: equal maps with different
   * iteration order are what broke it, and a contents-only assertion passes while the bug is live.
   */
  @Test
  void reportsTheSameTypeOrderOnEveryConstruction() {
    List<Outcome> cohort = new ArrayList<>();
    // Three types, so the map has something to reorder. Added score-first deliberately: the
    // committed evidence serialised score/noul/choice, so insertion order and sorted order differ
    // and a test that happened to agree with both would prove nothing.
    cohort.addAll(repeat("score", QuestionType.SCORE, 5, 4, true, 0.9));
    cohort.addAll(noul(4, true, 0.9));
    cohort.addAll(choice(4, true, 0.9));

    Report report = DecisionCalibrationQualification.assess(cohort);

    // Pin the ORDER to the order the cohort was assessed in, not merely "stable within this JVM".
    //
    // Repeating assess() in one JVM and comparing runs does NOT catch this, and the first version
    // of this test made exactly that mistake and passed against the bug. Map.copyOf salts its
    // hash once per JVM, so every call in a single process agrees with itself; the divergence only
    // appears between processes, which is why the committed evidence and a re-run differed while a
    // loop inside one test could not tell. Asserting the concrete expected order is what
    // distinguishes an insertion-ordered map from a hash-ordered one inside a single JVM.
    assertThat(new ArrayList<>(report.accuracyByType().keySet()))
        .as("byType must serialise in cohort order, or reportSha256 varies between JVMs")
        .containsExactly("score", "noul", "choice");
    assertThat(new ArrayList<>(report.casesByType().keySet()))
        .containsExactly("score", "noul", "choice");
    assertThat(new ArrayList<>(report.chanceByType().keySet()))
        .containsExactly("score", "noul", "choice");
  }

  private static List<Outcome> choice(int count, boolean correct, double confidence) {
    return repeat("choice", QuestionType.CHOICE, 4, count, correct, confidence);
  }

  private static List<Outcome> noul(int count, boolean correct, double confidence) {
    return repeat("noul", QuestionType.NOUL, 2, count, correct, confidence);
  }

  private static List<Outcome> repeat(
      String prefix,
      QuestionType type,
      int options,
      int count,
      boolean correct,
      double confidence) {
    List<Outcome> outcomes = new ArrayList<>();
    for (int index = 0; index < count; index++) {
      outcomes.add(
          new Outcome(
              prefix + "-" + correct + "-" + index, type, options, correct, confidence, false));
    }
    return outcomes;
  }
}
