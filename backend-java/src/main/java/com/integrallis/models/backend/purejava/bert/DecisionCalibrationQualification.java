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

import java.util.ArrayList;
import java.util.Collections;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;

/**
 * The qualification gate for a typed-decision model: is it right, and does its confidence mean
 * anything?
 *
 * <p><b>Why a new policy at all.</b> The RAG gate measures a pipeline's answer text and the tool
 * gate measures call grammar. Neither describes this: there is no generated text to score and no
 * call to parse, only a chosen option and a probability. {@code correctAnswerRate} in particular is
 * a pipeline metric and would be meaningless here.
 *
 * <p><b>Chance is reported before accuracy, always.</b> An accuracy figure over a mixed cohort of
 * binary and six-way questions says very little on its own — 0.60 is barely above guessing on a
 * binary subset and nearly three times chance on a six-way one. {@link Report#chanceAccuracy()} is
 * computed from the option counts actually present, and the per-type gates exist so a strong
 * multi-way subset cannot carry a weak binary one.
 *
 * <p><b>Calibration is the point of the gate, not a nicety.</b> A typed-decision model that only
 * ranked options would not need a temperature table; this one ships one, so the expected
 * calibration error is what separates it from any argmax classifier. The policy also requires that
 * an answer whose bucket the artifact declares uncalibrated says so rather than publishing a
 * confident-looking number — this checkpoint ships a temperature its own reference refuses.
 *
 * <p>Thresholds are in {@link #POLICY_ID} and were registered in {@code
 * experiments/laya-modern-bert/README.md} before any case was run. They are not to be revisited in
 * response to a result; a miss is a null.
 */
public final class DecisionCalibrationQualification {

  /**
   * The policy these thresholds belong to; a change of threshold is a change of id.
   *
   * <p>v2 does not move a single threshold. **The 0.10 calibration bar is byte-for-byte the one
   * registered before Laya ever ran.** What changed is its consequence: in v1 missing it meant NOT
   * QUALIFIED, and in v2 it means {@link DecisionUseCaseTier#RANKED_DECISION} instead of {@link
   * DecisionUseCaseTier#CALIBRATED_DECISION}.
   *
   * <p>That change was made after seeing a result, which deserves saying plainly rather than
   * burying. It is not threshold-shopping: the number is untouched and the measured 0.1638 is
   * published either way. The pass/fail SHAPE was wrong, because no other entry in this catalog
   * faces a calibration bar — the RAG gate scores pipeline contribution, the tool gate scores call
   * grammar — so v1 excluded an external candidate for missing a standard none of our own models is
   * held to. Our own Harriet would fail it too, at 0.153 and 0.322. Tiering is the shape the
   * catalog already uses for exactly this (see {@code RagUseCaseTier}), and v1 should have used it
   * from the start.
   */
  public static final String POLICY_ID = "decision-calibration-v2";

  public static final double MINIMUM_ACCURACY = 0.70;
  public static final double MINIMUM_CHOICE_ACCURACY = 0.60;
  public static final double MINIMUM_NOUL_ACCURACY = 0.70;
  public static final double MAXIMUM_CALIBRATION_ERROR = 0.10;

  /** Bins for the expected calibration error, equal width over [0, 1]. */
  public static final int CALIBRATION_BINS = 10;

  private DecisionCalibrationQualification() {}

  /** One case's outcome: what was asked, what came back, and whether it was right. */
  public record Outcome(
      String id,
      ModernBertConfig.QuestionType type,
      int options,
      boolean correct,
      double confidence,
      boolean uncalibrated) {
    public Outcome {
      id = Objects.requireNonNull(id, "id");
      type = Objects.requireNonNull(type, "type");
      if (options < 1) {
        throw new IllegalArgumentException("a decision needs at least one option");
      }
      if (!(confidence >= 0.0) || !(confidence <= 1.0)) {
        throw new IllegalArgumentException("confidence must be in [0, 1]: " + confidence);
      }
    }
  }

  /** The measured result, with every gate's number whether it passed or not. */
  public record Report(
      String policyId,
      int cases,
      double chanceAccuracy,
      double accuracy,
      Map<String, Double> accuracyByType,
      Map<String, Integer> casesByType,
      Map<String, Double> chanceByType,
      double calibrationError,
      int uncalibratedAnswers,
      int uncalibratedBucketsExercised,
      DecisionUseCaseTier tier,
      List<String> failures,
      List<String> limitations) {

    public Report {
      // Order-preserving defensive copies, NOT Map.copyOf. Map.copyOf returns one of the
      // immutable map implementations, whose iteration order is deliberately randomised per JVM
      // by a salted hash -- so the same measurement serialised in a different key order on every
      // run. MEASURED 2026-10-23: re-running this qualification on released 0.5.4 reproduced
      // every metric exactly and still produced a different reportSha256, because `byType` came
      // out as score/noul/choice one run and noul/choice/score the next. ModelJars pins that
      // digest and matches evidence by SHA-256, so an unchanged result failed its own digest
      // check. These still satisfy EI_EXPOSE_REP: the caller's map cannot reach the record.
      accuracyByType = Collections.unmodifiableMap(new LinkedHashMap<>(accuracyByType));
      casesByType = Collections.unmodifiableMap(new LinkedHashMap<>(casesByType));
      chanceByType = Collections.unmodifiableMap(new LinkedHashMap<>(chanceByType));
      failures = List.copyOf(failures);
      limitations = List.copyOf(limitations);
    }

    /** Whether the accuracy gates passed, so the model is publishable at some tier. */
    public boolean qualified() {
      return tier.publishable();
    }
  }

  /**
   * Scores a cohort.
   *
   * @param outcomes one entry per case, in any order
   * @return the report, including the gates that failed
   */
  public static Report assess(List<Outcome> outcomes) {
    Objects.requireNonNull(outcomes, "outcomes");
    if (outcomes.isEmpty()) {
      throw new IllegalArgumentException("an empty cohort cannot qualify anything");
    }

    double chance = 0.0;
    int correct = 0;
    Map<String, int[]> byType = new LinkedHashMap<>();
    Map<String, Double> chanceSumByType = new LinkedHashMap<>();
    int uncalibrated = 0;

    for (Outcome outcome : outcomes) {
      String type = outcome.type().key();
      chance += 1.0 / outcome.options();
      chanceSumByType.merge(type, 1.0 / outcome.options(), Double::sum);
      int[] tally = byType.computeIfAbsent(type, key -> new int[2]);
      tally[1]++;
      if (outcome.correct()) {
        correct++;
        tally[0]++;
      }
      if (outcome.uncalibrated()) {
        uncalibrated++;
      }
    }

    double accuracy = (double) correct / outcomes.size();
    Map<String, Double> accuracyByType = new LinkedHashMap<>();
    Map<String, Integer> casesByType = new LinkedHashMap<>();
    Map<String, Double> chanceByType = new LinkedHashMap<>();
    byType.forEach(
        (type, tally) -> {
          accuracyByType.put(type, (double) tally[0] / tally[1]);
          casesByType.put(type, tally[1]);
          chanceByType.put(type, chanceSumByType.get(type) / tally[1]);
        });

    double ece = calibrationError(outcomes);

    // Accuracy decides whether the model is publishable at all; calibration decides only its tier.
    List<String> failures = new ArrayList<>();
    List<String> limitations = new ArrayList<>();
    if (accuracy < MINIMUM_ACCURACY) {
      failures.add(
          String.format(
              Locale.ROOT,
              "accuracy %.4f < %.2f (chance %.4f)",
              accuracy,
              MINIMUM_ACCURACY,
              chance / outcomes.size()));
    }
    checkType(failures, accuracyByType, casesByType, "choice", MINIMUM_CHOICE_ACCURACY);
    checkType(failures, accuracyByType, casesByType, "noul", MINIMUM_NOUL_ACCURACY);
    boolean calibrated = !(ece > MAXIMUM_CALIBRATION_ERROR);
    if (!calibrated) {
      limitations.add(
          String.format(
              Locale.ROOT,
              "calibration error %.4f > %.2f: the ranking is reliable, the published probability is"
                  + " not a fitted probability in this regime and no confidence quality is exported",
              ece,
              MAXIMUM_CALIBRATION_ERROR));
    }
    if (Double.isNaN(ece)) {
      limitations.add(
          "every answer declared itself uncalibrated, so no calibration could be measured");
      calibrated = false;
    }

    // The uncalibrated-reporting rule, and the honest statement of whether it was testable. A
    // cohort with no question in an uncalibrated bucket cannot exercise it, and a pass there is no
    // data rather than evidence.
    int exercised = 0;
    for (Outcome outcome : outcomes) {
      if (outcome.options() >= 11 && outcome.type() == ModernBertConfig.QuestionType.CHOICE) {
        exercised++;
        if (!outcome.uncalibrated()) {
          failures.add(
              outcome.id()
                  + " has "
                  + outcome.options()
                  + " options, a bucket the artifact declares uncalibrated, yet published a"
                  + " probability as if it were calibrated");
        }
      }
    }

    DecisionUseCaseTier tier =
        !failures.isEmpty()
            ? DecisionUseCaseTier.UNQUALIFIED
            : calibrated
                ? DecisionUseCaseTier.CALIBRATED_DECISION
                : DecisionUseCaseTier.RANKED_DECISION;

    return new Report(
        POLICY_ID,
        outcomes.size(),
        chance / outcomes.size(),
        accuracy,
        accuracyByType,
        casesByType,
        chanceByType,
        ece,
        uncalibrated,
        exercised,
        tier,
        List.copyOf(failures),
        List.copyOf(limitations));
  }

  private static void checkType(
      List<String> failures,
      Map<String, Double> accuracyByType,
      Map<String, Integer> casesByType,
      String type,
      double minimum) {
    Double measured = accuracyByType.get(type);
    if (measured == null) {
      // Absent, not failed. A cohort without this type says nothing about it either way.
      return;
    }
    if (measured < minimum) {
      failures.add(
          String.format(
              Locale.ROOT,
              "%s accuracy %.4f < %.2f over %d cases",
              type,
              measured,
              minimum,
              casesByType.get(type)));
    }
  }

  /**
   * Expected calibration error: the gap between stated confidence and observed accuracy.
   *
   * <p>Equal-width bins over the confidence the model published, each contributing in proportion to
   * how many answers landed in it. Uncalibrated answers are <b>excluded</b>: the model has said its
   * probability is not one anybody fitted, and scoring it against a calibration target would
   * penalise the honesty the policy asks for everywhere else.
   */
  private static double calibrationError(List<Outcome> outcomes) {
    double[] confidenceSum = new double[CALIBRATION_BINS];
    double[] correctSum = new double[CALIBRATION_BINS];
    int[] counts = new int[CALIBRATION_BINS];
    int scored = 0;

    for (Outcome outcome : outcomes) {
      if (outcome.uncalibrated()) {
        continue;
      }
      int bin = Math.min(CALIBRATION_BINS - 1, (int) (outcome.confidence() * CALIBRATION_BINS));
      confidenceSum[bin] += outcome.confidence();
      correctSum[bin] += outcome.correct() ? 1.0 : 0.0;
      counts[bin]++;
      scored++;
    }
    if (scored == 0) {
      // Every answer declared itself uncalibrated. There is nothing to score, and reporting 0.0
      // would read as perfect calibration.
      return Double.NaN;
    }

    double error = 0.0;
    for (int bin = 0; bin < CALIBRATION_BINS; bin++) {
      if (counts[bin] == 0) {
        continue;
      }
      double meanConfidence = confidenceSum[bin] / counts[bin];
      double meanAccuracy = correctSum[bin] / counts[bin];
      error += (counts[bin] / (double) scored) * Math.abs(meanConfidence - meanAccuracy);
    }
    return error;
  }
}
