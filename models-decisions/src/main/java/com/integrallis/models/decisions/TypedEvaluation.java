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

import java.util.ArrayList;
import java.util.HashSet;
import java.util.List;
import java.util.Objects;
import java.util.Set;

/**
 * Fits a typed head, calibrates it, and reads the sealed split once.
 *
 * <p>The binary sibling is {@link NoulEvaluation}; this one carries answer spaces of any width, so
 * it serves Choice and Score. The discipline is identical and deliberately so: statistics come
 * from the training rows alone, the temperature comes from the calibration rows alone, and the
 * sealed rows are opened after the fitting is finished and refuse to be opened twice.
 *
 * <p>The sealed split is scored <em>through the artifact object</em> that a release would ship, so
 * the numbers reported here cannot describe a different construction from the file on disk.
 */
public final class TypedEvaluation {

  private static final int CALIBRATION_BINS = 15;

  private final String corpus;
  private final AnswerSpace space;
  private final List<TypedRecord> train = new ArrayList<>();
  private final List<TypedRecord> calibration = new ArrayList<>();
  private final List<TypedRecord> sealed = new ArrayList<>();
  private final LogisticHeadTrainer trainer;
  private final String baseModel;
  private final String baseDigest;
  private boolean sealedRead;
  private DecisionArtifact artifact;

  /**
   * Prepares an evaluation over one corpus's harvest.
   *
   * @param corpus the corpus name, carried into the report
   * @param space the answer space the head will decide in
   * @param records every harvested record, already carrying its split
   * @param trainer the head trainer
   * @param baseModel the base whose states were harvested
   * @param baseDigest the base file's SHA-256, or empty if unrecorded
   */
  public TypedEvaluation(
      String corpus,
      AnswerSpace space,
      List<TypedRecord> records,
      LogisticHeadTrainer trainer,
      String baseModel,
      String baseDigest) {
    this.corpus = Objects.requireNonNull(corpus, "corpus");
    this.space = Objects.requireNonNull(space, "space");
    this.trainer = Objects.requireNonNull(trainer, "trainer");
    this.baseModel = Objects.requireNonNull(baseModel, "baseModel");
    this.baseDigest = Objects.requireNonNull(baseDigest, "baseDigest");
    Objects.requireNonNull(records, "records");

    Set<String> seen = new HashSet<>();
    for (TypedRecord record : records) {
      if (!seen.add(record.id())) {
        throw new IllegalArgumentException(
            "item " + record.id() + " appears in more than one split; the splits must be disjoint");
      }
      if (record.outcome() >= space.size()) {
        throw new IllegalArgumentException(
            "outcome " + record.outcome() + " is outside the space's " + space.size() + " labels");
      }
      switch (record.split()) {
        case "train" -> train.add(record);
        case "calibration" -> calibration.add(record);
        case "sealed" -> sealed.add(record);
        default -> throw new IllegalArgumentException("unknown split " + record.split());
      }
    }
    if (train.isEmpty() || calibration.isEmpty() || sealed.isEmpty()) {
      throw new IllegalArgumentException(
          "each split must hold at least one item; an empty sealed split scores as perfect and "
              + "means nothing");
    }
  }

  /**
   * Fits, calibrates, and reads the sealed split.
   *
   * @throws IllegalStateException if called more than once
   */
  public TypedReport scoreSealedOnce() {
    if (sealedRead) {
      throw new IllegalStateException(
          "the sealed split is read once; a second reading is a new experiment needing a new "
              + "pre-registration and a fresh split");
    }
    sealedRead = true;

    FeatureStandardizer standardizer = FeatureStandardizer.fit(states(train));
    LinearDecisionHead head =
        trainer.fit(space, standardizer.applyAll(states(train)), outcomes(train));
    double temperature =
        TemperatureFitter.fit(logits(head, standardizer, calibration), outcomes(calibration));
    artifact = new DecisionArtifact(space, standardizer, head, temperature, baseModel, baseDigest);

    int size = sealed.size();
    int outcomeCount = space.size();
    boolean ordered = space instanceof Score;

    double[] confidences = new double[size];
    boolean[] correct = new boolean[size];
    double brierTotal = 0.0;
    double ordinalTotal = 0.0;
    int hits = 0;
    int truncated = 0;
    int[] perOutcome = new int[outcomeCount];

    for (int index = 0; index < size; index++) {
      TypedRecord record = sealed.get(index);
      double[] probabilities = artifact.decide(record.stateInternal()).probabilities();

      int predicted = 0;
      for (int outcome = 1; outcome < outcomeCount; outcome++) {
        if (probabilities[outcome] > probabilities[predicted]) {
          predicted = outcome;
        }
      }
      correct[index] = predicted == record.outcome();
      hits += correct[index] ? 1 : 0;
      confidences[index] = probabilities[predicted];

      for (int outcome = 0; outcome < outcomeCount; outcome++) {
        double target = outcome == record.outcome() ? 1.0 : 0.0;
        double error = probabilities[outcome] - target;
        brierTotal += error * error;
      }
      if (ordered) {
        // Expected level, so a confident near-miss is scored as nearer than a confident far miss.
        double expected = 0.0;
        for (int outcome = 0; outcome < outcomeCount; outcome++) {
          expected += outcome * probabilities[outcome];
        }
        ordinalTotal += Math.abs(expected - record.outcome());
      }
      truncated += record.truncated() ? 1 : 0;
      perOutcome[record.outcome()]++;
    }

    int majority = 0;
    for (int count : perOutcome) {
      majority = Math.max(majority, count);
    }

    return new TypedReport(
        corpus,
        size,
        outcomeCount,
        (double) majority / size,
        (double) hits / size,
        Calibration.expectedCalibrationError(confidences, correct, CALIBRATION_BINS),
        brierTotal / size,
        ordered ? ordinalTotal / size : Double.NaN,
        temperature,
        (double) truncated / size);
  }

  /**
   * The artifact that produced the report, ready to be written and released.
   *
   * @throws IllegalStateException if the sealed split has not been scored yet
   */
  public DecisionArtifact artifact() {
    if (artifact == null) {
      throw new IllegalStateException("score the sealed split before releasing the artifact");
    }
    return artifact;
  }

  private static float[][] states(List<TypedRecord> records) {
    float[][] states = new float[records.size()][];
    for (int index = 0; index < records.size(); index++) {
      states[index] = records.get(index).stateInternal();
    }
    return states;
  }

  private static int[] outcomes(List<TypedRecord> records) {
    int[] outcomes = new int[records.size()];
    for (int index = 0; index < records.size(); index++) {
      outcomes[index] = records.get(index).outcome();
    }
    return outcomes;
  }

  private static double[][] logits(
      LinearDecisionHead head, FeatureStandardizer standardizer, List<TypedRecord> records) {
    double[][] logits = new double[records.size()][];
    for (int index = 0; index < records.size(); index++) {
      logits[index] = head.logits(standardizer.apply(records.get(index).stateInternal()));
    }
    return logits;
  }
}
