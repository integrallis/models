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
 * Fits a head on the train split, its temperature on the calibration split, and reads the sealed
 * split exactly once.
 *
 * <p>The once-only read is enforced by this object rather than left to discipline. A sealed split
 * that <em>can</em> be scored twice is one that eventually <em>will</em> be, with the second
 * reading kept because the first disappointed.
 */
public final class NoulEvaluation {

  private static final int CALIBRATION_BINS = 15;

  private final String corpus;
  private final AnswerSpace space;
  private final List<HarvestRecord> train = new ArrayList<>();
  private final List<HarvestRecord> calibration = new ArrayList<>();
  private final List<HarvestRecord> sealed = new ArrayList<>();
  private final LogisticHeadTrainer trainer;
  private final String baseModel;
  private final String baseDigest;
  private boolean sealedRead;
  private DecisionArtifact artifact;

  /**
   * Prepares an evaluation over one corpus's harvest.
   *
   * @param corpus the corpus name, carried into the report
   * @param space the answer space, which must be binary for a Noul evaluation
   * @param records every harvested record, already carrying its split
   * @param trainer the head trainer
   * @param baseModel the base whose hidden states were harvested, carried into the artifact
   * @param baseDigest the base file's SHA-256, so a wrong base is caught rather than guessed at
   * @throws IllegalArgumentException if an item appears in more than one split, or a split is empty
   */
  public NoulEvaluation(
      String corpus,
      AnswerSpace space,
      List<HarvestRecord> records,
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
    for (HarvestRecord record : records) {
      if (!seen.add(record.id())) {
        throw new IllegalArgumentException(
            "item " + record.id() + " appears in more than one split; the splits must be disjoint");
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
  public NoulReport scoreSealedOnce() {
    if (sealedRead) {
      throw new IllegalStateException(
          "the sealed split is read once; a second reading is a new experiment needing a new "
              + "pre-registration and a fresh split");
    }
    sealedRead = true;

    // Statistics come from the training rows alone. Fitting them over every split would leak the
    // sealed rows' distribution into the model that is about to be scored on them, and the leak
    // would show up as a good number rather than as an error.
    FeatureStandardizer standardizer = FeatureStandardizer.fit(features(train));
    LinearDecisionHead head =
        trainer.fit(space, standardizer.applyAll(features(train)), labels(train));
    double temperature =
        TemperatureFitter.fit(logits(head, standardizer, calibration), labels(calibration));
    artifact =
        new DecisionArtifact(space, standardizer, head, temperature, baseModel, baseDigest);

    int size = sealed.size();
    double[] probabilities = new double[size];
    boolean[] correct = new boolean[size];
    boolean[] outcomes = new boolean[size];
    int headHits = 0;
    int decodeHits = 0;
    int agreements = 0;
    int truncated = 0;
    int positives = 0;

    for (int index = 0; index < size; index++) {
      HarvestRecord record = sealed.get(index);
      // Scored through the artifact, so the released file is what this report describes.
      Verdict verdict = artifact.decide(record.hiddenInternal());
      boolean headSaysTrue = verdict.probabilityOfTrue() >= 0.5;

      probabilities[index] = verdict.probabilityOfTrue();
      outcomes[index] = record.label();
      correct[index] = headSaysTrue == record.label();
      headHits += correct[index] ? 1 : 0;
      decodeHits += record.decode() == record.label() ? 1 : 0;
      agreements += headSaysTrue == record.decode() ? 1 : 0;
      truncated += record.truncated() ? 1 : 0;
      positives += record.label() ? 1 : 0;
    }

    // Confidence in the head's own answer, which is what calibration is about.
    double[] confidences = new double[size];
    for (int index = 0; index < size; index++) {
      confidences[index] = Math.max(probabilities[index], 1.0 - probabilities[index]);
    }

    double share = (double) positives / size;
    return new NoulReport(
        corpus,
        size,
        Math.max(share, 1.0 - share),
        (double) headHits / size,
        (double) decodeHits / size,
        (double) agreements / size,
        Calibration.expectedCalibrationError(confidences, correct, CALIBRATION_BINS),
        Calibration.brierScore(probabilities, outcomes),
        temperature,
        (double) truncated / size);
  }

  /**
   * The artifact that produced the report, ready to be written and released.
   *
   * <p>This is the same object the sealed split was scored through, so the published numbers and
   * the shipped file cannot drift apart.
   *
   * @throws IllegalStateException if the sealed split has not been scored yet
   */
  public DecisionArtifact artifact() {
    if (artifact == null) {
      throw new IllegalStateException("score the sealed split before releasing the artifact");
    }
    return artifact;
  }

  private static float[][] features(List<HarvestRecord> records) {
    float[][] features = new float[records.size()][];
    for (int index = 0; index < records.size(); index++) {
      features[index] = records.get(index).hiddenInternal();
    }
    return features;
  }

  private static int[] labels(List<HarvestRecord> records) {
    int[] labels = new int[records.size()];
    for (int index = 0; index < records.size(); index++) {
      labels[index] = records.get(index).label() ? 1 : 0;
    }
    return labels;
  }

  private static double[][] logits(
      LinearDecisionHead head, FeatureStandardizer standardizer, List<HarvestRecord> records) {
    double[][] logits = new double[records.size()][];
    for (int index = 0; index < records.size(); index++) {
      logits[index] = head.logits(standardizer.apply(records.get(index).hiddenInternal()));
    }
    return logits;
  }
}
