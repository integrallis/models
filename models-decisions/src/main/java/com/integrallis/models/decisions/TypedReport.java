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

import java.util.Objects;

/**
 * What a typed head scored on a sealed split.
 *
 * <p>The majority floor sits beside the accuracy because a Choice over four labels and a Choice
 * over forty are not comparable achievements, and an accuracy quoted without its floor hides
 * which one it was.
 *
 * @param corpus the corpus name
 * @param sealedSize how many items were scored
 * @param outcomes how wide the answer space is
 * @param majorityFloor the share of the most common outcome in the sealed split
 * @param accuracy the head's top-label accuracy
 * @param expectedCalibrationError ECE of the head's confidence in its own answer
 * @param brier the multi-class Brier score
 * @param ordinalMeanAbsoluteError mean absolute error in level units, or NaN for an unordered space
 * @param temperature the fitted calibration temperature
 * @param truncatedRate the share of sealed items whose state was cut by the window
 */
public record TypedReport(
    String corpus,
    int sealedSize,
    int outcomes,
    double majorityFloor,
    double accuracy,
    double expectedCalibrationError,
    double brier,
    double ordinalMeanAbsoluteError,
    double temperature,
    double truncatedRate) {

  /** Validates the report. */
  public TypedReport {
    Objects.requireNonNull(corpus, "corpus");
  }

  /**
   * Whether the head beat its own majority floor by the given margin.
   *
   * <p>A head that does not is not a weak head, it is no head: predicting the most common label
   * without reading the state would have done as well.
   */
  public boolean beatsFloor(double requiredMargin) {
    return accuracy - majorityFloor >= requiredMargin;
  }
}
