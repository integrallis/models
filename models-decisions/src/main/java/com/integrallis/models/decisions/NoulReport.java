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

/**
 * What one corpus's sealed split yielded, with everything needed to read it honestly.
 *
 * <p>Accuracy and agreement are separate fields on purpose. Accuracy is measured against the
 * corpus's own labels and is the only one of the two that is correctness; agreement is measured
 * against the base's decode verdict and says merely that a head reproduces what the decode loop
 * did, however wrong that was. Reporting one figure that blends them is the error this whole
 * protocol exists to avoid.
 *
 * <p>{@code majorityFloor} and {@code sealedSize} travel with every number so nobody has to go
 * looking for them. A margin narrower than the sample supports is not a difference.
 *
 * @param corpus which corpus this scores
 * @param sealedSize how many items the sealed split held
 * @param majorityFloor the score a model reaches by always answering the more common way
 * @param headAccuracy the head's accuracy against the corpus labels
 * @param decodeAccuracy the decode loop's accuracy against the corpus labels
 * @param agreementWithDecode how often the head reproduces the decode verdict
 * @param expectedCalibrationError the head's ECE over fifteen equal-width bins
 * @param brier the head's Brier score
 * @param temperature the calibration temperature, fitted on the calibration split alone
 * @param sealedTruncatedRate the share of sealed items whose state the window cut
 */
public record NoulReport(
    String corpus,
    int sealedSize,
    double majorityFloor,
    double headAccuracy,
    double decodeAccuracy,
    double agreementWithDecode,
    double expectedCalibrationError,
    double brier,
    double temperature,
    double sealedTruncatedRate) {

  /**
   * Returns whether the base could do this task at all, by the pre-registered sanity condition.
   *
   * <p>If the decode verdict cannot beat its own floor by this margin then agreement with it is
   * agreement with a coin flip, and the arm is uninformative rather than passed or failed.
   */
  public boolean teacherIsInformative(double requiredMargin) {
    return decodeAccuracy - majorityFloor >= requiredMargin;
  }
}
