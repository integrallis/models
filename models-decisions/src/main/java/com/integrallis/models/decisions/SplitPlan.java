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

import java.util.List;
import java.util.Objects;

/**
 * A drawn sample, divided into the split that fits a head, the split that calibrates it, and the
 * split that scores it.
 *
 * <p>The sealed split is read once, after the head and the temperature are frozen.
 *
 * @param train the items a head may be fitted on
 * @param calibration the items a temperature may be fitted on
 * @param sealedTest the items reserved for the single scoring read
 */
public record SplitPlan(
    List<CorpusItem> train, List<CorpusItem> calibration, List<CorpusItem> sealedTest) {

  /** Validates and copies the splits. */
  public SplitPlan {
    train = List.copyOf(Objects.requireNonNull(train, "train"));
    calibration = List.copyOf(Objects.requireNonNull(calibration, "calibration"));
    sealedTest = List.copyOf(Objects.requireNonNull(sealedTest, "sealedTest"));
  }

  /**
   * Returns the proportion the majority label takes in the sealed split.
   *
   * <p>This is the score a model reaches by always answering the more common way, and no accuracy
   * figure means anything without it beside. It is computed from the sample actually drawn rather
   * than from the corpus's published base rate, because the draw is what was measured.
   */
  public double sealedTestMajorityFloor() {
    if (sealedTest.isEmpty()) {
      throw new IllegalStateException("an empty sealed split has no floor");
    }
    long answerable = sealedTest.stream().filter(CorpusItem::answerable).count();
    double share = (double) answerable / sealedTest.size();
    return Math.max(share, 1.0 - share);
  }
}
