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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/**
 * Per-dimension standardisation of hidden states before a head is fitted.
 *
 * <p>Raw transformer hidden states arrive on wildly different scales per dimension. Fed straight to
 * gradient descent they saturate the softmax, which pins probabilities to zero and one and leaves
 * calibration nothing to work with. Standardising first is not cosmetic: it is what makes the
 * optimisation and the resulting confidences meaningful.
 *
 * <p>The statistics come from the training split alone. Computing them over everything would leak
 * the evaluation set into the fit through the back door, quietly and without touching a label.
 */
@Tag("unit")
class FeatureStandardizerTest {

  @Test
  void aStandardisedColumnHasZeroMeanAndUnitVariance() {
    float[][] rows = {{1.0f, 10.0f}, {2.0f, 20.0f}, {3.0f, 30.0f}, {4.0f, 40.0f}};
    FeatureStandardizer standardizer = FeatureStandardizer.fit(rows);

    float[][] out = new float[rows.length][];
    for (int i = 0; i < rows.length; i++) {
      out[i] = standardizer.apply(rows[i]);
    }
    for (int column = 0; column < 2; column++) {
      double mean = 0.0;
      for (float[] row : out) {
        mean += row[column];
      }
      mean /= out.length;
      double var = 0.0;
      for (float[] row : out) {
        var += (row[column] - mean) * (row[column] - mean);
      }
      var /= out.length;
      assertThat(mean).isCloseTo(0.0, within(1e-6));
      assertThat(var).isCloseTo(1.0, within(1e-5));
    }
  }

  @Test
  void aConstantColumnSurvivesInsteadOfDividingByZero() {
    float[][] rows = {{5.0f, 1.0f}, {5.0f, 2.0f}, {5.0f, 3.0f}};
    FeatureStandardizer standardizer = FeatureStandardizer.fit(rows);

    float[] out = standardizer.apply(new float[] {5.0f, 2.0f});

    assertThat(out[0]).isFinite().isEqualTo(0.0f);
    assertThat(out[1]).isFinite();
  }

  @Test
  void theStatisticsComeFromTheFittingRowsAlone() {
    // Fit on a narrow range, then apply to a far-off row: it must land far from zero rather than
    // being renormalised as though it had been part of the fit.
    float[][] train = {{1.0f}, {2.0f}, {3.0f}};
    FeatureStandardizer standardizer = FeatureStandardizer.fit(train);

    assertThat(standardizer.apply(new float[] {100.0f})[0]).isGreaterThan(50.0f);
  }

  @Test
  void applyingTheSameRowTwiceIsBitIdentical() {
    float[][] rows = {{0.3f, 7.1f}, {-2.0f, 0.5f}, {4.4f, -1.25f}};
    FeatureStandardizer standardizer = FeatureStandardizer.fit(rows);
    float[] row = {1.1f, 2.2f};

    assertThat(standardizer.apply(row)).isEqualTo(standardizer.apply(row));
  }

  @Test
  void theWidthMustMatchWhatWasFitted() {
    FeatureStandardizer standardizer = FeatureStandardizer.fit(new float[][] {{1.0f, 2.0f}});

    assertThatThrownBy(() -> standardizer.apply(new float[] {1.0f}))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void emptyFittingDataIsRefused() {
    assertThatThrownBy(() -> FeatureStandardizer.fit(new float[0][0]))
        .isInstanceOf(IllegalArgumentException.class);
  }

  @Test
  void standardisingDoesNotAlterTheInputRow() {
    float[][] rows = {{1.0f, 10.0f}, {3.0f, 30.0f}};
    FeatureStandardizer standardizer = FeatureStandardizer.fit(rows);
    float[] row = {2.0f, 20.0f};
    float[] copy = row.clone();

    standardizer.apply(row);

    assertThat(row).isEqualTo(copy);
  }
}
