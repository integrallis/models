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
 * Per-dimension mean and scale, fitted on training rows and applied to every row thereafter.
 *
 * <p>Transformer hidden states arrive on very different scales per dimension. Handed straight to
 * gradient descent they saturate the softmax, pinning probabilities to zero and one and leaving a
 * calibrator nothing to work with. Standardising is therefore not cosmetic; it is what makes both
 * the optimisation and the resulting confidences mean anything.
 *
 * <p>The statistics are fitted on the training rows alone. Computing them across everything would
 * leak the evaluation set into the fit without ever touching a label, which is the kind of leak
 * that never shows up as an error and quietly inflates a result.
 *
 * <p>A dimension with no variance is mapped to zero rather than divided by zero: it carries no
 * information, and a constant feature should contribute nothing instead of producing infinities.
 */
public final class FeatureStandardizer {

  private final double[] mean;
  private final double[] scale;

  private FeatureStandardizer(double[] mean, double[] scale) {
    this.mean = mean;
    this.scale = scale;
  }

  /**
   * Fits the statistics on the given rows.
   *
   * @param rows the training rows, all of the same width
   */
  public static FeatureStandardizer fit(float[][] rows) {
    Objects.requireNonNull(rows, "rows");
    if (rows.length == 0) {
      throw new IllegalArgumentException("standardisation needs at least one row");
    }
    int width = Objects.requireNonNull(rows[0], "row").length;
    if (width == 0) {
      throw new IllegalArgumentException("a row must not be empty");
    }

    double[] mean = new double[width];
    for (float[] row : rows) {
      if (Objects.requireNonNull(row, "row").length != width) {
        throw new IllegalArgumentException("every row must be " + width + " wide");
      }
      for (int index = 0; index < width; index++) {
        mean[index] += row[index];
      }
    }
    for (int index = 0; index < width; index++) {
      mean[index] /= rows.length;
    }

    double[] scale = new double[width];
    for (float[] row : rows) {
      for (int index = 0; index < width; index++) {
        double delta = row[index] - mean[index];
        scale[index] += delta * delta;
      }
    }
    for (int index = 0; index < width; index++) {
      double variance = scale[index] / rows.length;
      // A dimension with no spread carries nothing; map it to zero rather than to infinity.
      scale[index] = variance > 1e-12 ? Math.sqrt(variance) : 0.0;
    }
    return new FeatureStandardizer(mean, scale);
  }

  /** Returns the width this standardiser was fitted on. */
  public int width() {
    return mean.length;
  }

  /**
   * Returns a standardised copy of one row. The input is not modified.
   *
   * @param row a row of the fitted width
   */
  public float[] apply(float[] row) {
    Objects.requireNonNull(row, "row");
    if (row.length != mean.length) {
      throw new IllegalArgumentException("row must be " + mean.length + " wide, got " + row.length);
    }
    float[] out = new float[row.length];
    for (int index = 0; index < row.length; index++) {
      out[index] = scale[index] == 0.0 ? 0.0f : (float) ((row[index] - mean[index]) / scale[index]);
    }
    return out;
  }

  /** Standardises every row, returning a fresh array. */
  public float[][] applyAll(float[][] rows) {
    Objects.requireNonNull(rows, "rows");
    float[][] out = new float[rows.length][];
    for (int index = 0; index < rows.length; index++) {
      out[index] = apply(rows[index]);
    }
    return out;
  }
}
