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
package com.integrallis.models.runtime;

import java.util.Objects;

/** Immutable affine head over a model's final normalized hidden state. */
public final class ToolApplicabilityHead {
  private final float[] mean;
  private final float[] scale;
  private final float[] weight;
  private final float bias;

  public ToolApplicabilityHead(float[] mean, float[] scale, float[] weight, float bias) {
    this.mean = copyFinite(mean, "mean");
    this.scale = copyFinite(scale, "scale");
    this.weight = copyFinite(weight, "weight");
    if (this.mean.length == 0
        || this.scale.length != this.mean.length
        || this.weight.length != this.mean.length) {
      throw new IllegalArgumentException("head vectors must have the same positive dimension");
    }
    for (float value : this.scale) {
      if (!(value > 0.0f)) {
        throw new IllegalArgumentException("scale values must be positive");
      }
    }
    if (!Float.isFinite(bias)) {
      throw new IllegalArgumentException("bias must be finite");
    }
    this.bias = bias;
  }

  public int dimension() {
    return mean.length;
  }

  /** Scores with float32 products and a deterministic left-to-right float64 reduction. */
  public ToolApplicabilityScore score(float[] hiddenState) {
    Objects.requireNonNull(hiddenState, "hiddenState");
    if (hiddenState.length != mean.length) {
      throw new IllegalArgumentException(
          "hidden-state dimension differs: " + hiddenState.length + " != " + mean.length);
    }
    double sum = 0.0;
    for (int index = 0; index < hiddenState.length; index++) {
      float value = hiddenState[index];
      if (!Float.isFinite(value)) {
        throw new IllegalArgumentException("hidden state must be finite");
      }
      float standardized = (value - mean[index]) / scale[index];
      float product = standardized * weight[index];
      sum += product;
    }
    return new ToolApplicabilityScore(sum + (double) bias);
  }

  private static float[] copyFinite(float[] values, String name) {
    Objects.requireNonNull(values, name);
    float[] copy = values.clone();
    for (float value : copy) {
      if (!Float.isFinite(value)) {
        throw new IllegalArgumentException(name + " values must be finite");
      }
    }
    return copy;
  }
}
