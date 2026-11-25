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
package com.integrallis.models.runtime.fusion;

import java.util.Objects;

/**
 * How a {@link FusedInferenceBackend} combines its members.
 *
 * @param rule the combination rule applied when the gate consults the auxiliaries
 * @param weights one weight per member, lead first, on the simplex (they sum to one)
 * @param gate decides per step whether the auxiliaries are consulted
 */
public record FusionRecipe(FusionRule rule, double[] weights, EnsembleGate gate) {

  /** Validates and defensively copies the weights. */
  public FusionRecipe {
    Objects.requireNonNull(rule, "rule");
    Objects.requireNonNull(weights, "weights");
    Objects.requireNonNull(gate, "gate");
    if (weights.length < 2) {
      throw new IllegalArgumentException("a fusion needs at least two members");
    }
    FusionMath.validateWeights(weights);
    weights = weights.clone();
  }

  /** Product of experts with uniform weights over {@code members} members and the given gate. */
  public static FusionRecipe uniform(int members, EnsembleGate gate) {
    if (members < 2) {
      throw new IllegalArgumentException("a fusion needs at least two members");
    }
    double[] weights = new double[members];
    java.util.Arrays.fill(weights, 1.0 / members);
    return new FusionRecipe(FusionRule.POE, weights, gate);
  }

  /** Number of members the recipe expects. */
  public int members() {
    return weights.length;
  }

  @Override
  public double[] weights() {
    return weights.clone();
  }
}
