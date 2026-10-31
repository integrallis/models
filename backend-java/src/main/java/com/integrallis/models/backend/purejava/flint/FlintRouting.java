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
package com.integrallis.models.backend.purejava.flint;

import com.integrallis.models.backend.purejava.ops.ExpertRouting;

/**
 * Sigmoid routing: the optional bias affects selection only, never the mixture weights.
 *
 * <p>The implementation moved to {@link ExpertRouting#selectExpertsWithSigmoidGating} so the three
 * other MoE ports in the queue can reach it; this stays as flint's call site. {@code
 * FlintRoutingTest} is unchanged across that move, which is what says the move changed nothing.
 */
final class FlintRouting {
  private FlintRouting() {}

  static void select(
      float[] logits,
      float[] bias,
      int[] selected,
      float[] weights,
      boolean normalize,
      float scale) {
    ExpertRouting.selectExpertsWithSigmoidGating(logits, bias, selected, weights, normalize, scale);
  }
}
