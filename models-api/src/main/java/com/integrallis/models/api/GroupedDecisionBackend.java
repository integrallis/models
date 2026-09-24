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
package com.integrallis.models.api;

/**
 * A backend that can answer several questions about one prefilled prefix in a single weight sweep.
 *
 * <p>Reading a model's weights costs the same whether one token rides along or forty do. Asking N
 * questions one at a time sweeps the weights N times; asking them together sweeps once. Measured
 * with Harriet on an 8-vCPU EPYC-Milan box, twenty questions over one contract cost 9.12 s, a flat
 * 0.46 s each, almost all of it re-reading weights for a dozen tokens of question.
 *
 * <p>What a grouped answer must never do is let one question read another. Each branch keeps its
 * own recurrent state and its own key and value positions; only the weights are shared.
 */
public interface GroupedDecisionBackend extends InferenceBackend {

  /** Whether the loaded model graph can answer a group, which not every architecture can. */
  default boolean supportsGroupedDecisions() {
    return false;
  }

  /** The largest group this backend will answer at once, bounded by per-branch state. */
  default int maximumGroupSize() {
    return 1;
  }

  /**
   * Answers each suffix against the currently prefilled prefix, sweeping the weights once.
   *
   * <p>The backend must be positioned at the end of the shared prefix. Every suffix is answered as
   * though it alone followed that prefix.
   *
   * @param suffixes each question's tokens, the criterion and its rendered options
   * @return final-position logits per question, in the order given
   */
  float[][] decideGrouped(int[][] suffixes);
}
