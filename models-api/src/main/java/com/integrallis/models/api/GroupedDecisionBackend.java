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
 * A backend that can answer several questions about one prefilled prefix in lockstep.
 *
 * <p>A batch of decisions is one piece of evidence and many criteria. Grouping answers them
 * together: the evidence's state is forked per question, and a step runs one token from each
 * unfinished question as a batch of rows.
 *
 * <p>MEASURED 2026-09-24, Harriet on a Hetzner CCX33 (8 vCPU, EPYC Milan, 4 physical cores): twenty
 * questions over one contract cost 8.95 s one at a time and 8.63 s grouped. The gain is 4%, and
 * grouping loses below about ten questions.
 *
 * <p>This interface was introduced expecting much more, on the belief that a decision is bandwidth
 * bound -- that reading the weights dominates and a dozen tokens of question ride along for free.
 * On that box it is not: batched prefill measures 18.5 ms per token, linear with no fixed cost, and
 * halves from one thread to two and again to four before saturating on four physical cores. It is
 * compute bound, so N questions cost N questions' arithmetic however they are arranged. What
 * grouping does save is the bandwidth-bound single-token step ending each question -- one per group
 * instead of one per question, worth 65 ms apiece on that box, which accounts for very nearly all
 * of the 4%. Where that step is dearer the gain is larger: with the native decode kernel off, the
 * same grouping measured 1.69x.
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
   * The group size at which answering together starts to beat answering one at a time.
   *
   * <p>There is no universal answer, which is why this is asked of the backend rather than fixed by
   * the caller. Grouping trades the bandwidth-bound single-token step that ends each question --
   * one per group instead of one per question -- against a lockstep walk whose own batches are
   * narrow. Whether that trade pays depends entirely on what that step costs on this backend, and a
   * backend with a fast one has nothing to sell.
   *
   * <p>MEASURED 2026-09-24, Harriet on a Hetzner CCX33, twenty questions over one contract: with
   * the native quantized decode kernel the two paths are within run-to-run noise at every group
   * size from ten to thirty (0.94x, 1.00x, 1.05x, noise band +-5%). With that kernel off, grouping
   * is worth 1.21x at two questions and 1.69x at twenty. Same code, same box; the only thing that
   * changed was the price of the step grouping removes.
   *
   * <p>The default is never, so a backend that has not measured this does not silently opt in.
   *
   * @return the smallest group worth answering together, or {@link Integer#MAX_VALUE} for never
   */
  default int groupedDecisionBreakEven() {
    return Integer.MAX_VALUE;
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
