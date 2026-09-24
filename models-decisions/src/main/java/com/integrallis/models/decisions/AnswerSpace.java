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
import java.util.Map;

/**
 * A finite set of outcomes a decision is allowed to return.
 *
 * <p>The space is declared before the decision is taken, so a verdict can only ever name one of
 * these labels. That is what makes the result safe to hand straight to code: there is no parsing
 * step to fail and no value outside the declaration for a model to invent.
 */
public sealed interface AnswerSpace permits Noul, Choice, Score {

  /** Returns the question this space answers. */
  String question();

  /** Returns the outcomes, in declaration order. The list is unmodifiable. */
  List<String> labels();

  /** Returns how many outcomes the space holds. */
  default int size() {
    return labels().size();
  }

  /**
   * Returns what each outcome means, keyed by label. Empty when the labels speak for themselves.
   *
   * <p>A label is a token, not an explanation. "3" or "tier_b" or "pay_subject_to_15000_sublimit"
   * tells a model nothing about when to pick it, and a decision that has to be defensible usually
   * rests on a written rule rather than on the reader guessing what the option was meant to cover.
   * This carries that rule to the model.
   *
   * <p>MEASURED 2026-09-24 and the reason this exists. JevBench gives every system a per-label
   * rubric and this API had nowhere to put one, so the published score was taken on a prompt the
   * shipped code could not build. Same model, same cohort of 120, same kernel, the only difference
   * being whether the rubric reaches the prompt: accuracy 0.8917 with it and 0.7500 without,
   * Intelligence 88.0 against 72.2. Per family it is starker still -- ordinal 0.9167 against 0.2500
   * and routing 0.8333 against 0.3333, which is what "A: 3" means to a reader who was never told
   * what 3 is. For scale, swapping an entire kernel moves one item in that cohort.
   *
   * <p>Keys must be labels of this space; any subset may be given. The map is unmodifiable.
   */
  default Map<String, String> criteria() {
    return Map.of();
  }
}
