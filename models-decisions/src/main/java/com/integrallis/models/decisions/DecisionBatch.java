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
 * The verdicts from one batch of questions, together with the evidence that they were taken the way
 * the tier claims.
 *
 * <p>The sharing figures are carried in the result rather than logged, so an ablation can read
 * whether the prefix was genuinely shared instead of inferring it from how long the batch took.
 *
 * @param verdicts one verdict per question, in the order the questions were asked
 * @param questionCount how many questions were asked
 * @param prefixFreezes how many times the state was frozen; one, however many questions were asked
 * @param allSharedPrefixStorage whether every branch was proven to reference the same physical
 *     prefix storage, as reported by the backend rather than assumed
 */
public record DecisionBatch(
    List<Verdict> verdicts, int questionCount, int prefixFreezes, boolean allSharedPrefixStorage) {

  /** Validates and copies the verdicts. */
  public DecisionBatch {
    verdicts = List.copyOf(Objects.requireNonNull(verdicts, "verdicts"));
    if (verdicts.size() != questionCount) {
      throw new IllegalArgumentException("need one verdict per question");
    }
    if (prefixFreezes < 0) {
      throw new IllegalArgumentException("prefixFreezes must not be negative");
    }
  }
}
