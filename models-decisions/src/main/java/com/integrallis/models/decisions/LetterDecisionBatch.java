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
 * The verdicts from one batch of lettered questions over one state, with what it cost and whether
 * the state was really shared.
 *
 * @param verdicts one verdict per answer space, in the order they were asked
 * @param inputTokens tokens the model read: the state once when it was shared and once per question
 *     when it was not, plus every question's suffix
 * @param statePrefills how many times the state was read; one when a prefix was shared
 * @param sharedPrefixProven whether every question's branch was proven by the backend to reference
 *     the same physical prefix storage; {@code false} when nothing was shared or it was not proven
 */
public record LetterDecisionBatch(
    List<Verdict> verdicts, int inputTokens, int statePrefills, boolean sharedPrefixProven) {

  /** Copies the verdicts. */
  public LetterDecisionBatch {
    verdicts = List.copyOf(Objects.requireNonNull(verdicts, "verdicts"));
  }
}
