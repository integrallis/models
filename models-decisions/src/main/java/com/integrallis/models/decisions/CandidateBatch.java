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
 * A verdict together with the evidence that it was reached the way the tier claims.
 *
 * <p>The counts are carried out of the evaluation rather than logged, so an ablation can read
 * whether the state really was evaluated once and whether the prefix really was shared, instead of
 * inferring either from how long the call took.
 *
 * @param verdict the normalised distribution over the answer space
 * @param statePrefills how many times the state was evaluated; one, however many candidates
 * @param candidateTokensEvaluated forward passes spent on candidate tokens beyond the first
 * @param sharedPrefixProven whether the backend confirmed the branches share physical storage
 */
public record CandidateBatch(
    Verdict verdict, int statePrefills, int candidateTokensEvaluated, boolean sharedPrefixProven) {

  /** Validates the batch. */
  public CandidateBatch {
    Objects.requireNonNull(verdict, "verdict");
    if (statePrefills < 0 || candidateTokensEvaluated < 0) {
      throw new IllegalArgumentException("counts must not be negative");
    }
  }
}
