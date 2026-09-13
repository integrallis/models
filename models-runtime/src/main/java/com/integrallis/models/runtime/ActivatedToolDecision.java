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

/** One activated call/no-call score produced over a physically shared base-model prefix. */
public record ActivatedToolDecision(
    ToolDecisionScore score,
    int promptTokens,
    int sharedPrefixTokens,
    long sharedPrefixBytes,
    boolean physicallySharesPrefix) {

  public ActivatedToolDecision {
    Objects.requireNonNull(score, "score");
    if (promptTokens <= 0) {
      throw new IllegalArgumentException("promptTokens must be > 0");
    }
    if (sharedPrefixTokens <= 0 || sharedPrefixTokens > promptTokens) {
      throw new IllegalArgumentException(
          "sharedPrefixTokens must be within the prompt: "
              + sharedPrefixTokens
              + " of "
              + promptTokens);
    }
    if (sharedPrefixBytes <= 0) {
      throw new IllegalArgumentException("sharedPrefixBytes must be > 0");
    }
    if (!physicallySharesPrefix) {
      throw new IllegalArgumentException(
          "activated tool decision must physically share its prefix");
    }
  }
}
