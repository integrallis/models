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
package com.integrallis.models.rag;

/** Result of validating retrieved evidence before model generation. */
public enum GroundingContextDecision {
  ACCEPTED(true),
  EMPTY(false),
  LOW_CONFIDENCE(false),
  QUESTION_MISMATCH(false),
  TOO_MANY_DOCUMENTS(false),
  TOO_LARGE(false),
  PROMPT_INJECTION(false);

  private final boolean generationAllowed;

  GroundingContextDecision(boolean generationAllowed) {
    this.generationAllowed = generationAllowed;
  }

  /** Whether the retrieved evidence is safe and relevant enough to send to a model. */
  public boolean generationAllowed() {
    return generationAllowed;
  }
}
