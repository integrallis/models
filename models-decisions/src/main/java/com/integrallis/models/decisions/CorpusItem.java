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
 * One question drawn from a published corpus, with the passage it was asked about.
 *
 * <p>The passage identity is carried deliberately: splits are assigned by passage, so that no
 * context a head trained on can reappear in the split that scores it.
 *
 * @param id the corpus's own identifier for this question
 * @param passageId the identifier of the passage or document the question is asked about
 * @param question the question text
 * @param answerable whether the corpus labels this question as answerable from its passage
 */
public record CorpusItem(String id, String passageId, String question, boolean answerable) {

  /** Validates the item. */
  public CorpusItem {
    Objects.requireNonNull(id, "id");
    Objects.requireNonNull(passageId, "passageId");
    Objects.requireNonNull(question, "question");
    if (id.isBlank() || passageId.isBlank()) {
      throw new IllegalArgumentException("id and passageId must not be blank");
    }
  }
}
