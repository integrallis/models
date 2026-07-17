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

import java.util.List;
import java.util.Objects;

/** A question and deterministic retrieval/answer expectations. */
public record RagCase(
    String id,
    String question,
    List<String> relevantDocumentIds,
    List<String> requiredFacts,
    boolean answerable) {
  public RagCase {
    id = requireText(id, "id");
    question = requireText(question, "question");
    relevantDocumentIds =
        List.copyOf(Objects.requireNonNull(relevantDocumentIds, "relevantDocumentIds"));
    requiredFacts = List.copyOf(Objects.requireNonNull(requiredFacts, "requiredFacts"));
    if (answerable && (relevantDocumentIds.isEmpty() || requiredFacts.isEmpty())) {
      throw new IllegalArgumentException("answerable cases require sources and facts");
    }
    if (!answerable && (!relevantDocumentIds.isEmpty() || !requiredFacts.isEmpty())) {
      throw new IllegalArgumentException("unanswerable cases must not declare sources or facts");
    }
  }

  private static String requireText(String value, String name) {
    Objects.requireNonNull(value, name);
    if (value.isBlank()) {
      throw new IllegalArgumentException(name + " must not be blank");
    }
    return value;
  }
}
