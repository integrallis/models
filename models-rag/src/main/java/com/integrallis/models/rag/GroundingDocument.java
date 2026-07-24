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

import java.util.Objects;

/** Trusted text and retrieval evidence used to ground a generated answer. */
public record GroundingDocument(String id, String text, float score, int rank) {
  public GroundingDocument {
    id = requireText(id, "id");
    text = requireText(text, "text");
    if (!Float.isFinite(score)) {
      throw new IllegalArgumentException("score must be finite");
    }
    if (rank < 1) {
      throw new IllegalArgumentException("rank must be positive");
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
