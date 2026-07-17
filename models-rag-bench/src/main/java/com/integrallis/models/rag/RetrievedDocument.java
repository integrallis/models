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

/** A ranked retrieval result. */
public record RetrievedDocument(RagDocument document, float score, int rank) {
  public RetrievedDocument {
    document = Objects.requireNonNull(document, "document");
    if (!Float.isFinite(score)) {
      throw new IllegalArgumentException("score must be finite");
    }
    if (rank < 1) {
      throw new IllegalArgumentException("rank must be positive");
    }
  }
}
