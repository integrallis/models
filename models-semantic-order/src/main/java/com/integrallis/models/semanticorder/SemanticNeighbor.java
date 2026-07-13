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
package com.integrallis.models.semanticorder;

import java.util.Objects;

/** A term near a query term on a semantic cycle. */
public record SemanticNeighbor(String term, int rank, int signedOffset) {
  public SemanticNeighbor {
    Objects.requireNonNull(term, "term");
    if (rank < 0) {
      throw new IllegalArgumentException("rank must be >= 0");
    }
    if (signedOffset == 0) {
      throw new IllegalArgumentException("signedOffset must not be zero");
    }
  }

  /** Returns the unsigned cyclic distance from the query term. */
  public int distance() {
    return Math.abs(signedOffset);
  }
}
