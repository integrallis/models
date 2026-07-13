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

import java.util.List;
import java.util.OptionalInt;

/** A vocabulary arranged on a cyclic one-dimensional semantic order. */
public interface SemanticOrder {
  /** Returns the number of terms in the order. */
  int size();

  /** Returns the zero-based rank of an exact term, or an empty result for an unknown term. */
  OptionalInt rank(String term);

  /** Returns the term at a zero-based rank. */
  String termAt(int rank);

  /** Returns all terms in rank order as an immutable list. */
  List<String> terms();

  /**
   * Returns unique neighboring terms, ordered by increasing cyclic distance. At each distance the
   * predecessor is returned before the successor.
   */
  List<SemanticNeighbor> neighbors(String term, int radius);

  /** Returns the shortest number of steps between two known terms on the cycle. */
  int cyclicDistance(String left, String right);

  /** Returns a stable SHA-256 fingerprint of the exact ranked vocabulary. */
  String fingerprint();
}
