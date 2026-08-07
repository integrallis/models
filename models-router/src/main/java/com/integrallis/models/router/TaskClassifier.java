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
package com.integrallis.models.router;

/**
 * Assigns a query to a task type, which selects the quality column used for scoring.
 *
 * <p>Deliberately an interface rather than a fixed implementation. Classification is the part most
 * worth swapping: an embedding-similarity classifier over pinned exemplars, a trained classifier,
 * or a keyword rule can all satisfy this, and the scoring engine does not care which.
 */
@FunctionalInterface
public interface TaskClassifier {

  /**
   * Classifies one query.
   *
   * @param query the user's request
   * @return a task type, or null when unknown
   */
  String classify(String query);

  /**
   * A classifier that declines to guess.
   *
   * <p>Scoring then averages each candidate's quality across the tasks it declares, so an
   * unclassified request still routes on cost, latency and reliability.
   *
   * @return a classifier returning null
   */
  static TaskClassifier none() {
    return query -> null;
  }
}
