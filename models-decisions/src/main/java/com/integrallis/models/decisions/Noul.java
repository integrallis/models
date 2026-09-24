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

import java.util.List;
import java.util.Map;

/**
 * A binary proposition: the decision reports how probable it is that the proposition holds.
 *
 * @param proposition the statement being judged
 * @param criteria optional rubric saying when the proposition holds and when it does not, keyed by
 *     {@code "true"} and {@code "false"}; see {@link AnswerSpace#criteria()} for what it is worth
 */
public record Noul(String proposition, Map<String, String> criteria) implements AnswerSpace {

  private static final List<String> LABELS = List.of("false", "true");

  /** Validates the proposition and its rubric. */
  public Noul {
    AnswerSpaces.requireQuestion(proposition);
    criteria = AnswerSpaces.requireCriteria(criteria, LABELS);
  }

  /** A proposition whose wording is expected to carry its own meaning. */
  public Noul(String proposition) {
    this(proposition, Map.of());
  }

  /**
   * A proposition with a written rule for each outcome.
   *
   * @param proposition the statement being judged
   * @param whenTrue what must hold for the proposition to be true
   * @param whenFalse what makes it false
   */
  public Noul(String proposition, String whenTrue, String whenFalse) {
    this(proposition, Map.of("true", whenTrue, "false", whenFalse));
  }

  @Override
  public String question() {
    return proposition;
  }

  @Override
  public List<String> labels() {
    return LABELS;
  }

  /** Returns the index carrying the probability that the proposition holds. */
  public int trueIndex() {
    return 1;
  }
}
