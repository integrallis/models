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

/**
 * A binary proposition: the decision reports how probable it is that the proposition holds.
 *
 * @param proposition the statement being judged
 */
public record Noul(String proposition) implements AnswerSpace {

  private static final List<String> LABELS = List.of("false", "true");

  /** Validates the proposition. */
  public Noul {
    AnswerSpaces.requireQuestion(proposition);
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
