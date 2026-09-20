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
 * One question asked of a state: the tokens that pose it, the head that scores it, and the
 * temperature its calibration established.
 *
 * <p>Questions are independent. Nothing one question asks can influence another, which is what
 * allows them all to fork the same frozen prefix.
 */
public final class Question {

  private final AnswerSpace space;
  private final int[] tokens;
  private final DecisionHead head;
  private final double temperature;

  /**
   * Creates a question.
   *
   * @param space the bounded answer space
   * @param tokens the tokens appended after the shared state to pose the question
   * @param head the head scoring this question's space
   * @param temperature the calibration temperature established for this head on held-out data
   */
  public Question(AnswerSpace space, int[] tokens, DecisionHead head, double temperature) {
    this.space = Objects.requireNonNull(space, "space");
    this.head = Objects.requireNonNull(head, "head");
    Objects.requireNonNull(tokens, "tokens");
    if (tokens.length == 0) {
      throw new IllegalArgumentException("a question must contribute at least one token");
    }
    if (!head.space().equals(space)) {
      throw new IllegalArgumentException("the head scores " + head.space() + ", not " + space);
    }
    if (!Double.isFinite(temperature) || temperature <= 0.0) {
      throw new IllegalArgumentException("temperature must be positive and finite");
    }
    this.tokens = tokens.clone();
    this.temperature = temperature;
  }

  /** Returns the bounded answer space. */
  public AnswerSpace space() {
    return space;
  }

  /** Returns a copy of the tokens that pose the question. */
  public int[] tokens() {
    return tokens.clone();
  }

  /** Returns the head scoring this question. */
  public DecisionHead head() {
    return head;
  }

  /** Returns the calibration temperature. */
  public double temperature() {
    return temperature;
  }

  int tokenCount() {
    return tokens.length;
  }

  int[] tokensInternal() {
    return tokens;
  }
}
