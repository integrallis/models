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
 * An ordinal decision placing its subject on ordered levels.
 *
 * <p>The levels are ordered, lowest first. Nothing here assumes the spacing between them is even.
 *
 * @param question the question being answered
 * @param levels between two and ten ordered levels, lowest first
 * @param criteria optional rubric saying what each level means, keyed by level; see {@link
 *     AnswerSpace#criteria()} for what it is worth. Ordinal spaces need it most: a bare "3" says
 *     nothing, and MEASURED 2026-09-24 this family fell from 0.9167 to 0.2500 without one
 */
public record Score(String question, List<String> levels, Map<String, String> criteria)
    implements AnswerSpace {

  /** The largest number of ordered levels a single ordinal decision may offer. */
  public static final int MAX_LEVELS = 10;

  /** Validates and copies the levels and their rubric. */
  public Score {
    question = AnswerSpaces.requireQuestion(question);
    levels = AnswerSpaces.requireLabels(levels, 2, MAX_LEVELS, "levels");
    criteria = AnswerSpaces.requireCriteria(criteria, levels);
  }

  /** Levels whose names are expected to carry their own meaning. */
  public Score(String question, List<String> levels) {
    this(question, levels, Map.of());
  }

  @Override
  public List<String> labels() {
    return levels;
  }
}
