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
 * A categorical decision over unordered options.
 *
 * @param question the question being answered
 * @param options between two and 255 distinct outcomes, in declaration order
 * @param criteria optional rubric saying what each option covers, keyed by option; see {@link
 *     AnswerSpace#criteria()} for what it is worth
 */
public record Choice(String question, List<String> options, Map<String, String> criteria)
    implements AnswerSpace {

  /** The largest number of options a single categorical decision may offer. */
  public static final int MAX_OPTIONS = 255;

  /** Validates and copies the options and their rubric. */
  public Choice {
    question = AnswerSpaces.requireQuestion(question);
    options = AnswerSpaces.requireLabels(options, 2, MAX_OPTIONS, "options");
    criteria = AnswerSpaces.requireCriteria(criteria, options);
  }

  /** Options whose names are expected to carry their own meaning. */
  public Choice(String question, List<String> options) {
    this(question, options, Map.of());
  }

  @Override
  public List<String> labels() {
    return options;
  }
}
