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

import java.util.LinkedHashSet;
import java.util.List;
import java.util.Objects;

/** Shared validation for the bounded answer spaces. */
final class AnswerSpaces {

  private AnswerSpaces() {}

  static String requireQuestion(String question) {
    Objects.requireNonNull(question, "question");
    if (question.isBlank()) {
      throw new IllegalArgumentException("question must not be blank");
    }
    return question;
  }

  static List<String> requireLabels(List<String> labels, int minimum, int maximum, String what) {
    Objects.requireNonNull(labels, what);
    if (labels.size() < minimum || labels.size() > maximum) {
      throw new IllegalArgumentException(
          what + " must hold between " + minimum + " and " + maximum + ", got " + labels.size());
    }
    LinkedHashSet<String> unique = new LinkedHashSet<>(labels.size());
    for (String label : labels) {
      Objects.requireNonNull(label, what + " entry");
      if (label.isBlank()) {
        throw new IllegalArgumentException(what + " must not hold a blank entry");
      }
      if (!unique.add(label)) {
        throw new IllegalArgumentException(what + " must not repeat " + label);
      }
    }
    return List.copyOf(unique);
  }
}
