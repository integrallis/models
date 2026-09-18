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
package com.integrallis.models.bench.fusion;

import java.util.List;
import java.util.Optional;

/** Dataset-specific extraction and equivalence of final answers. */
public interface AnswerExtractor {

  /** Extracts the answer stated to the user from the final response text. */
  Optional<AnswerExtraction> stated(String finalText, List<String> labels);

  /** Extracts the answer the reasoning itself reaches from a think block. */
  Optional<AnswerExtraction> reasoning(String reasoningText, List<String> labels);

  /** Whether two extracted (or gold) answers are the same answer. */
  boolean equivalent(String left, String right);

  /** Renders a gold answer in the answer format the prompt asks for (gate G4). */
  String renderGold(String gold);

  /** Canonical scorer identifier recorded in reports. */
  String id();

  /** Returns the extractor for a dataset; generic sets choose with {@link #forScorer(String)}. */
  static AnswerExtractor forDataset(DatasetKind dataset) {
    return switch (dataset) {
      case GSM8K -> new NumberAnswerExtractor();
      case ARC -> new LetterAnswerExtractor();
      case MATH500 -> new MathAnswerExtractor();
      case GENERIC ->
          throw new IllegalArgumentException(
              "generic datasets require --scorer number|letter|math");
    };
  }

  /** Returns the extractor named on the command line. */
  static AnswerExtractor forScorer(String scorer) {
    return switch (scorer) {
      case "number" -> new NumberAnswerExtractor();
      case "letter" -> new LetterAnswerExtractor();
      case "math" -> new MathAnswerExtractor();
      default -> throw new IllegalArgumentException("--scorer must be number, letter or math");
    };
  }

  /**
   * Finds the brace-matched content of the last {@code \boxed{...}} or {@code \fbox{...}}.
   *
   * <p>Escaped braces ({@code \{}, {@code \}}) do not count toward nesting.
   */
  static Optional<AnswerExtraction> lastBoxed(String text) {
    AnswerExtraction last = null;
    int from = 0;
    while (true) {
      int boxed = text.indexOf("\\boxed{", from);
      int fbox = text.indexOf("\\fbox{", from);
      int start;
      int open;
      if (boxed < 0 && fbox < 0) {
        break;
      } else if (fbox < 0 || (boxed >= 0 && boxed < fbox)) {
        start = boxed;
        open = boxed + "\\boxed{".length();
      } else {
        start = fbox;
        open = fbox + "\\fbox{".length();
      }
      int depth = 1;
      int index = open;
      while (index < text.length() && depth > 0) {
        char c = text.charAt(index);
        if (c == '\\' && index + 1 < text.length()) {
          index += 2;
          continue;
        }
        if (c == '{') {
          depth++;
        } else if (c == '}') {
          depth--;
        }
        index++;
      }
      if (depth == 0) {
        last = new AnswerExtraction(text.substring(open, index - 1), open, index - 1);
        from = index;
      } else {
        from = start + 1;
      }
    }
    return Optional.ofNullable(last);
  }
}
