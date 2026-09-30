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
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;

/**
 * Multiple-choice label extraction for ARC.
 *
 * <p>Priority: last {@code "answer": "X"}; last "answer is X" / "answer: X" / "answer is (X)"; last
 * {@code \boxed{X}}; last line consisting only of {@code X} or {@code (X)}. X must be one of the
 * item's own labels.
 */
final class LetterAnswerExtractor implements AnswerExtractor {

  private static final List<String> DEFAULT_LABELS = List.of("A", "B", "C", "D", "E");

  @Override
  public Optional<AnswerExtraction> stated(String finalText, List<String> labels) {
    return extract(finalText, labels);
  }

  @Override
  public Optional<AnswerExtraction> reasoning(String reasoningText, List<String> labels) {
    return extract(reasoningText, labels);
  }

  private static Optional<AnswerExtraction> extract(String text, List<String> labels) {
    List<String> allowed = labels == null || labels.isEmpty() ? DEFAULT_LABELS : labels;
    String alternation =
        allowed.stream().map(Pattern::quote).collect(Collectors.joining("|", "(", ")"));
    Optional<AnswerExtraction> json =
        last(Pattern.compile("\"answer\"\\s*:\\s*\"\\s*\\(?" + alternation + "\\)?\\s*\""), text);
    if (json.isPresent()) {
      return json;
    }
    Optional<AnswerExtraction> phrase =
        last(
            Pattern.compile(
                "(?i:answer)\\s*(?:(?i:is)|:)\\s*:?\\s*\\**\\s*\\(?"
                    + alternation
                    + "\\)?(?![A-Za-z0-9])"),
            text);
    if (phrase.isPresent()) {
      return phrase;
    }
    Optional<AnswerExtraction> boxed = AnswerExtractor.lastBoxed(text);
    if (boxed.isPresent()) {
      Matcher inner =
          Pattern.compile("^\\s*(?:\\\\text\\{)?\\s*\\(?" + alternation + "\\)?\\s*\\}?\\s*$")
              .matcher(boxed.get().value());
      if (inner.find()) {
        int start = boxed.get().start() + inner.start(1);
        return Optional.of(
            new AnswerExtraction(inner.group(1), start, start + inner.group(1).length()));
      }
    }
    return last(Pattern.compile("(?m)^[ \\t]*\\(?" + alternation + "\\)?\\.?[ \\t]*$"), text);
  }

  private static Optional<AnswerExtraction> last(Pattern pattern, String text) {
    Matcher matcher = pattern.matcher(text);
    AnswerExtraction last = null;
    while (matcher.find()) {
      last = new AnswerExtraction(matcher.group(1), matcher.start(1), matcher.end(1));
    }
    return Optional.ofNullable(last);
  }

  @Override
  public boolean equivalent(String left, String right) {
    return left.trim().equals(right.trim());
  }

  @Override
  public String renderGold(String gold) {
    return "\"answer\": \"" + gold + "\"";
  }

  @Override
  public String id() {
    return "letter";
  }
}
