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

import java.math.BigDecimal;
import java.util.List;
import java.util.Optional;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * Final-number extraction for GSM8K-style answers.
 *
 * <p>Priority: last {@code \boxed{}}; last "answer is X" / "answer: X"; last {@code #### X}; last
 * number (in the last non-empty paragraph for reasoning, in the whole text for the stated answer).
 */
final class NumberAnswerExtractor implements AnswerExtractor {

  private static final String NUMBER = "-?(?:\\d[\\d,]*(?:\\.\\d+)?|\\.\\d+)";
  private static final Pattern NUMBER_PATTERN = Pattern.compile(NUMBER);
  private static final Pattern ANSWER_PHRASE =
      Pattern.compile("(?i)answer\\s*(?:is|:)\\s*:?\\s*\\**\\s*(?:\\\\?\\$\\s*)?(" + NUMBER + ")");
  private static final Pattern HASHES =
      Pattern.compile("####\\s*(?:\\\\?\\$\\s*)?(" + NUMBER + ")");
  private static final Pattern PARAGRAPH_BREAK = Pattern.compile("\\n\\s*\\n");

  @Override
  public Optional<AnswerExtraction> stated(String finalText, List<String> labels) {
    return extract(finalText, false);
  }

  @Override
  public Optional<AnswerExtraction> reasoning(String reasoningText, List<String> labels) {
    return extract(reasoningText, true);
  }

  private Optional<AnswerExtraction> extract(String text, boolean lastParagraphOnly) {
    Optional<AnswerExtraction> boxed = AnswerExtractor.lastBoxed(text);
    if (boxed.isPresent()) {
      AnswerExtraction inner = boxed.get();
      Optional<AnswerExtraction> number = lastNumber(inner.value(), 0);
      if (number.isPresent()) {
        return number.map(n -> n.shift(inner.start()));
      }
    }
    Optional<AnswerExtraction> phrase = lastGroup(ANSWER_PHRASE, text);
    if (phrase.isPresent()) {
      return phrase;
    }
    Optional<AnswerExtraction> hashes = lastGroup(HASHES, text);
    if (hashes.isPresent()) {
      return hashes;
    }
    if (!lastParagraphOnly) {
      return lastNumber(text, 0);
    }
    int paragraphStart = 0;
    Matcher breaks = PARAGRAPH_BREAK.matcher(text);
    int searchEnd = text.stripTrailing().length();
    while (breaks.find()) {
      if (breaks.end() <= searchEnd) {
        paragraphStart = breaks.end();
      }
    }
    return lastNumber(text.substring(paragraphStart), paragraphStart);
  }

  private static Optional<AnswerExtraction> lastGroup(Pattern pattern, String text) {
    Matcher matcher = pattern.matcher(text);
    AnswerExtraction last = null;
    while (matcher.find()) {
      String canonical = canonical(matcher.group(1));
      if (canonical != null) {
        last = new AnswerExtraction(canonical, matcher.start(1), matcher.end(1));
      }
    }
    return Optional.ofNullable(last);
  }

  private static Optional<AnswerExtraction> lastNumber(String text, int offset) {
    Matcher matcher = NUMBER_PATTERN.matcher(text);
    AnswerExtraction last = null;
    while (matcher.find()) {
      String canonical = canonical(matcher.group());
      if (canonical != null) {
        last = new AnswerExtraction(canonical, matcher.start() + offset, matcher.end() + offset);
      }
    }
    return Optional.ofNullable(last);
  }

  /** Canonical plain decimal without grouping commas or trailing zeros; null if not a number. */
  static String canonical(String raw) {
    String cleaned = raw.replace(",", "").replace("\\$", "").replace("$", "").trim();
    while (cleaned.endsWith(".")) {
      cleaned = cleaned.substring(0, cleaned.length() - 1);
    }
    if (cleaned.isEmpty() || "-".equals(cleaned)) {
      return null;
    }
    try {
      BigDecimal value = new BigDecimal(cleaned).stripTrailingZeros();
      if (value.signum() == 0) {
        return "0";
      }
      return value.toPlainString();
    } catch (NumberFormatException notANumber) {
      return null;
    }
  }

  @Override
  public boolean equivalent(String left, String right) {
    String a = canonical(left);
    String b = canonical(right);
    if (a == null || b == null) {
      return left.trim().equals(right.trim());
    }
    return new BigDecimal(a).compareTo(new BigDecimal(b)) == 0;
  }

  @Override
  public String renderGold(String gold) {
    return "The final answer is \\boxed{" + gold + "}.";
  }

  @Override
  public String id() {
    return "number";
  }
}
