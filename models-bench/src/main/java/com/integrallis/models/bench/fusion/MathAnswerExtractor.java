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
 * MATH-style extraction: the last brace-matched {@code \boxed{}} (or {@code \fbox{}}), else an
 * "answer is $...$" phrase, else "answer is ..." to the end of the sentence. Equivalence uses
 * {@link #normalize(String)}, a port of the Hendrycks MATH {@code strip_string} rules plus exact
 * rational comparison of plain numbers and integer fractions.
 */
final class MathAnswerExtractor implements AnswerExtractor {

  private static final Pattern DOLLAR_PHRASE =
      Pattern.compile("(?i)answer\\s*(?:is|:)\\s*:?\\s*\\$([^$]+)\\$");
  private static final Pattern SENTENCE_PHRASE =
      Pattern.compile("(?i)answer\\s*(?:is|:)\\s*:?\\s*([^\\n$]+?)(?=\\.\\s|\\.$|\\n|$)");
  private static final Pattern DECIMAL = Pattern.compile("^-?\\d+(?:\\.\\d+)?$");
  private static final Pattern FRACTION =
      Pattern.compile("^(-?)\\\\frac\\{(-?\\d+)\\}\\{(-?\\d+)\\}$");
  private static final Pattern THOUSANDS = Pattern.compile("^-?\\d{1,3}(?:,\\d{3})+(?:\\.\\d+)?$");
  private static final Pattern SLASH_FRACTION = Pattern.compile("^(-?\\d+)/(\\d+)$");

  @Override
  public Optional<AnswerExtraction> stated(String finalText, List<String> labels) {
    return extract(finalText);
  }

  @Override
  public Optional<AnswerExtraction> reasoning(String reasoningText, List<String> labels) {
    return extract(reasoningText);
  }

  private static Optional<AnswerExtraction> extract(String text) {
    Optional<AnswerExtraction> boxed = AnswerExtractor.lastBoxed(text);
    if (boxed.isPresent()) {
      return boxed.map(MathAnswerExtractor::trimmed).filter(a -> !a.value().isEmpty());
    }
    Optional<AnswerExtraction> dollar = lastGroup(DOLLAR_PHRASE, text);
    if (dollar.isPresent()) {
      return dollar;
    }
    return lastGroup(SENTENCE_PHRASE, text);
  }

  private static AnswerExtraction trimmed(AnswerExtraction raw) {
    String value = raw.value();
    int lead = value.length() - value.stripLeading().length();
    String stripped = value.strip();
    return new AnswerExtraction(
        stripped, raw.start() + lead, raw.start() + lead + stripped.length());
  }

  private static Optional<AnswerExtraction> lastGroup(Pattern pattern, String text) {
    Matcher matcher = pattern.matcher(text);
    AnswerExtraction last = null;
    while (matcher.find()) {
      last = trimmed(new AnswerExtraction(matcher.group(1), matcher.start(1), matcher.end(1)));
    }
    return Optional.ofNullable(last).filter(a -> !a.value().isEmpty());
  }

  @Override
  public boolean equivalent(String left, String right) {
    String a = normalize(left);
    String b = normalize(right);
    if (a.equals(b)) {
      return true;
    }
    BigDecimal[] x = rational(a);
    BigDecimal[] y = rational(b);
    return x != null && y != null && x[0].multiply(y[1]).compareTo(y[0].multiply(x[1])) == 0;
  }

  /** Returns {numerator, denominator} for plain decimals and integer fractions, else null. */
  private static BigDecimal[] rational(String normalized) {
    if (DECIMAL.matcher(normalized).matches()) {
      return new BigDecimal[] {new BigDecimal(normalized), BigDecimal.ONE};
    }
    Matcher fraction = FRACTION.matcher(normalized);
    if (fraction.matches()) {
      BigDecimal denominator = new BigDecimal(fraction.group(3));
      if (denominator.signum() == 0) {
        return null;
      }
      BigDecimal numerator = new BigDecimal(fraction.group(2));
      if (!fraction.group(1).isEmpty()) {
        numerator = numerator.negate();
      }
      return new BigDecimal[] {numerator, denominator};
    }
    return null;
  }

  /** Canonical string form used for equivalence. */
  static String normalize(String answer) {
    String s = answer.strip().replace("\n", "").replace("\\!", "");
    s = s.replace("\\\\", "\\");
    s = s.replace("\\tfrac", "\\frac").replace("\\dfrac", "\\frac");
    s = s.replace("\\left", "").replace("\\right", "");
    s = s.replace("^{\\circ}", "").replace("^\\circ", "");
    s = s.replace("\\$", "").replace("$", "");
    s = unwrap(s, "\\text{");
    s = unwrap(s, "\\textbf{");
    s = unwrap(s, "\\mbox{");
    s = unwrap(s, "\\mathrm{");
    s = s.replace("\\%", "").replace("%", "");
    s = s.replace("\\,", "").replace("\\;", "").replace("\\:", "").replace("\\ ", " ");
    s = s.replace(" .", " 0.").replace("{.", "{0.");
    if (s.startsWith(".")) {
      s = "0" + s;
    }
    String[] sides = s.split("=", -1);
    if (sides.length == 2 && sides[0].strip().length() <= 2) {
      s = sides[1];
    }
    s = s.replaceAll("\\\\sqrt(\\w)", "\\\\sqrt{$1}");
    s = s.replace(" ", "");
    s = s.replaceAll("\\\\frac(\\d)(\\d)", "\\\\frac{$1}{$2}");
    s = s.replaceAll("\\\\frac\\{([^{}]*)\\}(\\w)", "\\\\frac{$1}{$2}");
    s = s.replaceAll("\\\\frac(\\w)\\{", "\\\\frac{$1}{");
    Matcher slash = SLASH_FRACTION.matcher(s);
    if (slash.matches()) {
      s = "\\frac{" + slash.group(1) + "}{" + slash.group(2) + "}";
    }
    if (THOUSANDS.matcher(s).matches()) {
      s = s.replace(",", "");
    }
    while (s.endsWith(".")) {
      s = s.substring(0, s.length() - 1);
    }
    return s;
  }

  private static String unwrap(String s, String command) {
    int index = s.indexOf(command);
    while (index >= 0) {
      int open = index + command.length();
      int depth = 1;
      int cursor = open;
      while (cursor < s.length() && depth > 0) {
        char c = s.charAt(cursor);
        if (c == '\\' && cursor + 1 < s.length()) {
          cursor += 2;
          continue;
        }
        if (c == '{') {
          depth++;
        } else if (c == '}') {
          depth--;
        }
        cursor++;
      }
      if (depth != 0) {
        break;
      }
      s = s.substring(0, index) + s.substring(open, cursor - 1) + s.substring(cursor);
      index = s.indexOf(command, index);
    }
    return s;
  }

  @Override
  public String renderGold(String gold) {
    return "The final answer is $\\boxed{" + gold + "}$.";
  }

  @Override
  public String id() {
    return "math";
  }
}
