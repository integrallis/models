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
package com.integrallis.models.audio;

import java.text.Normalizer;
import java.util.LinkedHashMap;
import java.util.Locale;
import java.util.Map;
import java.util.Objects;
import java.util.function.Function;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/** English input normalization used by the official Soprano 1.1 inference pipeline. */
final class SopranoTextNormalizer {

  private static final Pattern COMMA_NUMBER = Pattern.compile("(\\d[\\d,]+\\d)");
  private static final Pattern DATE =
      Pattern.compile("(^|[^/])(\\d\\d?[/-]\\d\\d?[/-]\\d\\d(?:\\d\\d)?)($|[^/])");
  private static final Pattern PHONE =
      Pattern.compile("(\\(?\\d{3}\\)?[-.\\s]\\d{3}[-.\\s]?\\d{4})");
  private static final Pattern TIME = Pattern.compile("(\\d\\d?:\\d\\d(?::\\d\\d)?)");
  private static final Pattern DOLLARS = Pattern.compile("\\$([\\d.,]*\\d+)");
  private static final Pattern DECIMAL = Pattern.compile("(\\d+(?:\\.\\d+)+)");
  private static final Pattern ORDINAL = Pattern.compile("\\d+(?:st|nd|rd|th)");
  private static final Pattern NUMBER = Pattern.compile("\\d+");
  private static final Pattern MIXED_CASE = Pattern.compile("\\b(?:[A-Z][a-z]*)+\\b");
  private static final Pattern LETTER_RUN = Pattern.compile("[A-Z][a-z]*");

  private static final String[] SMALL = {
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen"
  };
  private static final String[] TENS = {
    "", "", "twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"
  };
  private static final String[] ORDINAL_SMALL = {
    "zeroth",
    "first",
    "second",
    "third",
    "fourth",
    "fifth",
    "sixth",
    "seventh",
    "eighth",
    "ninth",
    "tenth",
    "eleventh",
    "twelfth",
    "thirteenth",
    "fourteenth",
    "fifteenth",
    "sixteenth",
    "seventeenth",
    "eighteenth",
    "nineteenth"
  };
  private static final String[] ORDINAL_TENS = {
    "",
    "",
    "twentieth",
    "thirtieth",
    "fortieth",
    "fiftieth",
    "sixtieth",
    "seventieth",
    "eightieth",
    "ninetieth"
  };

  private static final Map<String, String> PERIOD_ABBREVIATIONS =
      Map.ofEntries(
          Map.entry("mrs", "misess"),
          Map.entry("ms", "miss"),
          Map.entry("mr", "mister"),
          Map.entry("dr", "doctor"),
          Map.entry("st", "saint"),
          Map.entry("co", "company"),
          Map.entry("jr", "junior"),
          Map.entry("maj", "major"),
          Map.entry("gen", "general"),
          Map.entry("drs", "doctors"),
          Map.entry("rev", "reverend"),
          Map.entry("lt", "lieutenant"),
          Map.entry("hon", "honorable"),
          Map.entry("sgt", "sergeant"),
          Map.entry("capt", "captain"),
          Map.entry("esq", "esquire"),
          Map.entry("ltd", "limited"),
          Map.entry("col", "colonel"),
          Map.entry("ft", "fort"));

  private static final Map<String, String> CASED_ABBREVIATIONS = casedAbbreviations();

  private SopranoTextNormalizer() {}

  /**
   * Normalizes text in the same order as {@code ekwek/Soprano-1.1-80M}'s Apache-2.0 {@code
   * clean_text}: transliteration, spoken numbers, abbreviations, symbols, and whitespace.
   */
  static String normalize(String input) {
    Objects.requireNonNull(input, "input");
    String text = transliterate(input.replace("—", " - "));
    text = normalizeNewlines(text);
    text = normalizeNumbers(text);
    text = normalizeSpecial(text);
    text = expandAbbreviations(text);
    text = normalizeMixedCase(text);
    text = expandSpecialCharacters(text);
    text = text.toLowerCase(Locale.ROOT);
    text = removeUnknownCharacters(text);
    text = text.replaceAll("\\s+", " ").replaceAll(" ([.!?,])", "$1");
    text = text.trim();
    text = deduplicatePunctuation(text);
    return collapseTripleLetters(text);
  }

  private static String normalizeNewlines(String text) {
    String[] lines = text.split("\\n", -1);
    StringBuilder result = new StringBuilder(text.length() + lines.length);
    for (String value : lines) {
      String line = value.trim();
      if (!line.isEmpty() && ".!?".indexOf(line.charAt(line.length() - 1)) < 0) {
        line += ".";
      }
      if (!result.isEmpty()) {
        result.append(' ');
      }
      result.append(line);
    }
    return result.toString();
  }

  private static String normalizeNumbers(String text) {
    text = replaceMatches(Pattern.compile("#\\d"), text, value -> "number " + value.charAt(1));
    text =
        replaceMatches(
            Pattern.compile("\\b\\d+[KMBT]\\b", Pattern.CASE_INSENSITIVE),
            text,
            value ->
                value.substring(0, value.length() - 1)
                    + " "
                    + switch (Character.toUpperCase(value.charAt(value.length() - 1))) {
                      case 'K' -> "thousand";
                      case 'M' -> "million";
                      case 'B' -> "billion";
                      case 'T' -> "trillion";
                      default -> throw new IllegalStateException();
                    });
    text = replaceMatches(COMMA_NUMBER, text, value -> value.replace(",", ""));
    text =
        replaceGroups(
            DATE,
            text,
            matcher ->
                matcher.group(1)
                    + String.join(" dash ", matcher.group(2).split("[./-]"))
                    + matcher.group(3));
    text =
        replaceMatches(
            PHONE,
            text,
            value -> {
              String digits = value.replaceAll("\\D", "");
              return spacedDigits(digits.substring(0, 3))
                  + ", "
                  + spacedDigits(digits.substring(3, 6))
                  + ", "
                  + spacedDigits(digits.substring(6));
            });
    text = replaceMatches(TIME, text, SopranoTextNormalizer::spokenTime);
    text = text.replaceAll("£([\\d,]*\\d+)", "$1 pounds");
    text = replaceGroups(DOLLARS, text, matcher -> spokenDollars(matcher.group(1)));
    text =
        replaceMatches(
            DECIMAL,
            text,
            value -> {
              String[] parts = value.split("\\.");
              StringBuilder result = new StringBuilder(parts[0]);
              for (int index = 1; index < parts.length; index++) {
                result.append(" point ").append(spacedDigits(parts[index]));
              }
              return result.toString();
            });
    text = replaceBinaryOperator(text, "\\*", " times ");
    text = replaceBinaryOperator(text, "/", " over ");
    text = replaceBinaryOperator(text, "\\+", " plus ");
    text =
        replaceMatches(
            Pattern.compile("\\d?\\s?-\\s?\\d"),
            text,
            value -> joinOperator(value, "-", " minus "));
    text =
        replaceMatches(
            Pattern.compile("\\d+(?:/\\d+)+"),
            text,
            value -> String.join(" over ", value.split("/")));
    text =
        replaceMatches(
            ORDINAL,
            text,
            value -> ordinalWords(Long.parseLong(value.substring(0, value.length() - 2))));
    for (int pass = 0; pass < 2; pass++) {
      text =
          replaceGroups(
              Pattern.compile("(\\d[a-z]|[a-z]\\d)", Pattern.CASE_INSENSITIVE),
              text,
              matcher -> matcher.group(1).charAt(0) + " " + matcher.group(1).charAt(1));
    }
    return replaceMatches(NUMBER, text, value -> numberWords(Long.parseLong(value)));
  }

  private static String replaceBinaryOperator(String text, String operator, String spoken) {
    Pattern pattern = Pattern.compile("\\d\\s?" + operator + "\\s?\\d");
    String literal = operator.replace("\\", "");
    return replaceMatches(pattern, text, value -> joinOperator(value, literal, spoken));
  }

  private static String joinOperator(String value, String operator, String spoken) {
    return String.join(spoken, value.split("\\s*" + Pattern.quote(operator) + "\\s*"));
  }

  private static String spokenTime(String value) {
    String[] parts = value.split(":");
    int hours = Integer.parseInt(parts[0]);
    if (parts.length == 2) {
      String minutes = parts[1];
      if ("00".equals(minutes)) {
        if (hours == 0) {
          return "0";
        }
        return hours > 12 ? parts[0] + " minutes" : parts[0] + " o'clock";
      }
      return parts[0] + " " + (minutes.startsWith("0") ? "oh " + minutes.charAt(1) : minutes);
    }
    String minutes = parts[1];
    String seconds = parts[2];
    if (hours != 0) {
      return parts[0]
          + " "
          + twoDigitClockPart(minutes)
          + ("00".equals(seconds) ? "" : " " + twoDigitClockPart(seconds));
    }
    if (!"00".equals(minutes)) {
      return minutes + " " + twoDigitClockPart(seconds);
    }
    return seconds;
  }

  private static String twoDigitClockPart(String value) {
    if ("00".equals(value)) {
      return "oh oh";
    }
    return value.startsWith("0") ? "oh " + value.charAt(1) : value;
  }

  private static String spokenDollars(String value) {
    String[] parts = value.replace(",", "").split("\\.", -1);
    if (parts.length > 2) {
      return value + " dollars";
    }
    long dollars = parts[0].isEmpty() ? 0 : Long.parseLong(parts[0]);
    long cents = parts.length == 2 && !parts[1].isEmpty() ? Long.parseLong(parts[1]) : 0;
    if (dollars != 0 && cents != 0) {
      return dollars
          + (dollars == 1 ? " dollar, " : " dollars, ")
          + cents
          + (cents == 1 ? " cent" : " cents");
    }
    if (dollars != 0) {
      return dollars + (dollars == 1 ? " dollar" : " dollars");
    }
    if (cents != 0) {
      return cents + (cents == 1 ? " cent" : " cents");
    }
    return "zero dollars";
  }

  private static String normalizeSpecial(String text) {
    text = text.replaceAll("https?://", "h t t p s colon slash slash ");
    text =
        replaceMatches(
            Pattern.compile(". - ."), text, value -> value.charAt(0) + ", " + value.charAt(4));
    text =
        replaceMatches(
            Pattern.compile("[A-Z]\\.[A-Z]", Pattern.CASE_INSENSITIVE),
            text,
            value -> value.charAt(0) + " dot " + value.charAt(2));
    return replaceMatches(
        Pattern.compile("[({\\[].*[)}\\]](?:.|$)"),
        text,
        value ->
            value
                .replaceAll("[({\\[]", ", ")
                .replaceAll("[)}\\]][^$.!?,]", ", ")
                .replaceAll("[)}\\]]", ""));
  }

  private static String expandAbbreviations(String text) {
    for (Map.Entry<String, String> entry : PERIOD_ABBREVIATIONS.entrySet()) {
      text =
          text.replaceAll(
              "(?i)\\b" + Pattern.quote(entry.getKey()) + "\\.",
              Matcher.quoteReplacement(entry.getValue()));
    }
    for (Map.Entry<String, String> entry : CASED_ABBREVIATIONS.entrySet()) {
      text =
          text.replaceAll(
              "\\b" + Pattern.quote(entry.getKey()) + "\\b",
              Matcher.quoteReplacement(entry.getValue()));
    }
    return text;
  }

  private static String normalizeMixedCase(String text) {
    return replaceMatches(
        MIXED_CASE,
        text,
        value -> {
          Matcher parts = LETTER_RUN.matcher(value);
          StringBuilder result = new StringBuilder();
          int count = 0;
          while (parts.find()) {
            if (!result.isEmpty()) {
              result.append(' ');
            }
            result.append(parts.group());
            count++;
          }
          if (count == 1 || count == value.length()) {
            return value;
          }
          if (count == value.length() - 1 && value.endsWith("s")) {
            return value.substring(0, value.length() - 1) + "'s";
          }
          return result.toString();
        });
  }

  private static String expandSpecialCharacters(String text) {
    return text.replace("@", " at ")
        .replace("&", " and ")
        .replace("%", " percent ")
        .replace(':', '.')
        .replace(';', ',')
        .replace("+", " plus ")
        .replace("\\", " backslash ")
        .replace("~", " about ")
        .replaceAll("(^| )<3", " heart ")
        .replace("<=", " less than or equal to ")
        .replace(">=", " greater than or equal to ")
        .replace("<", " less than ")
        .replace(">", " greater than ")
        .replace("=", " equals ")
        .replace("/", " slash ")
        .replace("_", " ")
        .replace("*", " ");
  }

  private static String transliterate(String text) {
    text =
        text.replace('“', '"')
            .replace('”', '"')
            .replace('‘', '\'')
            .replace('’', '\'')
            .replace('–', '-')
            .replace("€", "EUR")
            .replace("©", "(c)")
            .replace("®", "(r)");
    String decomposed = Normalizer.normalize(text, Normalizer.Form.NFKD);
    StringBuilder ascii = new StringBuilder(decomposed.length());
    decomposed
        .codePoints()
        .filter(codePoint -> Character.getType(codePoint) != Character.NON_SPACING_MARK)
        .filter(codePoint -> codePoint <= 0x7f || codePoint == '£')
        .forEach(ascii::appendCodePoint);
    return ascii.toString();
  }

  private static String removeUnknownCharacters(String text) {
    StringBuilder known = new StringBuilder(text.length());
    for (int index = 0; index < text.length(); index++) {
      char value = text.charAt(index);
      if (Character.isLetterOrDigit(value) || " !$%&'*+,-./<>?_".indexOf(value) >= 0) {
        known.append(value);
      }
    }
    return known.toString().replaceAll("[<>/_+]", "");
  }

  private static String deduplicatePunctuation(String text) {
    return text.replaceAll("\\.\\.+", "[ELLIPSIS]")
        .replaceAll(",+", ",")
        .replaceAll("[.,]*\\.[.,]*", ".")
        .replaceAll("[.,!]*![.,!]*", "!")
        .replaceAll("[.,!?]*\\?[.,!?]*", "?")
        .replace("[ELLIPSIS]", "...");
  }

  private static String collapseTripleLetters(String text) {
    return replaceMatches(Pattern.compile("(\\w)\\1{2,}"), text, value -> value.substring(0, 2));
  }

  private static String numberWords(long number) {
    if (number > 1000 && number < 3000) {
      if (number == 2000) {
        return "two thousand";
      }
      if (number > 2000 && number < 2010) {
        return "two thousand " + belowHundred((int) (number % 100));
      }
      if (number % 100 == 0) {
        return numberWords(number / 100) + " hundred";
      }
      return groupedPairs(number);
    }
    return cardinal(number);
  }

  private static String groupedPairs(long number) {
    String digits = Long.toString(number);
    StringBuilder result = new StringBuilder();
    int first = digits.length() % 2;
    int index = 0;
    if (first != 0) {
      result.append(SMALL[digits.charAt(0) - '0']);
      index = 1;
    }
    while (index < digits.length()) {
      int pair = Integer.parseInt(digits.substring(index, index + 2));
      if (!result.isEmpty()) {
        result.append(' ');
      }
      result.append(pair < 10 ? "oh " + SMALL[pair] : belowHundred(pair));
      index += 2;
    }
    return result.toString();
  }

  private static String cardinal(long number) {
    if (number < 100) {
      return belowHundred((int) number);
    }
    if (number < 1000) {
      return SMALL[(int) (number / 100)]
          + " hundred"
          + (number % 100 == 0 ? "" : " " + belowHundred((int) (number % 100)));
    }
    long[] sizes = {1_000_000_000_000L, 1_000_000_000L, 1_000_000L, 1_000L};
    String[] names = {"trillion", "billion", "million", "thousand"};
    for (int index = 0; index < sizes.length; index++) {
      if (number >= sizes[index]) {
        long rest = number % sizes[index];
        return cardinal(number / sizes[index])
            + " "
            + names[index]
            + (rest == 0 ? "" : " " + cardinal(rest));
      }
    }
    throw new IllegalStateException("unreachable number " + number);
  }

  private static String belowHundred(int number) {
    if (number < 20) {
      return SMALL[number];
    }
    return TENS[number / 10] + (number % 10 == 0 ? "" : "-" + SMALL[number % 10]);
  }

  private static String ordinalWords(long number) {
    if (number < 20) {
      return ORDINAL_SMALL[(int) number];
    }
    if (number < 100) {
      return number % 10 == 0
          ? ORDINAL_TENS[(int) (number / 10)]
          : TENS[(int) (number / 10)] + "-" + ORDINAL_SMALL[(int) (number % 10)];
    }
    if (number < 1000 && number % 100 == 0) {
      return SMALL[(int) (number / 100)] + " hundredth";
    }
    if (number < 1000) {
      return cardinal(number - number % 100) + " " + ordinalWords(number % 100);
    }
    return cardinal(number) + "th";
  }

  private static String spacedDigits(String digits) {
    return String.join(" ", digits.split(""));
  }

  private static String replaceMatches(
      Pattern pattern, String input, Function<String, String> replacement) {
    return replaceGroups(pattern, input, matcher -> replacement.apply(matcher.group()));
  }

  private static String replaceGroups(
      Pattern pattern, String input, Function<Matcher, String> replacement) {
    Matcher matcher = pattern.matcher(input);
    StringBuilder result = new StringBuilder(input.length());
    while (matcher.find()) {
      matcher.appendReplacement(result, Matcher.quoteReplacement(replacement.apply(matcher)));
    }
    matcher.appendTail(result);
    return result.toString();
  }

  private static Map<String, String> casedAbbreviations() {
    Map<String, String> values = new LinkedHashMap<>();
    values.put("Hz", "hertz");
    values.put("kHz", "kilohertz");
    values.put("KBs", "kilobytes");
    values.put("KB", "kilobyte");
    values.put("MBs", "megabytes");
    values.put("MB", "megabyte");
    values.put("GBs", "gigabytes");
    values.put("GB", "gigabyte");
    values.put("TBs", "terabytes");
    values.put("TB", "terabyte");
    values.put("APIs", "a p i's");
    values.put("API", "a p i");
    values.put("CLIs", "c l i's");
    values.put("CLI", "c l i");
    values.put("CPUs", "c p u's");
    values.put("CPU", "c p u");
    values.put("GPUs", "g p u's");
    values.put("GPU", "g p u");
    values.put("Ave", "avenue");
    values.put("etc", "et cetera");
    values.put("Mon", "monday");
    values.put("Tues", "tuesday");
    values.put("Wed", "wednesday");
    values.put("Thurs", "thursday");
    values.put("Fri", "friday");
    values.put("Sat", "saturday");
    values.put("Jan", "january");
    values.put("Feb", "february");
    values.put("Mar", "march");
    values.put("Apr", "april");
    values.put("Aug", "august");
    values.put("Sept", "september");
    values.put("Oct", "october");
    values.put("Nov", "november");
    values.put("Dec", "december");
    values.put("and/or", "and or");
    return values;
  }
}
