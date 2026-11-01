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
package com.integrallis.models.audio.phonemizer;

import java.math.BigDecimal;
import java.util.ArrayList;
import java.util.List;

/**
 * English number words in the style the misaki phonemizer expects from its number library:
 * cardinals with {@code and} before a final tens group ({@code one hundred and five}), ordinals,
 * two-part years ({@code nineteen ninety-eight}, {@code eighteen oh-five}) and decimals read digit
 * by digit after {@code point}. Written from the conventions, not from any existing library.
 */
public final class EnglishNumbers {

  private static final String[] ONES = {
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
  private static final String[] ONES_ORDINAL = {
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
  private static final String[] TENS_ORDINAL = {
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
  private static final String[] SCALES = {
    "", "thousand", "million", "billion", "trillion", "quadrillion", "quintillion"
  };

  private EnglishNumbers() {}

  /** {@code 1998} becomes {@code one thousand, nine hundred and ninety-eight}. */
  public static String cardinal(long value) {
    if (value < 0) {
      return "minus " + cardinal(-value);
    }
    if (value < 20) {
      return ONES[(int) value];
    }
    List<String> groups = new ArrayList<>();
    long remaining = value;
    int scale = 0;
    while (remaining > 0) {
      int group = (int) (remaining % 1000);
      if (group != 0) {
        String words = belowThousand(group);
        groups.add(0, scale == 0 ? words : words + " " + SCALES[scale]);
      }
      remaining /= 1000;
      scale++;
    }
    // "and" joins a final group below one hundred to the groups above it, as in
    // "one thousand and five"; groups are otherwise separated by commas.
    StringBuilder text = new StringBuilder();
    for (int i = 0; i < groups.size(); i++) {
      if (i > 0) {
        boolean last = i == groups.size() - 1;
        text.append(last && value % 1000 < 100 && value % 1000 != 0 ? " and " : ", ");
      }
      text.append(groups.get(i));
    }
    return text.toString();
  }

  private static String belowThousand(int value) {
    StringBuilder text = new StringBuilder();
    int hundreds = value / 100;
    int rest = value % 100;
    if (hundreds > 0) {
      text.append(ONES[hundreds]).append(" hundred");
      if (rest > 0) {
        text.append(" and ");
      }
    }
    if (rest > 0) {
      text.append(belowHundred(rest));
    }
    return text.toString();
  }

  private static String belowHundred(int value) {
    if (value < 20) {
      return ONES[value];
    }
    return value % 10 == 0 ? TENS[value / 10] : TENS[value / 10] + "-" + ONES[value % 10];
  }

  /**
   * {@code 22} becomes {@code twenty-second}, {@code 101} becomes {@code one hundred and first}.
   */
  public static String ordinal(long value) {
    if (value < 0) {
      return "minus " + ordinal(-value);
    }
    if (value < 20) {
      return ONES_ORDINAL[(int) value];
    }
    String words = cardinal(value);
    int lastSpace = Math.max(words.lastIndexOf(' '), words.lastIndexOf('-'));
    String head = lastSpace < 0 ? "" : words.substring(0, lastSpace + 1);
    String tail = words.substring(lastSpace + 1);
    return head + ordinalWord(tail);
  }

  private static String ordinalWord(String word) {
    for (int i = 0; i < ONES.length; i++) {
      if (ONES[i].equals(word)) {
        return ONES_ORDINAL[i];
      }
    }
    for (int i = 2; i < TENS.length; i++) {
      if (TENS[i].equals(word)) {
        return TENS_ORDINAL[i];
      }
    }
    return word + "th";
  }

  /**
   * Years read in two halves: {@code 1998} is {@code nineteen ninety-eight}, {@code 1805} is {@code
   * eighteen oh-five}, {@code 1900} is {@code nineteen hundred}; {@code 2005} and anything with a
   * zero hundreds digit and a single-digit remainder fall back to the cardinal.
   */
  public static String year(long value) {
    if (value < 0) {
      return cardinal(value);
    }
    long high = value / 100;
    long low = value % 100;
    if (high == 0 || (high % 10 == 0 && low < 10) || high >= 100) {
      return cardinal(value);
    }
    String lowText;
    if (low == 0) {
      lowText = "hundred";
    } else if (low < 10) {
      lowText = "oh-" + cardinal(low);
    } else {
      lowText = cardinal(low);
    }
    return cardinal(high) + " " + lowText;
  }

  /** {@code 4.25} becomes {@code four point two five}; the fraction is read digit by digit. */
  public static String decimal(String literal) {
    BigDecimal value = new BigDecimal(literal.replace(",", "")).stripTrailingZeros();
    String plain = value.toPlainString();
    boolean negative = plain.startsWith("-");
    if (negative) {
      plain = plain.substring(1);
    }
    int dot = plain.indexOf('.');
    String integer = dot < 0 ? plain : plain.substring(0, dot);
    StringBuilder text = new StringBuilder();
    if (negative) {
      text.append("minus ");
    }
    text.append(cardinal(Long.parseLong(integer.isEmpty() ? "0" : integer)));
    if (dot >= 0) {
      text.append(" point");
      for (char digit : plain.substring(dot + 1).toCharArray()) {
        text.append(' ').append(ONES[digit - '0']);
      }
    }
    return text.toString();
  }
}
