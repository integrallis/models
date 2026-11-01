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

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class EnglishNumbersTest {

  @Test
  void readsCardinalsWithAndBeforeAFinalSmallGroup() {
    assertThat(EnglishNumbers.cardinal(0)).isEqualTo("zero");
    assertThat(EnglishNumbers.cardinal(7)).isEqualTo("seven");
    assertThat(EnglishNumbers.cardinal(19)).isEqualTo("nineteen");
    assertThat(EnglishNumbers.cardinal(20)).isEqualTo("twenty");
    assertThat(EnglishNumbers.cardinal(42)).isEqualTo("forty-two");
    assertThat(EnglishNumbers.cardinal(100)).isEqualTo("one hundred");
    assertThat(EnglishNumbers.cardinal(105)).isEqualTo("one hundred and five");
    assertThat(EnglishNumbers.cardinal(999)).isEqualTo("nine hundred and ninety-nine");
    assertThat(EnglishNumbers.cardinal(1000)).isEqualTo("one thousand");
    assertThat(EnglishNumbers.cardinal(1001)).isEqualTo("one thousand and one");
    assertThat(EnglishNumbers.cardinal(1100)).isEqualTo("one thousand, one hundred");
    assertThat(EnglishNumbers.cardinal(1998))
        .isEqualTo("one thousand, nine hundred and ninety-eight");
    assertThat(EnglishNumbers.cardinal(12_000)).isEqualTo("twelve thousand");
    assertThat(EnglishNumbers.cardinal(4_200_000_000L))
        .isEqualTo("four billion, two hundred million");
    assertThat(EnglishNumbers.cardinal(1_000_001)).isEqualTo("one million and one");
    assertThat(EnglishNumbers.cardinal(-15)).isEqualTo("minus fifteen");
  }

  @Test
  void readsOrdinals() {
    assertThat(EnglishNumbers.ordinal(1)).isEqualTo("first");
    assertThat(EnglishNumbers.ordinal(3)).isEqualTo("third");
    assertThat(EnglishNumbers.ordinal(5)).isEqualTo("fifth");
    assertThat(EnglishNumbers.ordinal(8)).isEqualTo("eighth");
    assertThat(EnglishNumbers.ordinal(9)).isEqualTo("ninth");
    assertThat(EnglishNumbers.ordinal(12)).isEqualTo("twelfth");
    assertThat(EnglishNumbers.ordinal(20)).isEqualTo("twentieth");
    assertThat(EnglishNumbers.ordinal(22)).isEqualTo("twenty-second");
    assertThat(EnglishNumbers.ordinal(101)).isEqualTo("one hundred and first");
    assertThat(EnglishNumbers.ordinal(1000)).isEqualTo("one thousandth");
    assertThat(EnglishNumbers.ordinal(2015)).isEqualTo("two thousand and fifteenth");
    assertThat(EnglishNumbers.ordinal(-2)).isEqualTo("minus second");
  }

  @Test
  void readsYearsInTwoHalvesExceptWhereTheConventionDoesNot() {
    assertThat(EnglishNumbers.year(1998)).isEqualTo("nineteen ninety-eight");
    assertThat(EnglishNumbers.year(1805)).isEqualTo("eighteen oh-five");
    assertThat(EnglishNumbers.year(1900)).isEqualTo("nineteen hundred");
    assertThat(EnglishNumbers.year(2010)).isEqualTo("twenty ten");
    assertThat(EnglishNumbers.year(2000)).isEqualTo("two thousand");
    assertThat(EnglishNumbers.year(2005)).isEqualTo("two thousand and five");
    assertThat(EnglishNumbers.year(2026)).isEqualTo("twenty twenty-six");
    assertThat(EnglishNumbers.year(99)).isEqualTo("ninety-nine");
    assertThat(EnglishNumbers.year(12345))
        .isEqualTo("twelve thousand, three hundred and forty-five");
    assertThat(EnglishNumbers.year(-44)).isEqualTo("minus forty-four");
  }

  @Test
  void readsDecimalsDigitByDigit() {
    assertThat(EnglishNumbers.decimal("4.2")).isEqualTo("four point two");
    assertThat(EnglishNumbers.decimal("4.25")).isEqualTo("four point two five");
    assertThat(EnglishNumbers.decimal("4.20")).isEqualTo("four point two");
    assertThat(EnglishNumbers.decimal("3.14159")).isEqualTo("three point one four one five nine");
    assertThat(EnglishNumbers.decimal("1,234.5"))
        .isEqualTo("one thousand, two hundred and thirty-four point five");
    assertThat(EnglishNumbers.decimal("12")).isEqualTo("twelve");
    assertThat(EnglishNumbers.decimal("-0.5")).isEqualTo("minus zero point five");
    assertThat(EnglishNumbers.decimal("0.0")).isEqualTo("zero");
  }
}
