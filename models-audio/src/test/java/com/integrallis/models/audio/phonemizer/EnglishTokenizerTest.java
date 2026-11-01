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

import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class EnglishTokenizerTest {

  private static List<String> texts(String input) {
    return EnglishTokenizer.tokenize(input).stream().map(tk -> tk.text).toList();
  }

  private static List<String> tags(String input) {
    return EnglishTokenizer.tokenize(input).stream().map(tk -> tk.tag).toList();
  }

  @Test
  void peelsPunctuationAndKeepsContractionsAndAbbreviations() {
    assertThat(texts("Dr. Smith arrived at 3 p.m., didn't he?"))
        .containsExactly("Dr.", "Smith", "arrived", "at", "3", "p.m.", ",", "didn't", "he", "?");
    assertThat(texts("\"Hello,\" she said (quietly)."))
        .containsExactly("\"", "Hello", ",", "\"", "she", "said", "(", "quietly", ")", ".");
    assertThat(texts("#1 fan")).containsExactly("#", "1", "fan");
    assertThat(texts("well-known U.S. prices: $4.20 and 1,000."))
        .containsExactly(
            "well", "-", "known", "U.S.", "prices", ":", "$", "4.20", "and", "1,000", ".");
    assertThat(texts("the dogs' bowls")).containsExactly("the", "dogs'", "bowls");
    assertThat(texts("wait… what—now?")).containsExactly("wait", "…", "what", "—", "now", "?");
    assertThat(texts("  leading   and trailing  ")).containsExactly("leading", "and", "trailing");
    assertThat(texts("'quoted'")).containsExactly("'", "quoted", "'");
  }

  @Test
  void recordsTrailingWhitespacePerToken() {
    List<MToken> tokens = EnglishTokenizer.tokenize("Hi, there.");
    assertThat(tokens.stream().map(tk -> tk.whitespace).toList()).containsExactly("", " ", "", "");
  }

  @Test
  void assignsTheTagsTheLexiconRulesRead() {
    assertThat(tags("I am the one to blame."))
        .containsExactly("PRP", "VBP", "DT", "NN", "TO", "NN", ".");
    assertThat(tags("He used to read in Paris, by 1998!"))
        .containsExactly("PRP", "VBD", "TO", "NN", "IN", "NNP", ",", "IN", "CD", ".");
    assertThat(tags("The quickly running NASA used it."))
        .containsExactly("DT", "RB", "VBG", "NNP", "VBN", "PRP", ".");
    assertThat(tags("\"Yes\" - no; and: $5 (ok) [x] {y} 'z' #1 … 22nd ***"))
        .containsExactly(
            "\u0060\u0060",
            "NN",
            "''",
            "HYPH",
            "DT",
            ":",
            "CC",
            ":",
            "$",
            "CD",
            "-LRB-",
            "NN",
            "-RRB-",
            "-LRB-",
            "NN",
            "-RRB-",
            "-LRB-",
            "NN",
            "-RRB-",
            "''",
            "NN",
            "''",
            "#",
            "CD",
            ":",
            "CD",
            "NFP");
    assertThat(tags("Was it Dr. Who? Are we? Does he? Has she been? Be here now, not there."))
        .startsWith(
            "VBD", "PRP", "NNP", "NNP", ".", "VBP", "PRP", ".", "VBZ", "PRP", ".", "VBZ", "PRP",
            "VBN", ".", "VB", "RB", "RB", ",", "RB", "RB", ".");
    assertThat(EnglishTokenizer.tagOf("“", "", false)).isEqualTo("\u0060\u0060");
    assertThat(EnglishTokenizer.tagOf("”", "", false)).isEqualTo("''");
    assertThat(EnglishTokenizer.tagOf("’", "", false)).isEqualTo("''");
    assertThat(EnglishTokenizer.tagOf("–", "", false)).isEqualTo(":");
    assertThat(EnglishTokenizer.tagOf("Hello", "", true)).isEqualTo("NN");
    assertThat(EnglishTokenizer.tagOf("Hello", "", false)).isEqualTo("NNP");
    assertThat(EnglishTokenizer.tagOf("used", "to", false)).isEqualTo("VBD");
    assertThat(EnglishTokenizer.tagOf("have", "", false)).isEqualTo("VBP");
    assertThat(EnglishTokenizer.tagOf("had", "", false)).isEqualTo("VBD");
    assertThat(EnglishTokenizer.tagOf("or", "", false)).isEqualTo("CC");
    assertThat(EnglishTokenizer.tagOf("them", "", false)).isEqualTo("PRP");
    assertThat(EnglishTokenizer.tagOf("every", "", false)).isEqualTo("DT");
    assertThat(EnglishTokenizer.tagOf("e.g.", "", false)).isEqualTo("NNP");
  }
}
