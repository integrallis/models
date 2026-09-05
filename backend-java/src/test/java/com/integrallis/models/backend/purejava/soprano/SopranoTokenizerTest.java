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
package com.integrallis.models.backend.purejava.soprano;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class SopranoTokenizerTest {

  @Test
  void normalizesTextAppliesRankedMergesAndWrapsTheSpeechPrompt() {
    SopranoTokenizer tokenizer =
        SopranoTokenizer.fromJson(
            """
            {
              "added_tokens": [
                {"id": 1, "content": "[TEXT]"},
                {"id": 2, "content": "[START]"},
                {"id": 3, "content": "[STOP]"}
              ],
              "model": {
                "vocab": {"[UNK]": 0, "[TEXT]": 1, "[START]": 2, "[STOP]": 3,
                          "h": 4, "i": 5, "hi": 6, " ": 7},
                "merges": [["h", "i"]]
              }
            }
            """);

    assertThat(tokenizer.encodePrompt("  HI  ")).containsExactly(3, 1, 7, 6, 7, 2);
    assertThat(tokenizer.eosToken()).isEqualTo(3);
  }

  @Test
  void isolatesDigitsWordsPunctuationAndUnicodeCodePointsBeforeBpe() {
    SopranoTokenizer tokenizer =
        SopranoTokenizer.fromJson(
            """
            {
              "added_tokens": [
                {"id": 1, "content": "[TEXT]"},
                {"id": 2, "content": "[START]"},
                {"id": 3, "content": "[STOP]"}
              ],
              "model": {
                "vocab": {"[UNK]": 0, "[TEXT]": 1, "[START]": 2, "[STOP]": 3,
                          "a": 4, "1": 5, "2": 6, ",": 7, " ": 8, "é": 9, "a1": 10},
                "merges": [["a", "1"]]
              }
            }
            """);

    assertThat(tokenizer.encodePrompt("A12,  É 😀"))
        .containsExactly(3, 1, 4, 5, 6, 7, 8, 9, 8, 0, 2);
  }

  @Test
  void mapsUnknownCharactersToTheUnknownToken() {
    SopranoTokenizer tokenizer =
        SopranoTokenizer.fromJson(
            """
            {
              "added_tokens": [
                {"id": 1, "content": "[TEXT]"},
                {"id": 2, "content": "[START]"},
                {"id": 3, "content": "[STOP]"}
              ],
              "model": {
                "vocab": {"[UNK]": 0, "[TEXT]": 1, "[START]": 2, "[STOP]": 3},
                "merges": []
              }
            }
            """);

    assertThat(tokenizer.encodePrompt("x")).containsExactly(3, 1, 0, 2);
  }
}
