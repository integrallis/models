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
package com.integrallis.models.backend.purejava.tokenizer;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class BpePreTokenizerTest {

  @Test
  void appliesGpt2WordBoundaries() {
    assertThat(BpePreTokenizer.forName("gpt-2").split("2024 code::foo"))
        .containsExactly("2024", " code", "::", "foo");
  }

  @Test
  void appliesStarCoderSingleDigitBoundaries() {
    assertThat(BpePreTokenizer.forName("starcoder").split("2024 code::foo"))
        .containsExactly("2", "0", "2", "4", " code", "::", "foo");
  }

  @Test
  void appliesSmolLmSingleDigitBoundaries() {
    assertThat(BpePreTokenizer.forName("smollm").split("2024 code::foo"))
        .containsExactly("2", "0", "2", "4", " code", "::", "foo");
  }

  @Test
  void appliesDeepSeekCoderBoundaries() {
    assertThat(BpePreTokenizer.forName("deepseek-coder").split("2024 code::foo\n"))
        .containsExactly("2", "0", "2", "4", " code", "::", "foo", "\n");
  }

  @Test
  void appliesQwen35CombiningMarkBoundaries() {
    assertThat(BpePreTokenizer.forName("qwen35").split(" cafe\u0301 2024"))
        .containsExactly(" cafe\u0301", " ", "2", "0", "2", "4");
  }

  @Test
  void leavesUnknownPreTokenizerTextWhole() {
    assertThat(BpePreTokenizer.forName("vendor-specific").split("2024 code::foo"))
        .containsExactly("2024 code::foo");
  }
}
