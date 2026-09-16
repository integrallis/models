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
package com.integrallis.models.api;

import static org.assertj.core.api.Assertions.assertThat;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class TokenizerTest {

  @Test
  void singleTerminalTokenizerReportsOnlyItsEos() {
    assertThat(tokenizer(false).endOfGenerationTokenIds()).containsExactly(3);
  }

  @Test
  void endOfGenerationSetFollowsAnOverriddenPredicateInAscendingOrder() {
    assertThat(tokenizer(true).endOfGenerationTokenIds()).containsExactly(1, 3, 6);
  }

  private static Tokenizer tokenizer(boolean multipleTerminals) {
    return new Tokenizer() {
      @Override
      public int[] encode(String text) {
        return new int[0];
      }

      @Override
      public String decode(int[] tokens) {
        return "";
      }

      @Override
      public String decode(int token) {
        return "";
      }

      @Override
      public int vocabSize() {
        return 8;
      }

      @Override
      public int bosToken() {
        return 0;
      }

      @Override
      public int eosToken() {
        return 3;
      }

      @Override
      public boolean isEndOfGeneration(int token) {
        return multipleTerminals ? token == 1 || token == 3 || token == 6 : token == eosToken();
      }
    };
  }
}
