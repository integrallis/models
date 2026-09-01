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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class UnigramTokenizerTest {

  @Test
  void viterbiSelectsTheHighestScoringCompleteSegmentation() {
    UnigramTokenizer tokenizer =
        tokenizer(
            new String[] {"<unk>", "▁", "a", "b", "▁a", "ab", "▁ab"},
            new float[] {0, -3, -2, -2, 1, 2, 3},
            List.of(2, 1, 1, 1, 1, 1, 1),
            true,
            true);

    assertThat(tokenizer.encode("ab")).containsExactly(6);
    assertThat(tokenizer.encode("  a   b  ")).containsExactly(4, 1, 3);
    assertThat(tokenizer.encode("")).isEmpty();
  }

  @Test
  void preservesUserDefinedPiecesAndCoalescesUnknownRuns() {
    UnigramTokenizer tokenizer =
        tokenizer(
            new String[] {"<unk>", "a", "<transit>"},
            new float[] {0, -1, -100},
            List.of(2, 1, 4),
            false,
            false);

    assertThat(tokenizer.encode("<transit>a")).containsExactly(2, 1);
    assertThat(tokenizer.encode("東京xyz")).containsExactly(0);
  }

  @Test
  void rejectsMalformedPrecompiledCharacterMapsAtConstruction() {
    assertThatThrownBy(
            () ->
                new UnigramTokenizer(
                    new String[] {"<unk>"},
                    new float[] {0},
                    List.of(2),
                    new byte[] {1, 2, 3},
                    false,
                    false,
                    0))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("smaller than its header");

    assertThatThrownBy(
            () ->
                new UnigramTokenizer(
                    new String[] {"<unk>"},
                    new float[] {0},
                    List.of(2),
                    new byte[] {4, 0, 0, 0},
                    false,
                    false,
                    0))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("invalid precompiled charsmap XCDA length");
  }

  private static UnigramTokenizer tokenizer(
      String[] vocabulary,
      float[] scores,
      List<Integer> tokenTypes,
      boolean addSpacePrefix,
      boolean removeExtraWhitespaces) {
    return new UnigramTokenizer(
        vocabulary, scores, tokenTypes, new byte[0], addSpacePrefix, removeExtraWhitespaces, 0);
  }
}
