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
package com.integrallis.models.runtime;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.Tokenizer;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class PromptVisibilityPlanTest {

  @Test
  void visibilityKeepsRecentTokensAndPinnedSinks() {
    PromptVisibilityPlan plan = PromptVisibilityPlan.of(8, 3, 0, 2);

    assertThat(plan.visiblePositions(7)).containsExactly(0, 2, 5, 6, 7);
    assertThat(plan.visible(7, 1)).isFalse();
    assertThat(plan.visible(7, 2)).isTrue();
    assertThat(plan.visible(7, 5)).isTrue();
    assertThat(plan.visible(3, 4)).isFalse();
  }

  @Test
  void structuredPromptControlSegmentsBecomeSinks() {
    Tokenizer tokenizer =
        new Tokenizer() {
          @Override
          public int[] encode(String text) {
            return text.chars().map(ch -> ch - 'a' + 10).toArray();
          }

          @Override
          public int[] encodeControl(String text) {
            return text.chars().map(ch -> ch).toArray();
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
            return 256;
          }

          @Override
          public int bosToken() {
            return 0;
          }

          @Override
          public int eosToken() {
            return 1;
          }
        };
    ModelPrompt prompt =
        ModelPrompt.builder().control("<s>").text("ab").control("</s>").text("cd").build();

    PromptVisibilityPlan plan = PromptVisibilityPlan.forPrompt(prompt, tokenizer, 2);

    assertThat(plan.tokenCount()).isEqualTo(11);
    assertThat(plan.sinkPositions()).containsExactly(0, 1, 2, 5, 6, 7, 8);
    assertThat(plan.visiblePositions(10)).containsExactly(0, 1, 2, 5, 6, 7, 8, 9, 10);
  }

  @Test
  void rejectsInvalidPositionsAndWindows() {
    assertThatThrownBy(() -> PromptVisibilityPlan.of(4, 0))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("recentWindow");
    assertThatThrownBy(() -> PromptVisibilityPlan.of(4, 2, 4))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("sink position");
    assertThatThrownBy(() -> PromptVisibilityPlan.of(4, 2).visible(4, 0))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("queryPosition");
  }
}
