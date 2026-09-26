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

import static org.assertj.core.api.Assertions.assertThat;

import java.util.List;
import org.junit.jupiter.api.Test;

class PromptBuilderTest {

  @Test
  void thinkingModeSelectsTheQwen3TemplateVariant() {
    PromptBuilder prompts = PromptBuilder.bundled();
    DatasetItem item = new DatasetItem("gsm8k-test-0000", "What is 2+3?", "5", List.of());
    String off = prompts.render(DatasetKind.GSM8K, "number", item, false).text();
    String on = prompts.render(DatasetKind.GSM8K, "number", item, true).text();
    assertThat(off)
        .isEqualTo(
            "<|im_start|>user\nWhat is 2+3?\nPlease reason step by step, and put your final answer"
                + " within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    assertThat(on).endsWith("<|im_start|>assistant\n");
    assertThat(prompts.templateSha256(false)).isNotEqualTo(prompts.templateSha256(true));
    assertThat(prompts.sha256()).hasSize(64);
  }

  @Test
  void arcListsChoicesWithTheirOwnLabels() {
    PromptBuilder prompts = PromptBuilder.bundled();
    DatasetItem item =
        new DatasetItem(
            "arc-1",
            "Which is a gas?",
            "2",
            List.of(new DatasetItem.Choice("1", "ice"), new DatasetItem.Choice("2", "steam")));
    String text = prompts.render(DatasetKind.ARC, "letter", item, false).text();
    assertThat(text).contains("Which is a gas?\n\n1. ice\n2. steam\n\nPlease show your choice");
    assertThat(prompts.loglikContinuation("2")).isEqualTo(" 2");
    assertThat(prompts.loglikAssistantPrefix()).isEqualTo("Answer:");
  }
}
