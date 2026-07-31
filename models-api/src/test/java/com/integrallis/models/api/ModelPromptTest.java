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
import static org.assertj.core.api.Assertions.assertThatNullPointerException;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ModelPromptTest {

  @Test
  void preservesTextWhileCoalescingAdjacentSegmentsOfTheSameKind() {
    ModelPrompt prompt =
        ModelPrompt.builder()
            .control("<start>")
            .control("user\n")
            .text("hello")
            .text("<end>")
            .control("<end>")
            .build();

    assertThat(prompt.text()).isEqualTo("<start>user\nhello<end><end>");
    assertThat(prompt.segments())
        .containsExactly(
            new ModelPrompt.Segment(ModelPrompt.SegmentKind.CONTROL, "<start>user\n"),
            new ModelPrompt.Segment(ModelPrompt.SegmentKind.TEXT, "hello<end>"),
            new ModelPrompt.Segment(ModelPrompt.SegmentKind.CONTROL, "<end>"));
  }

  @Test
  void createsSingleSegmentPromptsAndRejectsNullText() {
    assertThat(ModelPrompt.text("question").segments())
        .containsExactly(new ModelPrompt.Segment(ModelPrompt.SegmentKind.TEXT, "question"));
    assertThat(ModelPrompt.control("<|im_start|>").segments())
        .containsExactly(new ModelPrompt.Segment(ModelPrompt.SegmentKind.CONTROL, "<|im_start|>"));
    assertThatNullPointerException().isThrownBy(() -> ModelPrompt.text(null));
    assertThatNullPointerException().isThrownBy(() -> ModelPrompt.control(null));
  }
}
