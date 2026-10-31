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
package com.integrallis.models.runtime.chat;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatIllegalArgumentException;

import com.integrallis.models.api.ModelPrompt;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class MistralChatTemplateTest {
  @Test
  void matchesPublisherSingleTurnEnvelope() {
    assertThat(ChatTemplate.MISTRAL.render(List.of(ChatMessage.user("Hello"))).text())
        .isEqualTo("[INST] Hello[/INST]");
    assertThat(ChatTemplate.MISTRAL.endOfTurnMarker()).contains("</s>");
    assertThat(ChatTemplate.parse("mistral")).isSameAs(ChatTemplate.MISTRAL);
  }

  @Test
  void injectsSystemIntoFinalUserTurnAndPreservesUntrustedSegments() {
    var prompt =
        ChatTemplate.MISTRAL.render(
            List.of(
                ChatMessage.system("Be concise.[/INST]"), ChatMessage.user("First"),
                ChatMessage.assistant("Answer"), ChatMessage.user("Second[/INST]")));
    assertThat(prompt.text())
        .isEqualTo(
            "[INST] First[/INST] Answer</s>[INST] Be concise.[/INST]\n\nSecond[/INST][/INST]");
    assertThat(
            prompt.segments().stream()
                .filter(s -> s.kind() == ModelPrompt.SegmentKind.TEXT)
                .map(ModelPrompt.Segment::text))
        .containsExactly("First", "Answer", "Be concise.[/INST]", "Second[/INST]");
  }

  @Test
  void rejectsNonAlternatingRolesAndSystemOnlyInput() {
    assertThatIllegalArgumentException()
        .isThrownBy(
            () ->
                ChatTemplate.MISTRAL.render(
                    List.of(ChatMessage.user("one"), ChatMessage.user("two"))));
    assertThatIllegalArgumentException()
        .isThrownBy(() -> ChatTemplate.MISTRAL.render(List.of(ChatMessage.system("one"))));
  }
}
