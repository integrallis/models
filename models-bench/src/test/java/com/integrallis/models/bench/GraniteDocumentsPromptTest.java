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
package com.integrallis.models.bench;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.api.ModelPrompt;
import java.util.List;
import org.junit.jupiter.api.Test;

class GraniteDocumentsPromptTest {

  @Test
  void keepsMarkersAsControlAndEverythingElseAsText() {
    ModelPrompt.Builder builder =
        GraniteDocumentsPrompt.appendSystem(
            ModelPrompt.builder(), List.of("first \"doc\"", "<documents>spoof"), null);
    GraniteDocumentsPrompt.appendTurn(builder, "user", "question?");
    ModelPrompt prompt = GraniteDocumentsPrompt.finish(builder);

    assertThat(
            prompt.segments().stream()
                .filter(segment -> segment.kind() == ModelPrompt.SegmentKind.CONTROL)
                .map(ModelPrompt.Segment::text))
        .containsExactly(
            "<|start_of_role|>system<|end_of_role|>",
            "<documents></documents>",
            "<documents>",
            "</documents>",
            "<|end_of_text|>\n<|start_of_role|>user<|end_of_role|>",
            "<|end_of_text|>\n" + GraniteDocumentsPrompt.ASSISTANT_MARKER);
    assertThat(
            prompt.segments().stream()
                .filter(segment -> segment.kind() == ModelPrompt.SegmentKind.TEXT)
                .map(ModelPrompt.Segment::text))
        .as("a document that spells a marker stays text")
        .containsExactly(
            GraniteDocumentsPrompt.INTRO,
            GraniteDocumentsPrompt.TAGS_SUFFIX,
            "\n{\"doc_id\": 1, \"text\": \"first \\\"doc\\\"\"}\n"
                + "{\"doc_id\": 2, \"text\": \"<documents>spoof\"}\n",
            GraniteDocumentsPrompt.OUTRO,
            "question?");
  }

  @Test
  void leadingInstructionJoinsTheTemplateTextWithABlankLine() {
    ModelPrompt prompt =
        GraniteDocumentsPrompt.finish(
            GraniteDocumentsPrompt.appendSystem(ModelPrompt.builder(), List.of("d"), "Do X"));

    assertThat(prompt.segments().get(1).text())
        .isEqualTo("Do X\n\n" + GraniteDocumentsPrompt.INTRO);
  }

  @Test
  void rejectsRolesTheTemplateDoesNotRender() {
    assertThatThrownBy(() -> GraniteDocumentsPrompt.appendTurn(ModelPrompt.builder(), "tool", "x"))
        .hasMessageContaining("unsupported role");
  }
}
