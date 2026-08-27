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

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ChatTemplateToolSyntaxTest {

  @Nested
  static class Mapping {

    @Test
    void everyTemplateDeclaresASyntax() {
      for (ChatTemplate template : ChatTemplate.values()) {
        assertThat(template.toolSyntax()).as("template %s", template).isNotNull();
      }
    }

    @Test
    void chatMlTemplatesUseTheQwenFamilyFormat() {
      assertThat(ChatTemplate.CHATML.toolSyntax()).isEqualTo(ToolSyntax.QWEN);
      assertThat(ChatTemplate.CHATML_NO_THINK.toolSyntax()).isEqualTo(ToolSyntax.QWEN);
      assertThat(ChatTemplate.CHATML_DIRECT.toolSyntax()).isEqualTo(ToolSyntax.QWEN);
      assertThat(ChatTemplate.CHATML_ANSWER.toolSyntax()).isEqualTo(ToolSyntax.QWEN);
    }

    @Test
    void llama3UsesItsOwnFormat() {
      assertThat(ChatTemplate.LLAMA3.toolSyntax()).isEqualTo(ToolSyntax.LLAMA3);
    }

    @Test
    void needle2UsesItsArrayWrappedToolFormat() {
      assertThat(ChatTemplate.NEEDLE2.toolSyntax()).isEqualTo(ToolSyntax.NEEDLE2);
      assertThat(ChatTemplate.NEEDLE2.canParseToolCalls()).isTrue();
      assertThat(ChatTemplate.NEEDLE2.toolSyntax().arrayWrapped()).isTrue();
    }

    @Test
    void familiesWithNoToolFormatDeclareNone() {
      // Verified against each family's published chat template: Gemma 2 and Phi-3.5 have no
      // `tools` variable at all, and the DeepSeek templates expose none either.
      assertThat(ChatTemplate.RAW.supportsTools()).isFalse();
      assertThat(ChatTemplate.GEMMA.supportsTools()).isFalse();
      assertThat(ChatTemplate.PHI3.supportsTools()).isFalse();
      assertThat(ChatTemplate.DEEPSEEK.supportsTools()).isFalse();
      assertThat(ChatTemplate.ZEPHYR.supportsTools()).isFalse();
      assertThat(ChatTemplate.H2O.supportsTools()).isFalse();
      assertThat(ChatTemplate.H2O_DIRECT.supportsTools()).isFalse();
    }

    @Test
    void taggedFamiliesHaveAFormatButCannotYetBeParsed() {
      // Gemma 4 emits <|tool_call>call:name{k:v}<tool_call|> and MiniCPM5 emits
      // <function name="f"><param name="k">v</param></function>. Both are real formats, but
      // tagged arguments carry no types, so producing JSON needs the declared schemas.
      for (ChatTemplate template :
          new ChatTemplate[] {ChatTemplate.GEMMA4, ChatTemplate.MINICPM5_NO_THINK}) {
        assertThat(template.supportsTools()).as("template %s", template).isTrue();
        assertThat(template.canParseToolCalls()).as("template %s", template).isFalse();
        assertThat(template.toolSyntax().mode())
            .as("template %s", template)
            .isEqualTo(ToolSyntax.Mode.TAG_WITH_TAGGED);
      }
    }

    @Test
    void gemma4RecordsItsAsymmetricDelimiters() {
      // The pipe moves from the leading to the trailing position; a symmetric assumption breaks.
      assertThat(ToolSyntax.GEMMA4.sectionStart()).isEqualTo("<|tool_call>");
      assertThat(ToolSyntax.GEMMA4.sectionEnd()).isEqualTo("<tool_call|>");
    }

    @Test
    void parsableFamiliesAreExactlyTheJsonShapedOnes() {
      for (ChatTemplate template : ChatTemplate.values()) {
        boolean jsonShaped =
            template.toolSyntax().mode() == ToolSyntax.Mode.JSON_NATIVE
                || template.toolSyntax().mode() == ToolSyntax.Mode.TAG_WITH_JSON;
        assertThat(template.canParseToolCalls()).as("template %s", template).isEqualTo(jsonShaped);
      }
    }
  }
}
