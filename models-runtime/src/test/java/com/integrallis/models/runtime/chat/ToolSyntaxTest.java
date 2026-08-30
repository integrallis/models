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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ToolSyntaxTest {

  @Nested
  static class Capability {

    @Test
    void noneReportsNoToolSupport() {
      assertThat(ToolSyntax.NONE.supportsTools()).isFalse();
      assertThat(ToolSyntax.NONE.mode()).isEqualTo(ToolSyntax.Mode.NONE);
    }

    @Test
    void everyPopulatedSyntaxReportsToolSupport() {
      assertThat(ToolSyntax.QWEN.supportsTools()).isTrue();
      assertThat(ToolSyntax.HERMES.supportsTools()).isTrue();
      assertThat(ToolSyntax.LLAMA3.supportsTools()).isTrue();
      assertThat(ToolSyntax.NEEDLE2.supportsTools()).isTrue();
      assertThat(ToolSyntax.HARMONY.supportsTools()).isTrue();
      assertThat(ToolSyntax.GEMMA4.supportsTools()).isTrue();
      assertThat(ToolSyntax.MINICPM5.supportsTools()).isTrue();
    }

    @Test
    void onlyJsonShapedFormatsAreParsableFromTextAlone() {
      assertThat(ToolSyntax.QWEN.parsable()).isTrue();
      assertThat(ToolSyntax.HERMES.parsable()).isTrue();
      assertThat(ToolSyntax.LLAMA3.parsable()).isTrue();
      assertThat(ToolSyntax.NEEDLE2.parsable()).isTrue();
      assertThat(ToolSyntax.HARMONY.parsable()).isTrue();
      // Tagged arguments carry no type information, so JSON cannot be reconstructed without
      // the declared tool schemas.
      assertThat(ToolSyntax.GEMMA4.parsable()).isFalse();
      assertThat(ToolSyntax.MINICPM5.parsable()).isFalse();
      assertThat(ToolSyntax.NONE.parsable()).isFalse();
    }
  }

  @Nested
  static class Families {

    @Test
    void qwenTagsCallsAndFeedsResultsBackThroughAUserTurn() {
      ToolSyntax syntax = ToolSyntax.QWEN;

      assertThat(syntax.mode()).isEqualTo(ToolSyntax.Mode.TAG_WITH_JSON);
      assertThat(syntax.sectionStart()).isEqualTo("<tool_call>");
      assertThat(syntax.sectionEnd()).isEqualTo("</tool_call>");
      assertThat(syntax.argsField()).isEqualTo("arguments");
      assertThat(syntax.parallelCalls()).isTrue();
      assertThat(syntax.resultStyle()).isEqualTo(ToolSyntax.ResultStyle.USER_WRAPPED);
      assertThat(syntax.resultStart()).isEqualTo("<tool_response>");
      assertThat(syntax.resultEnd()).isEqualTo("</tool_response>");
    }

    @Test
    void hermesSharesQwenCallSyntaxButUsesARealToolRole() {
      assertThat(ToolSyntax.HERMES.sectionStart()).isEqualTo(ToolSyntax.QWEN.sectionStart());
      assertThat(ToolSyntax.HERMES.sectionEnd()).isEqualTo(ToolSyntax.QWEN.sectionEnd());
      assertThat(ToolSyntax.HERMES.argsField()).isEqualTo(ToolSyntax.QWEN.argsField());
      assertThat(ToolSyntax.HERMES.resultStyle()).isEqualTo(ToolSyntax.ResultStyle.TOOL_ROLE);
    }

    @Test
    void llama3UsesParametersAndForbidsParallelCalls() {
      // Llama 3.x is the only major family keyed on "parameters", and its template raises
      // outright when asked to render more than one call.
      ToolSyntax syntax = ToolSyntax.LLAMA3;

      assertThat(syntax.mode()).isEqualTo(ToolSyntax.Mode.JSON_NATIVE);
      assertThat(syntax.argsField()).isEqualTo("parameters");
      assertThat(syntax.parallelCalls()).isFalse();
      assertThat(syntax.resultStyle()).isEqualTo(ToolSyntax.ResultStyle.IPYTHON);
    }

    @Test
    void needle2WrapsParallelCallsInOneArrayAndUsesPlainUserResults() {
      ToolSyntax syntax = ToolSyntax.NEEDLE2;

      assertThat(syntax.mode()).isEqualTo(ToolSyntax.Mode.TAG_WITH_JSON);
      assertThat(syntax.arrayWrapped()).isTrue();
      assertThat(syntax.parallelCalls()).isTrue();
      assertThat(syntax.resultStyle()).isEqualTo(ToolSyntax.ResultStyle.USER_PLAIN);
    }
  }

  @Nested
  static class Validation {

    @Test
    void taggedModesRequireDelimiters() {
      assertThatThrownBy(
              () ->
                  new ToolSyntax(
                      ToolSyntax.Mode.TAG_WITH_JSON,
                      "",
                      "</tool_call>",
                      "name",
                      "arguments",
                      false,
                      true,
                      ToolSyntax.ResultStyle.TOOL_ROLE,
                      "",
                      ""))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("sectionStart");
    }

    @Test
    void userWrappedResultsRequireResponseDelimiters() {
      assertThatThrownBy(
              () ->
                  new ToolSyntax(
                      ToolSyntax.Mode.TAG_WITH_JSON,
                      "<tool_call>",
                      "</tool_call>",
                      "name",
                      "arguments",
                      false,
                      true,
                      ToolSyntax.ResultStyle.USER_WRAPPED,
                      "",
                      ""))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("resultStart");
    }

    @Test
    void rejectsBlankFieldNames() {
      assertThatThrownBy(
              () ->
                  new ToolSyntax(
                      ToolSyntax.Mode.JSON_NATIVE,
                      "",
                      "",
                      "name",
                      " ",
                      false,
                      true,
                      ToolSyntax.ResultStyle.TOOL_ROLE,
                      "",
                      ""))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("argsField");
    }
  }
}
