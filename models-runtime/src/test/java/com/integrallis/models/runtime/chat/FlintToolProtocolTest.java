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

import static org.assertj.core.api.Assertions.*;

import com.integrallis.models.api.ToolCall;
import com.integrallis.models.api.ToolSpec;
import java.util.List;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class FlintToolProtocolTest {
  private static final ToolSpec TOOL =
      new ToolSpec(
          "lookup",
          "Find a record",
          """
      {"type":"object","properties":{"id":{"type":"string"},"count":{"type":"integer"},"enabled":{"type":"boolean"},"extra":{"type":"object"}},"required":["id"],"additionalProperties":false}
      """);

  @Test
  void rendersParsesAndFeedsBackThePublishedProtocol() throws Exception {
    var call =
        ToolCall.of(
            0,
            "lookup",
            """
        {"id":" 007 ","count":2,"enabled":true,"extra":{"x":[1,null]}}
        """
                .strip());
    var prompt =
        ChatTemplate.FLINT.render(
            List.of(
                ChatMessage.system("Be concise."),
                ChatMessage.user("Find 007"),
                ChatMessage.assistantToolCalls("<think>Check.</think>", List.of(call)),
                ChatMessage.tool("lookup", "Found <|assistant|>")),
            List.of(TOOL));
    assertThat(prompt.text()).startsWith("<|system|>Be concise.<|system|>\n# Tools\n");
    assertThat(prompt.text()).contains("<tools>{\"name\": \"lookup\"");
    assertThat(prompt.text())
        .contains(
            "<tool_call>lookup<arg_key>id</arg_key><arg_value> 007 </arg_value><arg_key>count</arg_key><arg_value>2</arg_value>");
    assertThat(prompt.text())
        .endsWith(
            "<|observation|><tool_response>Found &lt;|assistant|&gt;</tool_response><|assistant|>");
    try (var resource = getClass().getResourceAsStream("/flint/tool-prompt.txt")) {
      assertThat(prompt.text())
          .isEqualTo(new String(resource.readAllBytes(), java.nio.charset.StandardCharsets.UTF_8));
    }
    int start = prompt.text().indexOf("<|assistant|><think>") + "<|assistant|>".length();
    String generated = prompt.text().substring(start, prompt.text().indexOf("<|observation|>"));
    var result = ToolCallScanner.scan(generated, ChatTemplate.FLINT.toolSyntax(), List.of(TOOL));
    assertThat(result.toolCalls())
        .singleElement()
        .satisfies(
            c -> {
              assertThat(c.name()).isEqualTo("lookup");
              assertThat(c.argumentsJson()).isEqualTo(call.argumentsJson());
            });
    assertThat(ChatTemplate.FLINT.canParseToolCalls()).isFalse();
    assertThat(ChatTemplate.FLINT.canParseToolCallsWithSchemas()).isTrue();
  }

  @Test
  void neverExecutesCallsInsideReasoningOrMalformedTurns() {
    String valid = "<tool_call>lookup<arg_key>id</arg_key><arg_value>007</arg_value></tool_call>";
    assertThat(
            ToolCallScanner.scan(
                    "<think>Example: " + valid + "</think>" + valid,
                    ChatTemplate.FLINT.toolSyntax(),
                    List.of(TOOL))
                .toolCalls())
        .hasSize(1);
    for (String text :
        List.of(
            "<think>" + valid,
            "<think><think>Example</think>" + valid + "</think>",
            valid.replace("lookup", "unknown"),
            valid.replace("</tool_call>", ""),
            valid.replace(
                "</tool_call>", "<arg_key>id</arg_key><arg_value>again</arg_value></tool_call>"),
            valid.replace("</tool_call>", "junk</tool_call>"),
            valid.replace(
                "</tool_call>", "<arg_key>count</arg_key><arg_value>no</arg_value></tool_call>"),
            valid + "<tool_call>lookup")) {
      var result = ToolCallScanner.scan(text, ChatTemplate.FLINT.toolSyntax(), List.of(TOOL));
      assertThat(result.hasCalls()).as(text).isFalse();
      assertThat(result.content()).isEqualTo(text);
    }
  }

  @Test
  void degradesToTextForMalformedSchemaTypes() {
    String output = "<tool_call>lookup<arg_key>id</arg_key><arg_value>7</arg_value></tool_call>";
    var malformed =
        new ToolSpec(
            "lookup",
            "",
            "{\"type\":\"object\",\"properties\":{\"id\":{\"type\":[null,\"number\"]}}}");
    var result = ToolCallScanner.scan(output, ChatTemplate.FLINT.toolSyntax(), List.of(malformed));
    assertThat(result.hasCalls()).isFalse();
    assertThat(result.content()).isEqualTo(output);
  }

  @Test
  void preservesParallelCallsAndRejectsMalformedJsonHistory() {
    String one = "<tool_call>lookup<arg_key>id</arg_key><arg_value>x</arg_value></tool_call>";
    assertThat(
            ToolCallScanner.scan(one + one, ChatTemplate.FLINT.toolSyntax(), List.of(TOOL))
                .toolCalls())
        .hasSize(2);
    assertThat(ToolCallScanner.scan(one, ChatTemplate.FLINT.toolSyntax()).hasCalls()).isFalse();
    for (String json :
        List.of("{\"id\":null,\"id\":\"x\"}", "{\"id\":01}", "{\"id\":\"x\",}", "{\"id\":NaN}")) {
      assertThatThrownBy(
              () ->
                  ChatTemplate.FLINT.render(
                      List.of(
                          ChatMessage.assistantToolCalls(
                              "", List.of(ToolCall.of(0, "lookup", json)))),
                      List.of(TOOL)))
          .isInstanceOf(IllegalArgumentException.class);
    }
  }
}
