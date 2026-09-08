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
package com.integrallis.models.backend.purejava;

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.ToolSpec;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureRegistry;
import com.integrallis.models.backend.purejava.fixture.ModelFixtureRequirement;
import com.integrallis.models.runtime.InferencePipeline;
import com.integrallis.models.runtime.ToolCallTokenConstraints;
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ChatTemplate;
import com.integrallis.models.runtime.chat.VirtualChatModel;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

/** Real-weight facade gate for one chat-to-tool-to-answer virtual conversation. */
@Tag("integration")
class VirtualChatModelIntegrationTest {
  private static final ModelFixtureRequirement QWEN3_0_6B_Q4_0 =
      ModelFixtureRequirement.of("hf://ggml-org/Qwen3-0.6B-GGUF")
          .version("[3.0.0,4.0.0)")
          .variant("q4_0")
          .backend("pure-java")
          .capability("text-generation");
  private static final ModelFixtureRequirement QWEN3_1_7B_Q8_0 =
      ModelFixtureRequirement.of("hf://Qwen/Qwen3-1.7B-GGUF")
          .version("[3.0.0,4.0.0)")
          .variant("q8_0")
          .backend("pure-java")
          .capability("text-generation");
  private static final ToolSpec WEATHER =
      new ToolSpec(
          "get-weather-for-zipcode",
          "Gets weather for a given zipcode",
          """
          {"type":"object","properties":{"zipcode":{"type":"string"}},"required":["zipcode"]}
          """);
  private static final SamplingOptions CHAT_OPTIONS =
      SamplingOptions.builder().temperature(0.0f).maxTokens(8).build();
  private static final SamplingOptions TOOL_OPTIONS =
      SamplingOptions.builder().temperature(0.0f).maxTokens(64).build();

  @Test
  void routesOneCanonicalConversationAcrossQualifiedChatAndToolModels() {
    var fixtures = ModelFixtureRegistry.fromClasspath();
    var chatPath = fixtures.resolve(QWEN3_0_6B_Q4_0).orElseThrow().localPath().orElseThrow();
    var toolPath = fixtures.resolve(QWEN3_1_7B_Q8_0).orElseThrow().localPath().orElseThrow();

    try (PureJavaBackend chatBackend = PureJavaBackend.load(chatPath);
        PureJavaBackend toolBackend = PureJavaBackend.load(toolPath);
        InferencePipeline chatPipeline = new InferencePipeline(chatBackend);
        InferencePipeline toolPipeline = new InferencePipeline(toolBackend)) {
      VirtualChatModel model =
          VirtualChatModel.builder()
              .member(
                  "chat",
                  Set.of("chat"),
                  ChatTemplate.CHATML_NO_THINK,
                  chatPipeline::openGenerationSession)
              .member(
                  "tools",
                  Set.of("tool-use"),
                  ChatTemplate.CHATML_NO_THINK,
                  toolPipeline::openGenerationSession,
                  (session, turn) ->
                      ToolCallTokenConstraints.compile(
                          session.tokenizer(),
                          ChatTemplate.CHATML_NO_THINK.toolSyntax(),
                          turn.tools(),
                          ignored -> List.of("{\"zipcode\":\"88252\"}")))
              .build();

      try (VirtualChatModel.Session conversation =
          model.openSession(
              "real-qwen-pair",
              List.of(
                  ChatMessage.system("Answer directly and use a declared tool when needed.")))) {
        VirtualChatModel.Response greeting =
            conversation.generate(
                "chat", ChatMessage.user("Reply with exactly: READY"), List.of(), CHAT_OPTIONS);
        VirtualChatModel.Response call =
            conversation.generate(
                "tool-use",
                ChatMessage.user("What is the weather for 88252?"),
                List.of(WEATHER),
                TOOL_OPTIONS);
        VirtualChatModel.Response answer =
            conversation.generate(
                "chat",
                ChatMessage.tool(
                    "get-weather-for-zipcode",
                    "{\"zipcode\":\"88252\",\"conditions\":\"Raining cats and dogs\","
                        + "\"temperatureInFahrenheit\":78}"),
                List.of(WEATHER),
                TOOL_OPTIONS);

        assertThat(greeting.memberId()).isEqualTo("chat");
        assertThat(greeting.content()).containsIgnoringCase("READY");
        assertThat(call.memberId()).isEqualTo("tools");
        assertThat(call.boundary()).isEqualTo(VirtualChatModel.Boundary.FIRST_SWITCH);
        assertThat(call.toolCalls())
            .singleElement()
            .satisfies(
                toolCall -> {
                  assertThat(toolCall.name()).isEqualTo("get-weather-for-zipcode");
                  assertThat(toolCall.argumentsJson()).contains("88252");
                });
        assertThat(answer.memberId()).isEqualTo("chat");
        assertThat(answer.boundary()).isEqualTo(VirtualChatModel.Boundary.SWITCH_BACK);
        assertThat(answer.content()).contains("78").containsIgnoringCase("rain");
        assertThat(answer.content()).doesNotContain("<tool_call>", "<think>");
        assertThat(answer.metrics().promptCache().cacheReadInputTokens()).isPositive();
      }
    }
  }
}
