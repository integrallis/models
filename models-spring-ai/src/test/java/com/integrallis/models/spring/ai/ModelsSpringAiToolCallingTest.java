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
package com.integrallis.models.spring.ai;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.ModelPrompt;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.TextGenerationModel;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.runtime.chat.ChatTemplate;
import java.util.List;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.model.tool.ToolCallingChatOptions;
import org.springframework.ai.tool.ToolCallback;
import org.springframework.ai.tool.definition.DefaultToolDefinition;
import org.springframework.ai.tool.definition.ToolDefinition;

@Tag("unit")
class ModelsSpringAiToolCallingTest {

  private static final String SCHEMA =
      "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}";

  /** Records the rendered prompt and replays a canned completion. */
  private static final class ScriptedModel implements TextGenerationModel {
    private final String completion;
    private final AtomicReference<String> lastPrompt = new AtomicReference<>();

    ScriptedModel(String completion) {
      this.completion = completion;
    }

    @Override
    public String modelName() {
      return "ScriptedModel";
    }

    @Override
    public BackendDiagnostics diagnostics() {
      return BackendDiagnostics.unavailable("scripted");
    }

    @Override
    public String generate(String prompt, SamplingOptions options) {
      lastPrompt.set(prompt);
      return completion;
    }

    @Override
    public String generate(ModelPrompt prompt, SamplingOptions options) {
      lastPrompt.set(prompt.text());
      return completion;
    }

    @Override
    public void generate(String prompt, SamplingOptions options, TokenStream stream) {
      lastPrompt.set(prompt);
      stream.onToken(completion);
      stream.onComplete();
    }

    @Override
    public void generate(ModelPrompt prompt, SamplingOptions options, TokenStream stream) {
      lastPrompt.set(prompt.text());
      stream.onToken(completion);
      stream.onComplete();
    }

    String lastPrompt() {
      return lastPrompt.get();
    }
  }

  private static ToolCallback weatherCallback() {
    ToolDefinition definition =
        DefaultToolDefinition.builder()
            .name("get_weather")
            .description("Look up the forecast")
            .inputSchema(SCHEMA)
            .build();
    return new ToolCallback() {
      @Override
      public ToolDefinition getToolDefinition() {
        return definition;
      }

      @Override
      public String call(String toolInput) {
        return "{\"tempF\":88}";
      }
    };
  }

  private static Prompt promptWithTools(
      List<org.springframework.ai.chat.messages.Message> messages) {
    return new Prompt(
        messages,
        ToolCallingChatOptions.builder().toolCallbacks(List.of(weatherCallback())).build());
  }

  @Nested
  static class Declaration {

    @Test
    void rendersDeclaredToolsIntoThePrompt() {
      ScriptedModel model = new ScriptedModel("no call here");
      ModelsSpringAiChatModel chat =
          new ModelsSpringAiChatModel(
              model, ChatTemplate.CHATML, SamplingOptions.builder().build());

      chat.call(promptWithTools(List.of(new UserMessage("weather in Austin?"))));

      assertThat(model.lastPrompt()).contains("# Tools");
      assertThat(model.lastPrompt()).contains("get_weather");
      assertThat(model.lastPrompt()).contains(SCHEMA);
    }

    @Test
    void withoutToolsRendersThePlainEnvelope() {
      ScriptedModel model = new ScriptedModel("hi");
      ModelsSpringAiChatModel chat =
          new ModelsSpringAiChatModel(
              model, ChatTemplate.CHATML, SamplingOptions.builder().build());

      chat.call(new Prompt(List.of(new UserMessage("hello"))));

      assertThat(model.lastPrompt()).doesNotContain("# Tools");
    }

    @Test
    void refusesToolsWhenTheTemplateCannotExpressThem() {
      ScriptedModel model = new ScriptedModel("x");
      ModelsSpringAiChatModel chat =
          new ModelsSpringAiChatModel(model, ChatTemplate.GEMMA, SamplingOptions.builder().build());

      assertThatThrownBy(() -> chat.call(promptWithTools(List.of(new UserMessage("q")))))
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("gemma");
    }
  }

  @Nested
  static class Recovery {

    @Test
    void surfacesAToolCallOnTheAssistantMessage() {
      // Spring AI 2.0 keys tool execution on hasToolCalls(), not on a finish reason.
      ScriptedModel model =
          new ScriptedModel(
              "<tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Austin\"}}\n</tool_call>");
      ModelsSpringAiChatModel chat =
          new ModelsSpringAiChatModel(
              model, ChatTemplate.CHATML, SamplingOptions.builder().build());

      ChatResponse response = chat.call(promptWithTools(List.of(new UserMessage("weather?"))));

      assertThat(response.hasToolCalls()).isTrue();
      AssistantMessage assistant = response.getResult().getOutput();
      assertThat(assistant.getToolCalls())
          .singleElement()
          .satisfies(
              call -> {
                assertThat(call.name()).isEqualTo("get_weather");
                assertThat(call.arguments()).isEqualTo("{\"city\": \"Austin\"}");
                assertThat(call.type()).isEqualTo("function");
                assertThat(call.id()).isEqualTo("000000000");
              });
    }

    @Test
    void keepsProseAlongsideTheCall() {
      ScriptedModel model =
          new ScriptedModel(
              "Let me check.\n<tool_call>{\"name\":\"get_weather\",\"arguments\":{}}</tool_call>");
      ModelsSpringAiChatModel chat =
          new ModelsSpringAiChatModel(
              model, ChatTemplate.CHATML, SamplingOptions.builder().build());

      ChatResponse response = chat.call(promptWithTools(List.of(new UserMessage("weather?"))));

      assertThat(response.getResult().getOutput().getText()).isEqualTo("Let me check.");
      assertThat(response.hasToolCalls()).isTrue();
    }

    @Test
    void plainAnswersCarryNoToolCalls() {
      ScriptedModel model = new ScriptedModel("It is 88 degrees.");
      ModelsSpringAiChatModel chat =
          new ModelsSpringAiChatModel(
              model, ChatTemplate.CHATML, SamplingOptions.builder().build());

      ChatResponse response = chat.call(promptWithTools(List.of(new UserMessage("weather?"))));

      assertThat(response.hasToolCalls()).isFalse();
      assertThat(response.getResult().getOutput().getText()).isEqualTo("It is 88 degrees.");
    }

    @Test
    void doesNotScanForCallsWhenNoToolsWereDeclared() {
      // Text that merely looks like a call must stay text when the caller offered no tools.
      ScriptedModel model =
          new ScriptedModel("<tool_call>{\"name\":\"x\",\"arguments\":{}}</tool_call>");
      ModelsSpringAiChatModel chat =
          new ModelsSpringAiChatModel(
              model, ChatTemplate.CHATML, SamplingOptions.builder().build());

      ChatResponse response = chat.call(new Prompt(List.of(new UserMessage("hi"))));

      assertThat(response.hasToolCalls()).isFalse();
    }
  }

  @Nested
  static class ResultsComingBack {

    @Test
    void mapsToolResponsesIntoTheConversation() {
      ScriptedModel model = new ScriptedModel("It is 88 degrees.");
      ModelsSpringAiChatModel chat =
          new ModelsSpringAiChatModel(
              model, ChatTemplate.CHATML, SamplingOptions.builder().build());

      chat.call(
          promptWithTools(
              List.of(
                  new UserMessage("weather?"),
                  ToolResponseMessage.builder()
                      .responses(
                          List.of(
                              new ToolResponseMessage.ToolResponse(
                                  "000000000", "get_weather", "{\"tempF\":88}")))
                      .build())));

      assertThat(model.lastPrompt()).contains("<tool_response>");
      assertThat(model.lastPrompt()).contains("{\"tempF\":88}");
    }
  }
}
