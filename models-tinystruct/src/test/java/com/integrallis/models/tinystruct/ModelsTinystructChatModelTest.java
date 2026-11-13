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
package com.integrallis.models.tinystruct;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.api.BackendDiagnostics;
import com.integrallis.models.api.GenerationUsage;
import com.integrallis.models.api.ModelGenerationException;
import com.integrallis.models.api.SamplingOptions;
import com.integrallis.models.api.StopReason;
import com.integrallis.models.api.TextGenerationModel;
import com.integrallis.models.api.TokenStream;
import com.integrallis.models.api.ToolSpec;
import com.integrallis.models.router.ModelCandidate;
import com.integrallis.models.router.ModelFleet;
import com.integrallis.models.router.RoutingUsage;
import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ChatTemplate;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.tinystruct.data.component.Builder;
import org.tinystruct.data.component.Builders;

@Tag("unit")
class ModelsTinystructChatModelTest {

  @Test
  void generatesAndStreamsPlainChatAndToolCallsIntoBuilderStructures() {
    AtomicBoolean closed = new AtomicBoolean();
    TextGenerationModel stubModel =
        new TextGenerationModel() {
          @Override
          public String modelName() {
            return "qwen3-stub";
          }

          @Override
          public BackendDiagnostics diagnostics() {
            return BackendDiagnostics.unavailable("qwen3-stub");
          }

          @Override
          public void generate(String prompt, SamplingOptions options, TokenStream stream) {
            if (prompt.contains("weather")) {
              String toolCall =
                  "<tool_call>{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Austin\"}}</tool_call>";
              stream.onToken(toolCall);
              stream.onComplete(new GenerationUsage(14, 9), StopReason.EOS);
              return;
            }
            stream.onToken("Hello ");
            stream.onToken("world");
            stream.onComplete(new GenerationUsage(10, 2), StopReason.EOS);
          }

          @Override
          public void close() {
            closed.set(true);
          }
        };

    try (ModelsTinystructChatModel chatModel =
        new ModelsTinystructChatModel(
            stubModel, ChatTemplate.CHATML, SamplingOptions.builder().maxTokens(64).build())) {
      assertThat(chatModel.supportsTools()).isTrue();
      assertThat(chatModel.diagnostics()).isEqualTo(BackendDiagnostics.unavailable("qwen3-stub"));

      Builder plain = chatModel.chat("Say hello");
      assertThat(plain.get("text")).isEqualTo("Hello world");
      assertThat(plain.get("model")).isEqualTo("qwen3-stub");
      assertThat(plain.get("finish_reason")).isEqualTo("EOS");
      Builder usage = (Builder) plain.get("usage");
      assertThat(usage.get("prompt_tokens")).isEqualTo(10);
      assertThat(usage.get("completion_tokens")).isEqualTo(2);
      assertThat(usage.get("total_tokens")).isEqualTo(12);

      List<String> tokens = new ArrayList<>();
      ToolSpec weatherTool =
          new ToolSpec(
              "get_weather",
              "Get city weather",
              "{\"type\":\"object\",\"properties\":{\"city\":{\"type\":\"string\"}}}");
      Builder toolTurn =
          chatModel.stream(
              List.of(ChatMessage.user("Check weather in Austin")),
              null,
              List.of(weatherTool),
              tokens::add);

      assertThat(tokens).hasSize(1);
      assertThat(toolTurn.get("finish_reason")).isEqualTo("TOOL_CALLS");
      Builders toolCalls = (Builders) toolTurn.get("tool_calls");
      assertThat(toolCalls).hasSize(1);
      assertThat(toolCalls.get(0).get("name")).isEqualTo("get_weather");
      assertThat(toolCalls.get(0).get("arguments").toString()).contains("Austin");
    }
    assertThat(closed).isTrue();
  }

  @Test
  void propagatesCheckedAndRuntimeStreamErrorsAndRejectsEmptyMessages() {
    TextGenerationModel checkedErrorModel =
        new TextGenerationModel() {
          @Override
          public String modelName() {
            return "error-stub";
          }

          @Override
          public BackendDiagnostics diagnostics() {
            return BackendDiagnostics.unavailable("error-stub");
          }

          @Override
          public void generate(String prompt, SamplingOptions options, TokenStream stream) {
            stream.onError(new IOException("disk read error"));
          }
        };

    ModelsTinystructChatModel model = new ModelsTinystructChatModel(checkedErrorModel);
    assertThatThrownBy(() -> model.chat(List.of()))
        .isInstanceOf(IllegalArgumentException.class)
        .hasMessageContaining("messages must not be empty");
    assertThatThrownBy(() -> model.chat("trigger error"))
        .isInstanceOf(ModelGenerationException.class)
        .hasCauseInstanceOf(IOException.class);
  }

  @Test
  void routesChatAndStreamingAcrossModelFleetWithContinuityAndUsageSettlement() {
    TinystructChatModel failingPrimary =
        new TinystructChatModel() {
          @Override
          public Builder chat(
              List<ChatMessage> messages, SamplingOptions options, List<ToolSpec> tools) {
            throw new IllegalStateException("primary down");
          }

          @Override
          public Builder stream(
              List<ChatMessage> messages,
              SamplingOptions options,
              List<ToolSpec> tools,
              java.util.function.Consumer<String> onToken) {
            throw new IllegalStateException("primary stream down");
          }
        };

    TinystructChatModel backup =
        new TinystructChatModel() {
          @Override
          public Builder chat(
              List<ChatMessage> messages, SamplingOptions options, List<ToolSpec> tools) {
            Builder result = new Builder();
            result.put("text", "backup answer");
            Builder usage = new Builder();
            usage.put("prompt_tokens", 12);
            usage.put("completion_tokens", 4);
            result.put("usage", usage);
            return result;
          }

          @Override
          public Builder stream(
              List<ChatMessage> messages,
              SamplingOptions options,
              List<ToolSpec> tools,
              java.util.function.Consumer<String> onToken) {
            onToken.accept("streamed");
            return chat(messages, options, tools);
          }
        };

    ModelFleet<TinystructChatModel> fleet =
        ModelFleet.<TinystructChatModel>builder()
            .model(
                ModelCandidate.builder("local-primary")
                    .local(true)
                    .costPerMillionTokens(0.0, 0.0)
                    .timeToFirstTokenMillis(50)
                    .tokensPerSecond(30)
                    .contextWindow(4096)
                    .quality(Map.of("chat", 0.95))
                    .build(),
                failingPrimary)
            .model(
                ModelCandidate.builder("backup-secondary")
                    .local(false)
                    .costPerMillionTokens(10.0, 10.0)
                    .timeToFirstTokenMillis(100)
                    .tokensPerSecond(20)
                    .contextWindow(8192)
                    .quality(Map.of("chat", 0.85))
                    .build(),
                backup)
            .build();

    RoutedTinystructChatModel routed = new RoutedTinystructChatModel(fleet);
    Builder chatResponse = routed.chat("Hello router");
    assertThat(chatResponse.get("text")).isEqualTo("backup answer");

    List<String> streamed = new ArrayList<>();
    Builder streamResponse =
        routed.stream(
            List.of(ChatMessage.user("Call tool"), ChatMessage.tool("weather", "{\"temp\":72}")),
            null,
            List.of(),
            streamed::add);
    assertThat(streamed).containsExactly("streamed");
    assertThat(streamResponse.get("text")).isEqualTo("backup answer");

    assertThat(RoutedTinystructChatModel.extractUsage(null)).isNull();
    assertThat(RoutedTinystructChatModel.extractUsage(new Builder())).isNull();
    Builder zeroUsageResponse = new Builder();
    Builder zeroUsage = new Builder();
    zeroUsage.put("prompt_tokens", 0);
    zeroUsage.put("completion_tokens", 0);
    zeroUsageResponse.put("usage", zeroUsage);
    assertThat(RoutedTinystructChatModel.extractUsage(zeroUsageResponse)).isNull();
    assertThat(RoutedTinystructChatModel.extractUsage(chatResponse))
        .isEqualTo(new RoutingUsage(12, 4));
  }
}
