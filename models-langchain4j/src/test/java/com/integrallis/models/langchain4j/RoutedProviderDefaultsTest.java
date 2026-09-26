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
package com.integrallis.models.langchain4j;

import static org.assertj.core.api.Assertions.*;

import com.integrallis.models.router.*;
import dev.langchain4j.data.message.*;
import dev.langchain4j.model.chat.*;
import dev.langchain4j.model.chat.request.*;
import dev.langchain4j.model.chat.response.*;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class RoutedProviderDefaultsTest {
  private static ChatRequest request() {
    return ChatRequest.builder()
        .messages(UserMessage.from("hello"))
        .parameters(DefaultChatRequestParameters.builder().maxOutputTokens(10).build())
        .build();
  }

  private static ChatRequestParameters defaults() {
    return DefaultChatRequestParameters.builder()
        .modelName("provider-model")
        .temperature(0.2)
        .maxOutputTokens(999)
        .build();
  }

  private static void verify(ChatRequest actual, ChatRequest original) {
    assertThat(actual.messages()).isEqualTo(original.messages());
    assertThat(actual.modelName()).isEqualTo("provider-model");
    assertThat(actual.temperature()).isEqualTo(0.2);
    assertThat(actual.maxOutputTokens()).isEqualTo(10);
  }

  @Test
  void blockingUsesProviderPublicApiToApplyDefaultsAndKeepExplicitLimits() {
    AtomicReference<ChatRequest> seen = new AtomicReference<>();
    ChatModel provider =
        new ChatModel() {
          @Override
          public ChatRequestParameters defaultRequestParameters() {
            return defaults();
          }

          @Override
          public ChatResponse doChat(ChatRequest request) {
            seen.set(request);
            return ChatResponse.builder().aiMessage(AiMessage.from("answer")).build();
          }
        };
    var fleet =
        ModelFleet.<ChatModel>builder()
            .model(ModelCandidate.builder("a").build(), provider)
            .build();
    ChatRequest request = request();
    assertThat(new RoutedChatModel(fleet).doChat(request).aiMessage().text()).isEqualTo("answer");
    verify(seen.get(), request);
  }

  @Test
  void streamingUsesProviderPublicApiToApplyDefaultsAndKeepExplicitLimits() {
    AtomicReference<ChatRequest> seen = new AtomicReference<>();
    AtomicReference<ChatResponse> response = new AtomicReference<>();
    StreamingChatModel provider =
        new StreamingChatModel() {
          @Override
          public ChatRequestParameters defaultRequestParameters() {
            return defaults();
          }

          @Override
          public void doChat(ChatRequest request, StreamingChatResponseHandler handler) {
            seen.set(request);
            handler.onCompleteResponse(
                ChatResponse.builder().aiMessage(AiMessage.from("answer")).build());
          }
        };
    var fleet =
        ModelFleet.<StreamingChatModel>builder()
            .model(ModelCandidate.builder("a").build(), provider)
            .build();
    ChatRequest request = request();
    new RoutedStreamingChatModel(fleet)
        .doChat(
            request,
            new StreamingChatResponseHandler() {
              @Override
              public void onPartialResponse(String text) {}

              @Override
              public void onCompleteResponse(ChatResponse result) {
                response.set(result);
              }

              @Override
              public void onError(Throwable error) {
                throw new AssertionError(error);
              }
            });
    assertThat(response.get().aiMessage().text()).isEqualTo("answer");
    verify(seen.get(), request);
  }
}
