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

import static org.assertj.core.api.Assertions.assertThat;

import com.integrallis.models.router.ModelCandidate;
import com.integrallis.models.router.ModelFleet;
import com.integrallis.models.router.RoutingPolicy;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import java.util.Map;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class RoutedChatModelTest {

  @Test
  void routesAnUnchangedLangChain4jRequestToAHostedClient() {
    AtomicReference<ChatRequest> received = new AtomicReference<>();
    ChatModel local = model(request -> response("local"));
    ChatModel hosted =
        model(
            request -> {
              received.set(request);
              return response("hosted");
            });
    ModelFleet<ChatModel> fleet =
        ModelFleet.<ChatModel>builder()
            .model(candidate("local", true, 0.6, 0), local)
            .model(candidate("openai/gpt", false, 0.95, 10), hosted)
            .policy(RoutingPolicy.BEST_QUALITY)
            .build();
    RoutedChatModel routed = new RoutedChatModel(fleet);
    ChatRequest request = ChatRequest.builder().messages(UserMessage.from("hard question")).build();

    ChatResponse answer = routed.doChat(request);

    assertThat(answer.aiMessage().text()).isEqualTo("hosted");
    assertThat(received.get()).isSameAs(request);
  }

  private static ChatResponse response(String text) {
    return ChatResponse.builder().aiMessage(AiMessage.from(text)).build();
  }

  private static ChatModel model(java.util.function.Function<ChatRequest, ChatResponse> call) {
    return new ChatModel() {
      @Override
      public ChatResponse doChat(ChatRequest request) {
        return call.apply(request);
      }
    };
  }

  private static ModelCandidate candidate(String id, boolean local, double quality, double cost) {
    return ModelCandidate.builder(id)
        .local(local)
        .costPerMillionTokens(cost, cost)
        .timeToFirstTokenMillis(100)
        .tokensPerSecond(20)
        .quality(Map.of("chat", quality))
        .build();
  }
}
