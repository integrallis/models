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
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.BiConsumer;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class RoutedStreamingChatModelTest {

  @Test
  void routesAnUnchangedRequestToAHostedStreamingClient() {
    AtomicReference<ChatRequest> received = new AtomicReference<>();
    StreamingChatModel local = model((request, handler) -> complete(handler, "local"));
    StreamingChatModel hosted =
        model(
            (request, handler) -> {
              received.set(request);
              handler.onPartialResponse("hosted");
              complete(handler, "hosted");
            });
    RoutedStreamingChatModel routed =
        new RoutedStreamingChatModel(
            ModelFleet.<StreamingChatModel>builder()
                .model(candidate("local", true, 0.6, 0), local)
                .model(candidate("openai/gpt", false, 0.95, 10), hosted)
                .policy(RoutingPolicy.BEST_QUALITY)
                .build());
    ChatRequest request = ChatRequest.builder().messages(UserMessage.from("hard question")).build();
    RecordingHandler answer = new RecordingHandler();

    routed.doChat(request, answer);

    assertThat(received.get()).isSameAs(request);
    assertThat(answer.partials).containsExactly("hosted");
    assertThat(answer.completed.get().aiMessage().text()).isEqualTo("hosted");
    assertThat(answer.failed.get()).isNull();
  }

  @Test
  void fallsBackOnlyWhenTheFailedClientHasNotEmittedContent() {
    RuntimeException unavailable = new RuntimeException("provider unavailable");
    StreamingChatModel local = model((request, handler) -> handler.onError(unavailable));
    StreamingChatModel hosted = model((request, handler) -> complete(handler, "hosted"));
    ModelFleet<StreamingChatModel> fleet =
        ModelFleet.<StreamingChatModel>builder()
            .model(candidate("local", true, 0.9, 0), local)
            .model(candidate("hosted", false, 0.8, 10), hosted)
            .policy(RoutingPolicy.CHEAPEST)
            .build();
    RoutedStreamingChatModel routed = new RoutedStreamingChatModel(fleet);
    RecordingHandler answer = new RecordingHandler();

    routed.doChat(ChatRequest.builder().messages(UserMessage.from("question")).build(), answer);

    assertThat(answer.completed.get().aiMessage().text()).isEqualTo("hosted");
    assertThat(answer.failed.get()).isNull();
    assertThat(fleet.router().status("local").orElseThrow().consecutiveFailures()).isOne();
  }

  @Test
  void doesNotMixProvidersAfterStreamingStarts() {
    RuntimeException interrupted = new RuntimeException("connection interrupted");
    StreamingChatModel local =
        model(
            (request, handler) -> {
              handler.onPartialResponse("partial");
              handler.onError(interrupted);
            });
    AtomicInteger hostedCalls = new AtomicInteger();
    StreamingChatModel hosted =
        model(
            (request, handler) -> {
              hostedCalls.incrementAndGet();
              complete(handler, "hosted");
            });
    RoutedStreamingChatModel routed =
        new RoutedStreamingChatModel(
            ModelFleet.<StreamingChatModel>builder()
                .model(candidate("local", true, 0.9, 0), local)
                .model(candidate("hosted", false, 0.8, 10), hosted)
                .policy(RoutingPolicy.CHEAPEST)
                .build());
    RecordingHandler answer = new RecordingHandler();

    routed.doChat(ChatRequest.builder().messages(UserMessage.from("question")).build(), answer);

    assertThat(answer.partials).containsExactly("partial");
    assertThat(answer.failed.get()).isSameAs(interrupted);
    assertThat(answer.completed.get()).isNull();
    assertThat(hostedCalls).hasValue(0);
  }

  private static void complete(StreamingChatResponseHandler handler, String text) {
    handler.onCompleteResponse(ChatResponse.builder().aiMessage(AiMessage.from(text)).build());
  }

  private static StreamingChatModel model(
      BiConsumer<ChatRequest, StreamingChatResponseHandler> call) {
    return new StreamingChatModel() {
      @Override
      public void doChat(ChatRequest request, StreamingChatResponseHandler handler) {
        call.accept(request, handler);
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

  private static final class RecordingHandler implements StreamingChatResponseHandler {
    private final List<String> partials = new ArrayList<>();
    private final AtomicReference<ChatResponse> completed = new AtomicReference<>();
    private final AtomicReference<Throwable> failed = new AtomicReference<>();

    @Override
    public void onPartialResponse(String partialResponse) {
      partials.add(partialResponse);
    }

    @Override
    public void onCompleteResponse(ChatResponse response) {
      completed.set(response);
    }

    @Override
    public void onError(Throwable error) {
      failed.set(error);
    }
  }
}
