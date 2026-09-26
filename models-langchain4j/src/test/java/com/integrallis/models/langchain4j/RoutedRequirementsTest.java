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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.router.ModelCandidate;
import com.integrallis.models.router.ModelFleet;
import com.integrallis.models.router.NoEligibleModelException;
import com.integrallis.models.router.RoutingContinuity;
import com.integrallis.models.router.RoutingDataBoundary;
import com.integrallis.models.router.RoutingPolicy;
import com.integrallis.models.router.RoutingRequest;
import com.integrallis.models.router.RoutingRequirements;
import dev.langchain4j.data.message.AiMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.ChatModel;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicReference;
import java.util.function.Function;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

@Tag("unit")
class RoutedRequirementsTest {
  private static final RoutingRequirements LOCAL_TOOLS =
      RoutingRequirements.builder()
          .dataBoundary(RoutingDataBoundary.LOCAL_ONLY)
          .requireCapability("tool-calling")
          .build();

  @ParameterizedTest
  @ValueSource(booleans = {false, true})
  void readsFullHistoryAndReevaluatesEachRequest(boolean streaming) {
    List<ChatRequest> received = new ArrayList<>();
    var privateRequest = request("private-history", "ordinary-question");
    var publicRequest = request("public-history", "ordinary-question");
    var routed =
        routed(
            streaming,
            List.of(
                entry(
                    "local",
                    true,
                    0.5,
                    Set.of("tool-calling"),
                    value -> {
                      received.add(value);
                      return "local";
                    }),
                entry(
                    "hosted",
                    false,
                    1.0,
                    Set.of("tool-calling"),
                    value -> {
                      received.add(value);
                      return "hosted";
                    })),
            value ->
                firstText(value).equals("private-history")
                    ? LOCAL_TOOLS
                    : RoutingRequirements.none());
    assertThat(routed.apply(privateRequest)).isEqualTo("local");
    assertThat(routed.apply(publicRequest)).isEqualTo("hosted");
    assertThat(received).hasSize(2);
    // The provider public API can copy the request while merging its own defaults.
    assertThat(received.get(0)).isEqualTo(privateRequest);
    assertThat(received.get(1)).isEqualTo(publicRequest);
  }

  @ParameterizedTest
  @ValueSource(booleans = {false, true})
  void restrictsEveryFallbackByBoundaryAndCapability(boolean streaming) {
    List<String> calls = new ArrayList<>();
    var routed =
        routed(
            streaming,
            List.of(
                entry(
                    "hosted",
                    false,
                    1.0,
                    Set.of("tool-calling"),
                    value -> {
                      calls.add("hosted");
                      return "forbidden";
                    }),
                entry(
                    "local-text",
                    true,
                    0.95,
                    Set.of(),
                    value -> {
                      calls.add("local-text");
                      return "incapable";
                    }),
                entry(
                    "local-primary",
                    true,
                    0.9,
                    Set.of("tool-calling"),
                    value -> {
                      calls.add("primary");
                      throw new IllegalStateException("busy");
                    }),
                entry(
                    "local-fallback",
                    true,
                    0.8,
                    Set.of("tool-calling"),
                    value -> {
                      calls.add("fallback");
                      return "allowed";
                    })),
            value -> LOCAL_TOOLS);
    assertThat(routed.apply(request("private", "question"))).isEqualTo("allowed");
    assertThat(calls).containsExactly("primary", "fallback");
  }

  @ParameterizedTest
  @ValueSource(booleans = {false, true})
  void neverInvokesAClientWhenNoModelMeetsRequirements(boolean streaming) {
    List<String> calls = new ArrayList<>();
    var routed =
        routed(
            streaming,
            List.of(
                entry(
                    "hosted",
                    false,
                    1.0,
                    Set.of("tool-calling"),
                    value -> {
                      calls.add("hosted");
                      return "forbidden";
                    }),
                entry(
                    "local-text",
                    true,
                    0.9,
                    Set.of(),
                    value -> {
                      calls.add("local-text");
                      return "incapable";
                    })),
            value -> LOCAL_TOOLS);
    assertThatThrownBy(() -> routed.apply(request("private", "question")))
        .isInstanceOf(NoEligibleModelException.class);
    assertThat(calls).isEmpty();
  }

  @ParameterizedTest
  @ValueSource(booleans = {false, true})
  void rejectsNullRequirementsBeforeInvokingAClient(boolean streaming) {
    List<String> calls = new ArrayList<>();
    var routed =
        routed(
            streaming,
            List.of(
                entry(
                    "hosted",
                    false,
                    1.0,
                    Set.of(),
                    value -> {
                      calls.add("hosted");
                      return "answer";
                    })),
            value -> null);
    assertThatThrownBy(() -> routed.apply(request("private", "question")))
        .isInstanceOf(NullPointerException.class)
        .hasMessage("routing requirements");
    assertThat(calls).isEmpty();
  }

  private record Entry(ModelCandidate candidate, Function<ChatRequest, String> invoke) {}

  private static Entry entry(
      String id,
      boolean local,
      double quality,
      Set<String> capabilities,
      Function<ChatRequest, String> invoke) {
    return new Entry(
        ModelCandidate.builder(id)
            .local(local)
            .capabilities(capabilities)
            .quality(Map.of("chat", quality))
            .timeToFirstTokenMillis(100)
            .tokensPerSecond(20)
            .build(),
        invoke);
  }

  private static ChatRequest request(String history, String question) {
    return ChatRequest.builder()
        .messages(UserMessage.from(history), UserMessage.from(question))
        .build();
  }

  private static String firstText(ChatRequest request) {
    return ((UserMessage) request.messages().getFirst()).singleText();
  }

  private static Function<ChatRequest, String> routed(
      boolean streaming,
      List<Entry> entries,
      Function<ChatRequest, RoutingRequirements> requirements) {
    if (!streaming) {
      var builder = ModelFleet.<ChatModel>builder().policy(RoutingPolicy.BEST_QUALITY);
      for (Entry entry : entries) {
        builder.model(
            entry.candidate(),
            new ChatModel() {
              public ChatResponse doChat(ChatRequest request) {
                return response(entry.invoke().apply(request));
              }
            });
      }
      var model =
          new RoutedChatModel(
              builder.build(),
              request -> RoutingRequest.builder("question").build(),
              request -> RoutingContinuity.none(),
              requirements);
      return request -> model.doChat(request).aiMessage().text();
    }
    var builder = ModelFleet.<StreamingChatModel>builder().policy(RoutingPolicy.BEST_QUALITY);
    for (Entry entry : entries) {
      builder.model(
          entry.candidate(),
          new StreamingChatModel() {
            public void doChat(ChatRequest request, StreamingChatResponseHandler handler) {
              handler.onCompleteResponse(response(entry.invoke().apply(request)));
            }
          });
    }
    var model =
        new RoutedStreamingChatModel(
            builder.build(),
            request -> RoutingRequest.builder("question").build(),
            request -> RoutingContinuity.none(),
            requirements);
    return request -> {
      AtomicReference<ChatResponse> response = new AtomicReference<>();
      AtomicReference<Throwable> error = new AtomicReference<>();
      model.doChat(
          request,
          new StreamingChatResponseHandler() {
            public void onPartialResponse(String value) {}

            public void onCompleteResponse(ChatResponse value) {
              response.set(value);
            }

            public void onError(Throwable value) {
              error.set(value);
            }
          });
      if (error.get() != null) {
        throw new AssertionError("unexpected stream failure", error.get());
      }
      return response.get().aiMessage().text();
    };
  }

  private static ChatResponse response(String text) {
    return ChatResponse.builder().aiMessage(AiMessage.from(text)).build();
  }
}
