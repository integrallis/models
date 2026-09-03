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

import com.integrallis.models.router.ModelCandidate;
import com.integrallis.models.router.ModelFleet;
import com.integrallis.models.router.RoutingPolicy;
import com.integrallis.models.router.RoutingRequest;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicReference;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.model.Generation;
import org.springframework.ai.chat.prompt.Prompt;
import reactor.core.publisher.Flux;

@Tag("unit")
class RoutedSpringAiChatModelTest {

  @Test
  void routesBetweenLocalAndHostedSpringAiModelsAndFallsBack() {
    AtomicReference<Prompt> hostedPrompt = new AtomicReference<>();
    ChatModel failingLocal =
        model(
            prompt -> {
              throw new IllegalStateException("local runtime busy");
            });
    ChatModel hosted =
        model(
            prompt -> {
              hostedPrompt.set(prompt);
              return response("hosted answer");
            });
    ModelFleet<ChatModel> fleet =
        ModelFleet.<ChatModel>builder()
            .model(candidate("local", true, 0.9, 0), failingLocal)
            .model(candidate("openai/gpt", false, 0.8, 1), hosted)
            .policy(RoutingPolicy.CHEAPEST)
            .build();
    RoutedSpringAiChatModel routed =
        new RoutedSpringAiChatModel(
            fleet,
            prompt ->
                RoutingRequest.builder(prompt.getContents()).sessionId("spring-session").build());
    Prompt prompt = new Prompt("hello");

    ChatResponse answer = routed.call(prompt);

    assertThat(answer.getResult().getOutput().getText()).isEqualTo("hosted answer");
    assertThat(hostedPrompt.get()).isSameAs(prompt);
    assertThat(fleet.router().status("local").orElseThrow().consecutiveFailures()).isOne();
  }

  @Test
  void preservesStreamingWhenTheSelectedClientIsHosted() {
    ChatModel local = model(prompt -> response("local"));
    ChatModel hosted =
        new ChatModel() {
          @Override
          public ChatResponse call(Prompt prompt) {
            return response("hosted");
          }

          @Override
          public Flux<ChatResponse> stream(Prompt prompt) {
            return Flux.just(response("hosted "), response("stream"));
          }
        };
    ModelFleet<ChatModel> fleet =
        ModelFleet.<ChatModel>builder()
            .model(candidate("local", true, 0.5, 0), local)
            .model(candidate("anthropic/claude", false, 0.9, 5), hosted)
            .policy(RoutingPolicy.BEST_QUALITY)
            .build();
    RoutedSpringAiChatModel routed = new RoutedSpringAiChatModel(fleet);

    String answer =
        routed.stream(new Prompt("reason carefully"))
            .map(ChatResponse::getResult)
            .map(generation -> generation.getOutput().getText())
            .collectList()
            .blockOptional()
            .orElseThrow()
            .stream()
            .reduce("", String::concat);

    assertThat(answer).isEqualTo("hosted stream");
  }

  @Test
  void streamingFallsBackWhenTheFirstClientFailsBeforeReturningAFlux() {
    ChatModel local =
        new ChatModel() {
          @Override
          public ChatResponse call(Prompt prompt) {
            throw new UnsupportedOperationException();
          }

          @Override
          public Flux<ChatResponse> stream(Prompt prompt) {
            throw new IllegalStateException("local stream unavailable");
          }
        };
    ChatModel hosted = model(prompt -> response("hosted"));
    ModelFleet<ChatModel> fleet =
        ModelFleet.<ChatModel>builder()
            .model(candidate("local", true, 0.9, 0), local)
            .model(candidate("hosted", false, 0.8, 10), hosted)
            .policy(RoutingPolicy.CHEAPEST)
            .build();

    ChatResponse answer =
        new RoutedSpringAiChatModel(fleet).stream(new Prompt("question")).blockLast();

    assertThat(answer).isNotNull();
    assertThat(answer.getResult().getOutput().getText()).isEqualTo("hosted");
    assertThat(fleet.router().status("local").orElseThrow().consecutiveFailures()).isOne();
  }

  @Test
  void streamingDoesNotMixProvidersAfterAResponseWasEmitted() {
    AtomicInteger hostedCalls = new AtomicInteger();
    IllegalStateException interrupted = new IllegalStateException("stream interrupted");
    ChatModel local =
        new ChatModel() {
          @Override
          public ChatResponse call(Prompt prompt) {
            throw new UnsupportedOperationException();
          }

          @Override
          public Flux<ChatResponse> stream(Prompt prompt) {
            return Flux.concat(Flux.just(response("partial")), Flux.error(interrupted));
          }
        };
    ChatModel hosted =
        model(
            prompt -> {
              hostedCalls.incrementAndGet();
              return response("hosted");
            });
    ModelFleet<ChatModel> fleet =
        ModelFleet.<ChatModel>builder()
            .model(candidate("local", true, 0.9, 0), local)
            .model(candidate("hosted", false, 0.8, 10), hosted)
            .policy(RoutingPolicy.CHEAPEST)
            .build();
    RoutedSpringAiChatModel routed = new RoutedSpringAiChatModel(fleet);

    assertThatThrownBy(() -> routed.stream(new Prompt("question")).collectList().block())
        .isSameAs(interrupted);

    assertThat(hostedCalls).hasValue(0);
  }

  @Test
  void keepsTheSameModelForTheToolResultTurn() {
    AtomicInteger localCalls = new AtomicInteger();
    AtomicInteger hostedCalls = new AtomicInteger();
    AtomicInteger routingTurns = new AtomicInteger();
    ChatModel local =
        model(
            prompt -> {
              localCalls.incrementAndGet();
              return response("local");
            });
    ChatModel hosted =
        model(
            prompt -> {
              hostedCalls.incrementAndGet();
              return response("hosted");
            });
    ModelCandidate localCandidate =
        ModelCandidate.builder("local")
            .local(true)
            .tags(Set.of("local-task", "hosted-task"))
            .timeToFirstTokenMillis(100)
            .tokensPerSecond(20)
            .quality(Map.of("local-task", 1.0, "hosted-task", 0.0))
            .build();
    ModelCandidate hostedCandidate =
        ModelCandidate.builder("hosted")
            .tags(Set.of("local-task", "hosted-task"))
            .timeToFirstTokenMillis(100)
            .tokensPerSecond(20)
            .quality(Map.of("local-task", 0.0, "hosted-task", 1.0))
            .build();
    ModelFleet<ChatModel> fleet =
        ModelFleet.<ChatModel>builder()
            .model(localCandidate, local)
            .model(hostedCandidate, hosted)
            .policy(RoutingPolicy.BEST_QUALITY)
            .build();
    RoutedSpringAiChatModel routed =
        new RoutedSpringAiChatModel(
            fleet,
            prompt ->
                RoutingRequest.builder(prompt.getContents())
                    .taskType(routingTurns.getAndIncrement() == 0 ? "local-task" : "hosted-task")
                    .sessionId("tool-session")
                    .build());

    routed.call(new Prompt("use a tool"));
    routed.call(
        new Prompt(
            java.util.List.of(
                new UserMessage("use a tool"),
                ToolResponseMessage.builder()
                    .responses(
                        java.util.List.of(
                            new ToolResponseMessage.ToolResponse("call-1", "lookup", "result")))
                    .build())));

    assertThat(localCalls).hasValue(2);
    assertThat(hostedCalls).hasValue(0);
  }

  private static ChatModel model(java.util.function.Function<Prompt, ChatResponse> call) {
    return new ChatModel() {
      @Override
      public ChatResponse call(Prompt prompt) {
        return call.apply(prompt);
      }

      @Override
      public Flux<ChatResponse> stream(Prompt prompt) {
        return Flux.defer(() -> Flux.just(call(prompt)));
      }
    };
  }

  private static ChatResponse response(String text) {
    return new ChatResponse(java.util.List.of(new Generation(new AssistantMessage(text))));
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
