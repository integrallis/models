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
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.*;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.function.BiConsumer;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;

@Tag("unit")
class RoutedCancellationTest {
  @ParameterizedTest
  @CsvSource({
    "false,false,false,false",
    "true,false,false,false",
    "false,true,false,false",
    "true,true,false,false",
    "false,false,true,false",
    "true,false,true,false",
    "true,false,false,true",
    "true,true,false,true"
  })
  void cancellationIsTerminalWithoutHealthPenalty(
      boolean wrapped, boolean partial, boolean thrown, boolean interrupted) {
    Throwable cause =
        interrupted
            ? new InterruptedException("cancelled elsewhere")
            : new CancellationException("cancelled");
    RuntimeException failure = wrapped ? new CompletionException(cause) : (RuntimeException) cause;
    AtomicInteger fallback = new AtomicInteger(), terminals = new AtomicInteger();
    AtomicReference<Throwable> error = new AtomicReference<>();
    StreamingChatModel local =
        model(
            (request, handler) -> {
              if (partial) handler.onPartialResponse("partial");
              if (thrown) throw failure;
              handler.onError(failure);
              handler.onError(failure);
              handler.onCompleteResponse(
                  ChatResponse.builder().aiMessage(AiMessage.from("late")).build());
            });
    var fleet =
        ModelFleet.<StreamingChatModel>builder()
            .model(candidate("local", 0), local)
            .model(
                candidate("hosted", 1),
                model(
                    (request, handler) -> {
                      fallback.incrementAndGet();
                      handler.onCompleteResponse(
                          ChatResponse.builder().aiMessage(AiMessage.from("hosted")).build());
                    }))
            .policy(RoutingPolicy.CHEAPEST)
            .build();
    new RoutedStreamingChatModel(fleet)
        .doChat(
            ChatRequest.builder().messages(UserMessage.from("question")).build(),
            new StreamingChatResponseHandler() {
              @Override
              public void onCompleteResponse(ChatResponse response) {
                terminals.incrementAndGet();
              }

              @Override
              public void onError(Throwable failure) {
                error.set(failure);
                terminals.incrementAndGet();
              }
            });
    assertThat(fallback).hasValue(0);
    assertThat(terminals).hasValue(1);
    assertThat(error.get()).isSameAs(failure);
    assertThat(fleet.router().status("local").orElseThrow().consecutiveFailures()).isZero();
    assertThat(Thread.currentThread().isInterrupted()).isFalse();
  }

  @org.junit.jupiter.api.Test
  void cancellationInFallbackNeverStartsTheNextProvider() {
    AtomicInteger lastCalls = new AtomicInteger();
    AtomicReference<Throwable> result = new AtomicReference<>();
    CancellationException cancelled = new CancellationException("cancelled in fallback");
    var fleet =
        ModelFleet.<StreamingChatModel>builder()
            .model(
                candidate("first", 0),
                model(
                    (request, handler) ->
                        handler.onError(new IllegalStateException("unavailable"))))
            .model(candidate("second", 1), model((request, handler) -> handler.onError(cancelled)))
            .model(candidate("last", 2), model((request, handler) -> lastCalls.incrementAndGet()))
            .policy(RoutingPolicy.CHEAPEST)
            .build();
    new RoutedStreamingChatModel(fleet)
        .doChat(
            ChatRequest.builder().messages(UserMessage.from("question")).build(),
            new StreamingChatResponseHandler() {
              @Override
              public void onCompleteResponse(ChatResponse response) {
                throw new AssertionError("unexpected success");
              }

              @Override
              public void onError(Throwable failure) {
                result.set(failure);
              }
            });
    assertThat(result.get()).isSameAs(cancelled);
    assertThat(lastCalls).hasValue(0);
    assertThat(fleet.router().status("first").orElseThrow().consecutiveFailures()).isOne();
    assertThat(fleet.router().status("second").orElseThrow().consecutiveFailures()).isZero();
  }

  private static ModelCandidate candidate(String id, double cost) {
    return ModelCandidate.builder(id).local(cost == 0).costPerMillionTokens(cost, cost).build();
  }

  private static StreamingChatModel model(
      BiConsumer<ChatRequest, StreamingChatResponseHandler> action) {
    return new StreamingChatModel() {
      @Override
      public void doChat(ChatRequest request, StreamingChatResponseHandler handler) {
        action.accept(request, handler);
      }
    };
  }
}
