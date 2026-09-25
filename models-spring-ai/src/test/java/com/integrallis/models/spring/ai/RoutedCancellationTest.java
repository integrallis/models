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

import static org.assertj.core.api.Assertions.*;

import com.integrallis.models.router.*;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.function.Function;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.CsvSource;
import org.springframework.ai.chat.messages.AssistantMessage;
import org.springframework.ai.chat.model.*;
import org.springframework.ai.chat.prompt.Prompt;
import reactor.core.publisher.Flux;

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
    ChatModel local =
        model(
            prompt -> {
              if (thrown) throw failure;
              return partial
                  ? Flux.concat(
                      Flux.just(
                          new ChatResponse(
                              List.of(new Generation(new AssistantMessage("partial"))))),
                      Flux.error(failure))
                  : Flux.error(failure);
            });
    var fleet =
        fleet(
            local,
            model(
                prompt -> {
                  fallback.incrementAndGet();
                  return Flux.empty();
                }));
    new RoutedSpringAiChatModel(fleet)
        .stream(new Prompt("question"))
            .subscribe(
                value -> {},
                failureValue -> {
                  error.set(failureValue);
                  terminals.incrementAndGet();
                },
                terminals::incrementAndGet);
    assertThat(fallback).hasValue(0);
    assertThat(terminals).hasValue(1);
    assertThat(error.get()).isSameAs(failure);
    assertThat(fleet.router().status("local").orElseThrow().consecutiveFailures()).isZero();
    assertThat(Thread.currentThread().isInterrupted()).isFalse();
  }

  @Test
  void disposingSubscriptionCancelsUpstreamWithoutFallbackOrPenalty() {
    AtomicInteger cancellation = new AtomicInteger(), fallback = new AtomicInteger();
    var fleet =
        fleet(
            model(prompt -> Flux.<ChatResponse>never().doOnCancel(cancellation::incrementAndGet)),
            model(
                prompt -> {
                  fallback.incrementAndGet();
                  return Flux.empty();
                }));
    var subscription =
        new RoutedSpringAiChatModel(fleet).stream(new Prompt("question")).subscribe();
    subscription.dispose();
    assertThat(cancellation).hasValue(1);
    assertThat(fallback).hasValue(0);
    assertThat(fleet.router().status("local").orElseThrow().consecutiveFailures()).isZero();
    assertThat(Thread.currentThread().isInterrupted()).isFalse();
  }

  @Test
  void cancellationInFallbackNeverStartsTheNextProvider() {
    AtomicInteger lastCalls = new AtomicInteger();
    AtomicReference<Throwable> result = new AtomicReference<>();
    CancellationException cancelled = new CancellationException("cancelled in fallback");
    var fleet =
        ModelFleet.<ChatModel>builder()
            .model(
                ModelCandidate.builder("first").costPerMillionTokens(0, 0).build(),
                model(prompt -> Flux.error(new IllegalStateException("unavailable"))))
            .model(
                ModelCandidate.builder("second").costPerMillionTokens(1, 1).build(),
                model(prompt -> Flux.error(cancelled)))
            .model(
                ModelCandidate.builder("last").costPerMillionTokens(2, 2).build(),
                model(
                    prompt -> {
                      lastCalls.incrementAndGet();
                      return Flux.empty();
                    }))
            .policy(RoutingPolicy.CHEAPEST)
            .build();
    new RoutedSpringAiChatModel(fleet)
        .stream(new Prompt("question")).subscribe(value -> {}, result::set);
    assertThat(result.get()).isSameAs(cancelled);
    assertThat(lastCalls).hasValue(0);
    assertThat(fleet.router().status("first").orElseThrow().consecutiveFailures()).isOne();
    assertThat(fleet.router().status("second").orElseThrow().consecutiveFailures()).isZero();
  }

  private static ModelFleet<ChatModel> fleet(ChatModel local, ChatModel hosted) {
    return ModelFleet.<ChatModel>builder()
        .model(
            ModelCandidate.builder("local").local(true).costPerMillionTokens(0, 0).build(), local)
        .model(ModelCandidate.builder("hosted").costPerMillionTokens(1, 1).build(), hosted)
        .policy(RoutingPolicy.CHEAPEST)
        .build();
  }

  private static ChatModel model(Function<Prompt, Flux<ChatResponse>> action) {
    return new ChatModel() {
      @Override
      public ChatResponse call(Prompt prompt) {
        throw new UnsupportedOperationException();
      }

      @Override
      public Flux<ChatResponse> stream(Prompt prompt) {
        return action.apply(prompt);
      }
    };
  }
}
