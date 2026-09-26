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
import java.math.BigDecimal;
import java.time.Duration;
import java.util.List;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import java.util.function.Function;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.ai.chat.messages.*;
import org.springframework.ai.chat.metadata.*;
import org.springframework.ai.chat.model.*;
import org.springframework.ai.chat.prompt.*;
import reactor.core.publisher.Flux;

@Tag("unit")
class RoutedExecutionControlsTest {
  private static Prompt prompt() {
    return new Prompt("hello", ChatOptions.builder().maxTokens(100).build());
  }

  private static ModelCandidate candidate(String id) {
    return ModelCandidate.builder(id).costPerMillionTokens(1000, 1000).build();
  }

  private static ChatModel model(Function<Prompt, Flux<ChatResponse>> action) {
    return new ChatModel() {
      @Override
      public ChatResponse call(Prompt prompt) {
        return action.apply(prompt).blockLast();
      }

      @Override
      public Flux<ChatResponse> stream(Prompt prompt) {
        return action.apply(prompt);
      }
    };
  }

  private static RoutedSpringAiChatModel routed(
      ModelFleet<ChatModel> fleet, RoutingExecutionOptions options) {
    return new RoutedSpringAiChatModel(
        fleet,
        ignored -> RoutingRequest.builder("hello").build(),
        ignored -> RoutingContinuity.none(),
        ignored -> RoutingRequirements.none(),
        ignored -> options);
  }

  private static ChatResponse response(Usage usage) {
    return new ChatResponse(
        List.of(new Generation(new AssistantMessage("answer"))),
        ChatResponseMetadata.builder().usage(usage).build());
  }

  private static RoutingExecutionOptions options(RoutingBudget budget) {
    return RoutingExecutionOptions.builder()
        .tokenBounds(new RoutingTokenBounds(100, 100))
        .sharedBudget(budget)
        .build();
  }

  @Test
  void settlesCumulativeUsageOnceForBlockingAndStreaming() {
    for (boolean stream : List.of(false, true)) {
      RoutingBudget budget = new RoutingBudget(BigDecimal.ONE);
      AtomicReference<Prompt> received = new AtomicReference<>();
      var fleet =
          ModelFleet.<ChatModel>builder()
              .model(
                  candidate("a"),
                  model(
                      p -> {
                        received.set(p);
                        return Flux.just(
                            response(new DefaultUsage(10, 5)), response(new DefaultUsage(10, 20)));
                      }))
              .build();
      var routed = routed(fleet, options(budget));
      Prompt prompt = prompt();
      if (stream) routed.stream(prompt).blockLast();
      else routed.call(prompt);
      assertThat(received.get()).isSameAs(prompt);
      assertThat(budget.snapshot().spent()).isEqualByComparingTo("0.03");
      assertThat(budget.snapshot().reserved()).isEqualByComparingTo("0");
    }
  }

  @Test
  void unknownUsageAndDisposalRetainReservation() {
    for (boolean dispose : List.of(false, true)) {
      RoutingBudget budget = new RoutingBudget(BigDecimal.ONE);
      AtomicBoolean upstreamCancelled = new AtomicBoolean();
      var fleet =
          ModelFleet.<ChatModel>builder()
              .model(
                  candidate("a"),
                  model(
                      p ->
                          dispose
                              ? Flux.<ChatResponse>never()
                                  .doOnCancel(() -> upstreamCancelled.set(true))
                              : Flux.just(response(new EmptyUsage()))))
              .build();
      var routed = routed(fleet, options(budget));
      if (dispose) {
        var subscription = routed.stream(prompt()).subscribe();
        assertThat(budget.snapshot().reserved()).isEqualByComparingTo("0.2");
        subscription.dispose();
        assertThat(upstreamCancelled).isTrue();
      } else routed.stream(prompt()).blockLast();
      assertThat(budget.snapshot().spent()).isEqualByComparingTo("0.2");
      assertThat(budget.snapshot().reserved()).isEqualByComparingTo("0");
      assertThat(fleet.router().status("a").orElseThrow().consecutiveFailures()).isZero();
    }
  }

  @Test
  void absoluteDeadlineCancelsEvenWhenChunksKeepArriving() throws Exception {
    CountDownLatch upstreamCancelled = new CountDownLatch(1);
    AtomicInteger fallback = new AtomicInteger(), chunks = new AtomicInteger();
    CountDownLatch terminal = new CountDownLatch(1);
    AtomicReference<Throwable> error = new AtomicReference<>();
    var fleet =
        ModelFleet.<ChatModel>builder()
            .model(
                candidate("a"),
                model(
                    p ->
                        Flux.interval(Duration.ofMillis(10))
                            .map(i -> response(new EmptyUsage()))
                            .doOnCancel(() -> upstreamCancelled.countDown())))
            .model(
                candidate("b"),
                model(
                    p -> {
                      fallback.incrementAndGet();
                      return Flux.empty();
                    }))
            .build();
    var options = RoutingExecutionOptions.builder().timeout(Duration.ofSeconds(1)).build();
    var subscription =
        routed(fleet, options).stream(prompt())
            .subscribe(
                value -> chunks.incrementAndGet(),
                failure -> {
                  error.set(failure);
                  terminal.countDown();
                });
    try {
      assertThat(terminal.await(3, TimeUnit.SECONDS)).isTrue();
    } finally {
      subscription.dispose();
    }
    assertThat(error.get()).isInstanceOf(CancellationException.class);
    assertThat(chunks.get()).isGreaterThan(1);
    assertThat(upstreamCancelled.await(3, TimeUnit.SECONDS)).isTrue();
    assertThat(fallback).hasValue(0);
    assertThat(fleet.router().status("a").orElseThrow().consecutiveFailures()).isZero();
  }

  @Test
  void explicitCancellationBeforeFirstChunkCancelsUpstream() {
    RoutingCancellationToken token = new RoutingCancellationToken();
    AtomicBoolean upstreamCancelled = new AtomicBoolean();
    AtomicReference<Throwable> error = new AtomicReference<>();
    var fleet =
        ModelFleet.<ChatModel>builder()
            .model(
                candidate("a"),
                model(
                    p -> Flux.<ChatResponse>never().doOnCancel(() -> upstreamCancelled.set(true))))
            .build();
    var subscription =
        routed(fleet, RoutingExecutionOptions.builder().cancellation(token).build()).stream(
                prompt())
            .subscribe(value -> {}, error::set);
    token.cancel();
    subscription.dispose();
    assertThat(error.get()).isInstanceOf(CancellationException.class);
    assertThat(upstreamCancelled).isTrue();
  }

  @Test
  void interruptedSubscriptionDoesNotEvaluateFactories() {
    AtomicInteger factories = new AtomicInteger();
    AtomicReference<Throwable> error = new AtomicReference<>();
    var fleet =
        ModelFleet.<ChatModel>builder()
            .model(
                candidate("a"),
                model(
                    p -> {
                      throw new AssertionError();
                    }))
            .build();
    var routed =
        new RoutedSpringAiChatModel(
            fleet,
            p -> {
              factories.incrementAndGet();
              return RoutingRequest.builder("hello").build();
            });
    Thread.currentThread().interrupt();
    try {
      routed.stream(prompt()).subscribe(value -> {}, error::set);
      assertThat(Thread.currentThread().isInterrupted()).isTrue();
    } finally {
      Thread.interrupted();
    }
    assertThat(factories).hasValue(0);
    assertThat(error.get()).isInstanceOf(CancellationException.class);
  }

  @Test
  void accountingIncludesHistorySystemToolsMediaAndOutput() {
    Prompt prompt =
        new Prompt(
            List.of(
                new SystemMessage("system"),
                new UserMessage("old"),
                new AssistantMessage("previous"),
                new UserMessage("new")));
    var bounds = RoutingRequests.account(prompt, String::length, media -> 500, 7, 100);
    assertThat(bounds.inputTokens()).isEqualTo(6 + 3 + 8 + 3 + 7);
    assertThat(bounds.contextTokens()).isEqualTo(127);
    assertThat(RoutingRequests.estimate(prompt).estimatedTokens()).isGreaterThan(3);
  }

  @Test
  void reportedOverrunKeepsBudgetFailureAndNeverCallsFallback() {
    RoutingBudget budget = new RoutingBudget(BigDecimal.ONE);
    AtomicInteger fallback = new AtomicInteger();
    var fleet =
        ModelFleet.<ChatModel>builder()
            .model(candidate("a"), model(p -> Flux.just(response(new DefaultUsage(300, 100)))))
            .model(
                candidate("b"),
                model(
                    p -> {
                      fallback.incrementAndGet();
                      return Flux.empty();
                    }))
            .build();
    assertThatThrownBy(() -> routed(fleet, options(budget)).stream(prompt()).blockLast())
        .isInstanceOf(RoutingBudgetExceededException.class);
    assertThat(budget.snapshot().spent()).isEqualByComparingTo("0.4");
    assertThat(budget.snapshot().reserved()).isEqualByComparingTo("0");
    assertThat(fallback).hasValue(0);
    assertThat(fleet.router().status("a").orElseThrow().consecutiveFailures()).isZero();
  }
}
