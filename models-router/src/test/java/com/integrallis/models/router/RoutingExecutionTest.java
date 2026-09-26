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
package com.integrallis.models.router;

import static org.assertj.core.api.Assertions.*;

import java.math.BigDecimal;
import java.time.Duration;
import java.util.concurrent.*;
import java.util.concurrent.atomic.*;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class RoutingExecutionTest {
  private static ModelCandidate candidate(String id, double cost) {
    return ModelCandidate.builder(id).local(true).costPerMillionTokens(cost, cost).build();
  }

  private static RoutingExecutionOptions options(RoutingBudget budget) {
    return RoutingExecutionOptions.builder()
        .tokenBounds(new RoutingTokenBounds(100, 100))
        .sharedBudget(budget)
        .build();
  }

  private static RoutingRequest request() {
    return RoutingRequest.builder("test").build();
  }

  @Test
  void settlesKnownUsageAndChargesUnknownOnlyOnce() {
    RoutingBudget budget = new RoutingBudget(new BigDecimal("1"));
    try (RoutingExecution execution = new RoutingExecution(options(budget))) {
      var first = execution.beginAttempt(candidate("a", 1000));
      assertThat(budget.snapshot().reserved()).isEqualByComparingTo("0.2");
      first.settle(new RoutingUsage(10, 20));
      first.close();
      execution.beginAttempt(candidate("b", 1000)).close();
    }
    assertThat(budget.snapshot().spent()).isEqualByComparingTo("0.23");
    assertThat(budget.snapshot().reserved()).isEqualByComparingTo("0");
  }

  @Test
  void concurrentRequestsCannotOversubscribeSharedBudget() throws Exception {
    RoutingBudget budget = new RoutingBudget(new BigDecimal("0.2"));
    CountDownLatch start = new CountDownLatch(1);
    AtomicInteger admitted = new AtomicInteger();
    try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
      var futures =
          java.util.stream.IntStream.range(0, 40)
              .mapToObj(
                  i ->
                      executor.submit(
                          () -> {
                            start.await();
                            try (RoutingExecution execution =
                                new RoutingExecution(options(budget))) {
                              execution.beginAttempt(candidate("a", 1000));
                              admitted.incrementAndGet();
                            } catch (RoutingBudgetExceededException expected) {
                            }
                            return null;
                          }))
              .toList();
      start.countDown();
      for (var future : futures) future.get(5, TimeUnit.SECONDS);
    }
    assertThat(admitted).hasValue(1);
    assertThat(budget.snapshot().spent()).isEqualByComparingTo("0.2");
  }

  @Test
  void retriesConsumeTheSameRequestBudgetAndDoNotPenalizeUncalledModels() {
    AtomicInteger fallback = new AtomicInteger();
    ModelFleet<Callable<String>> fleet =
        ModelFleet.<Callable<String>>builder()
            .model(
                candidate("a", 1000),
                () -> {
                  throw new IllegalStateException("unknown billable failure");
                })
            .model(
                candidate("b", 2000),
                () -> {
                  fallback.incrementAndGet();
                  return "answer";
                })
            .policy(RoutingPolicy.CHEAPEST)
            .build();
    var controls =
        RoutingExecutionOptions.builder()
            .tokenBounds(new RoutingTokenBounds(100, 100))
            .requestBudget(new BigDecimal("0.5"))
            .build();
    assertThatThrownBy(
            () ->
                fleet.execute(
                    request(),
                    RoutingRequirements.none(),
                    RoutingContinuity.none(),
                    controls,
                    Callable::call,
                    ignored -> null))
        .isInstanceOf(RoutingBudgetExceededException.class);
    assertThat(fallback).hasValue(0);
    assertThat(fleet.router().status("a").orElseThrow().consecutiveFailures()).isEqualTo(1);
    assertThat(fleet.router().status("b").orElseThrow().consecutiveFailures()).isZero();
  }

  @Test
  void budgetCanSkipAnUnaffordableWinnerWithoutCallingIt() {
    AtomicInteger expensive = new AtomicInteger();
    ModelFleet<Callable<String>> fleet =
        ModelFleet.<Callable<String>>builder()
            .model(
                ModelCandidate.builder("a")
                    .quality(java.util.Map.of("chat", 1.0))
                    .costPerMillionTokens(1000, 1000)
                    .build(),
                () -> {
                  expensive.incrementAndGet();
                  return "costly";
                })
            .model(candidate("b", 0), () -> "free")
            .policy(RoutingPolicy.BEST_QUALITY)
            .build();
    var controls =
        RoutingExecutionOptions.builder()
            .tokenBounds(new RoutingTokenBounds(100, 100))
            .requestBudget(BigDecimal.ZERO)
            .build();
    var result =
        fleet.execute(
            request(),
            RoutingRequirements.none(),
            RoutingContinuity.none(),
            controls,
            Callable::call,
            ignored -> null);
    assertThat(result.value()).isEqualTo("free");
    assertThat(expensive).hasValue(0);
    assertThat(result.attempts()).hasSize(1);
  }

  @Test
  void overrunRecordsActualCostAndStopsInsteadOfFallingBack() {
    RoutingBudget budget = new RoutingBudget(new BigDecimal("0.2"));
    try (RoutingExecution execution = new RoutingExecution(options(budget))) {
      var attempt = execution.beginAttempt(candidate("a", 1000));
      assertThatThrownBy(() -> attempt.settle(new RoutingUsage(300, 100)))
          .isInstanceOf(RoutingBudgetExceededException.class);
      attempt.close();
    }
    assertThat(budget.snapshot().spent()).isEqualByComparingTo("0.4");
    assertThat(budget.snapshot().available()).isEqualByComparingTo("-0.2");
  }

  @Test
  void failedSharedReservationDoesNotLeakRequestReservation() {
    RoutingBudget shared = new RoutingBudget(BigDecimal.ZERO);
    var controls =
        RoutingExecutionOptions.builder()
            .tokenBounds(new RoutingTokenBounds(100, 100))
            .requestBudget(new BigDecimal("0.2"))
            .sharedBudget(shared)
            .build();
    try (RoutingExecution execution = new RoutingExecution(controls)) {
      assertThatThrownBy(() -> execution.beginAttempt(candidate("a", 1000)))
          .isInstanceOf(RoutingBudgetExceededException.class);
      execution.beginAttempt(candidate("free", 0)).settle(new RoutingUsage(100, 100));
    }
    assertThat(shared.snapshot().reserved()).isEqualByComparingTo("0");
  }

  @Test
  void fakeClockDeadlineIncludesEarlierWorkAndNeverResets() {
    AtomicLong clock = new AtomicLong(Long.MAX_VALUE - 5);
    var controls = RoutingExecutionOptions.builder().timeout(Duration.ofSeconds(30)).build();
    try (RoutingExecution execution = new RoutingExecution(controls, clock::get)) {
      execution.beginAttempt(candidate("a", 0)).close();
      clock.addAndGet(Duration.ofSeconds(30).toNanos());
      assertThatThrownBy(() -> execution.beginAttempt(candidate("b", 0)))
          .isInstanceOf(CancellationException.class);
    }
  }

  @Test
  void cancellationReturnsWhileUncooperativeProviderIsStillRunning() throws Exception {
    RoutingBudget budget = new RoutingBudget(new BigDecimal("1"));
    RoutingCancellationToken token = new RoutingCancellationToken();
    CountDownLatch entered = new CountDownLatch(1),
        release = new CountDownLatch(1),
        exited = new CountDownLatch(1);
    AtomicInteger fallback = new AtomicInteger();
    ModelFleet<Callable<String>> fleet =
        ModelFleet.<Callable<String>>builder()
            .model(
                candidate("a", 1000),
                () -> {
                  entered.countDown();
                  while (release.getCount() != 0) {
                    try {
                      release.await();
                    } catch (InterruptedException ignored) {
                    }
                  }
                  exited.countDown();
                  return "late";
                })
            .model(
                candidate("b", 2000),
                () -> {
                  fallback.incrementAndGet();
                  return "fallback";
                })
            .policy(RoutingPolicy.CHEAPEST)
            .build();
    var controls =
        RoutingExecutionOptions.builder()
            .tokenBounds(new RoutingTokenBounds(100, 100))
            .sharedBudget(budget)
            .cancellation(token)
            .build();
    try (var executor = Executors.newVirtualThreadPerTaskExecutor()) {
      var result =
          executor.submit(
              () ->
                  fleet.execute(
                      request(),
                      RoutingRequirements.none(),
                      RoutingContinuity.none(),
                      controls,
                      Callable::call,
                      ignored -> null));
      try {
        assertThat(entered.await(5, TimeUnit.SECONDS)).isTrue();
        token.cancel();
        assertThatThrownBy(() -> result.get(2, TimeUnit.SECONDS))
            .isInstanceOf(ExecutionException.class)
            .hasCauseInstanceOf(CancellationException.class);
        assertThat(exited.getCount()).isEqualTo(1);
      } finally {
        release.countDown();
      }
      assertThat(exited.await(5, TimeUnit.SECONDS)).isTrue();
    }
    assertThat(fallback).hasValue(0);
    assertThat(budget.snapshot().spent()).isEqualByComparingTo("0.2");
    assertThat(fleet.router().status("a").orElseThrow().consecutiveFailures()).isZero();
  }

  @Test
  void deadlineCoversClassificationBeforeAnyClientCall() {
    CountDownLatch classifierExited = new CountDownLatch(1);
    AtomicInteger calls = new AtomicInteger();
    ModelFleet<Callable<String>> fleet =
        ModelFleet.<Callable<String>>builder()
            .model(
                candidate("a", 0),
                () -> {
                  calls.incrementAndGet();
                  return "answer";
                })
            .classifier(
                query -> {
                  try {
                    new CountDownLatch(1).await();
                  } catch (InterruptedException ignored) {
                    Thread.currentThread().interrupt();
                  } finally {
                    classifierExited.countDown();
                  }
                  return "chat";
                })
            .build();
    var controls = RoutingExecutionOptions.builder().timeout(Duration.ofMillis(100)).build();
    assertThatThrownBy(
            () ->
                fleet.execute(
                    request(),
                    RoutingRequirements.none(),
                    RoutingContinuity.none(),
                    controls,
                    Callable::call,
                    ignored -> null))
        .isInstanceOf(CancellationException.class);
    assertThat(calls).hasValue(0);
  }

  @Test
  void boundsIncludeOutputInContextAdmission() {
    ModelFleet<Callable<String>> fleet =
        ModelFleet.<Callable<String>>builder()
            .model(ModelCandidate.builder("small").contextWindow(150).build(), () -> "answer")
            .build();
    var controls =
        RoutingExecutionOptions.builder().tokenBounds(new RoutingTokenBounds(100, 100)).build();
    assertThatThrownBy(
            () ->
                fleet.execute(
                    request(),
                    RoutingRequirements.none(),
                    RoutingContinuity.none(),
                    controls,
                    Callable::call,
                    ignored -> null))
        .isInstanceOf(NoEligibleModelException.class);
  }

  @Test
  void budgetedClassificationFailsBeforeUndeclaredExternalWork() {
    AtomicInteger calls = new AtomicInteger();
    ModelFleet<String> fleet =
        ModelFleet.<String>builder()
            .model(candidate("a", 0), "client")
            .classifier(
                query -> {
                  calls.incrementAndGet();
                  return "chat";
                })
            .build();
    try (RoutingExecution execution =
        new RoutingExecution(options(new RoutingBudget(BigDecimal.ONE)))) {
      assertThatThrownBy(
              () ->
                  fleet.decide(
                      request(), RoutingRequirements.none(), RoutingContinuity.none(), execution))
          .isInstanceOf(IllegalArgumentException.class);
    }
    assertThat(calls).hasValue(0);
  }
}
