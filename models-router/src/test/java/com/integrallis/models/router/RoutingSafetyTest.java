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

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatIllegalArgumentException;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.util.Map;
import java.util.concurrent.CancellationException;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

@Tag("unit")
class RoutingSafetyTest {
  @ParameterizedTest
  @ValueSource(doubles = {Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY, -0.01})
  void rejectsInvalidHardLimits(double value) {
    assertThatIllegalArgumentException()
        .isThrownBy(() -> RoutingPolicy.BALANCED.withMinimumQuality(value));
    assertThatIllegalArgumentException()
        .isThrownBy(() -> RoutingPolicy.BALANCED.withMaximumCostPerMillionTokens(value));
    assertThatIllegalArgumentException()
        .isThrownBy(() -> RoutingPolicy.BALANCED.withMaximumTimeToFirstTokenMillis(value));
  }

  @Test
  void qualityCannotExceedOneButCostAndLatencyCan() {
    assertThatIllegalArgumentException()
        .isThrownBy(() -> RoutingPolicy.BALANCED.withMinimumQuality(1.01));
    assertThat(RoutingPolicy.BALANCED.withMinimumQuality(0).minimumQuality()).hasValue(0);
    assertThat(RoutingPolicy.BALANCED.withMinimumQuality(1).minimumQuality()).hasValue(1);
    assertThat(
            RoutingPolicy.BALANCED.withMaximumCostPerMillionTokens(0).maximumCostPerMillionTokens())
        .hasValue(0);
    assertThat(
            RoutingPolicy.BALANCED
                .withMaximumCostPerMillionTokens(100)
                .maximumCostPerMillionTokens())
        .hasValue(100);
    assertThat(
            RoutingPolicy.BALANCED
                .withMaximumTimeToFirstTokenMillis(100)
                .maximumTimeToFirstTokenMillis())
        .hasValue(100);
  }

  @Test
  void interruptedDelegateNeverStartsHostedFallbackOrPenalizesModel() {
    AtomicInteger hosted = new AtomicInteger();
    ModelFleet<String> fleet = fleet();
    InterruptedException interruption = new InterruptedException("caller cancelled");
    try {
      assertThatThrownBy(
              () ->
                  fleet.execute(
                      RoutingRequest.builder("hello").build(),
                      client -> {
                        if (client.equals("local")) throw interruption;
                        hosted.incrementAndGet();
                        return "hosted answer";
                      }))
          .isInstanceOf(CancellationException.class)
          .hasCause(interruption);
      assertThat(hosted).hasValue(0);
      assertThat(Thread.currentThread().isInterrupted()).isTrue();
      assertThat(fleet.router().status("local").orElseThrow().consecutiveFailures()).isZero();
    } finally {
      Thread.interrupted();
    }
  }

  @Test
  void explicitCancellationIsPropagatedWithoutFallback() {
    AtomicInteger hosted = new AtomicInteger();
    CancellationException cancellation = new CancellationException("cancelled");
    ModelFleet<String> fleet = fleet();
    assertThatThrownBy(
            () ->
                fleet.execute(
                    RoutingRequest.builder("hello").build(),
                    client -> {
                      if (client.equals("local")) throw cancellation;
                      hosted.incrementAndGet();
                      return "hosted answer";
                    }))
        .isSameAs(cancellation);
    assertThat(hosted).hasValue(0);
    assertThat(fleet.router().status("local").orElseThrow().consecutiveFailures()).isZero();
  }

  @Test
  void alreadyInterruptedCallerInvokesNoClient() {
    AtomicInteger calls = new AtomicInteger();
    ModelFleet<String> fleet = fleet();
    Thread.currentThread().interrupt();
    try {
      assertThatThrownBy(
              () ->
                  fleet.execute(
                      RoutingRequest.builder("hello").build(), client -> calls.incrementAndGet()))
          .isInstanceOf(CancellationException.class);
      assertThat(calls).hasValue(0);
      assertThat(Thread.currentThread().isInterrupted()).isTrue();
    } finally {
      Thread.interrupted();
    }
  }

  private static ModelFleet<String> fleet() {
    return ModelFleet.<String>builder()
        .model(
            ModelCandidate.builder("local")
                .local(true)
                .costPerMillionTokens(0, 0)
                .quality(Map.of("chat", 0.9))
                .build(),
            "local")
        .model(
            ModelCandidate.builder("hosted")
                .local(false)
                .costPerMillionTokens(1, 1)
                .quality(Map.of("chat", 0.8))
                .build(),
            "hosted")
        .policy(RoutingPolicy.CHEAPEST)
        .build();
  }
}
