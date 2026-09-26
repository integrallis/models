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

import java.util.concurrent.*;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.ValueSource;

@Tag("unit")
class RoutedCancellationTest {
  @ParameterizedTest
  @ValueSource(booleans = {false, true})
  void wrappedCancellationStopsFallbackAndPreservesInterruption(boolean interrupted) {
    Throwable cause =
        interrupted
            ? new InterruptedException("cancelled")
            : new CancellationException("cancelled");
    RuntimeException failure =
        new CompletionException(new IllegalStateException("SDK wrapper", cause));
    AtomicInteger calls = new AtomicInteger();
    var fleet =
        ModelFleet.<String>builder()
            .model(
                ModelCandidate.builder("local").local(true).costPerMillionTokens(0, 0).build(),
                "local")
            .model(ModelCandidate.builder("hosted").costPerMillionTokens(1, 1).build(), "hosted")
            .policy(RoutingPolicy.CHEAPEST)
            .build();
    try {
      assertThatThrownBy(
              () ->
                  fleet.execute(
                      RoutingRequest.builder("question").build(),
                      client -> {
                        if (client.equals("local")) throw failure;
                        calls.incrementAndGet();
                        return "hosted";
                      }))
          .isInstanceOf(CancellationException.class)
          .hasCause(failure);
      assertThat(calls).hasValue(0);
      assertThat(fleet.router().status("local").orElseThrow().consecutiveFailures()).isZero();
      assertThat(Thread.currentThread().isInterrupted()).isEqualTo(interrupted);
    } finally {
      Thread.interrupted();
    }
  }
}
