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
package com.integrallis.models.tinystruct;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.integrallis.models.router.ModelCandidate;
import com.integrallis.models.router.ModelFleet;
import com.integrallis.models.router.RoutingRequest;
import com.integrallis.models.router.RoutingUsage;
import java.util.Map;
import java.util.Set;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.tinystruct.ApplicationException;
import org.tinystruct.data.component.Builder;
import org.tinystruct.typesafe.client.RoutingResult;
import org.tinystruct.typesafe.client.TypesafeClient;

@Tag("unit")
class RoutedTypesafeClientTest {

  @Test
  void routesToPrimaryClientAndFallsBackWhenPrimaryThrowsRuntimeFailure() throws Exception {
    AtomicInteger primaryCalls = new AtomicInteger();
    AtomicInteger fallbackCalls = new AtomicInteger();

    TypesafeClient failingPrimary =
        request -> {
          primaryCalls.incrementAndGet();
          throw new IllegalStateException("local backend overloaded");
        };

    TypesafeClient backupClient =
        request -> {
          fallbackCalls.incrementAndGet();
          Builder answers = new Builder();
          Builder choice = new Builder();
          choice.put("choice", "create-user");
          choice.put("confidence", 0.93);
          answers.put("__tool__", choice);
          Builder usage = new Builder();
          usage.put("input_tokens", 18);
          usage.put("output_tokens", 2);
          return new RoutingResult(answers, "backup-jev", usage);
        };

    ModelFleet<TypesafeClient> fleet =
        ModelFleet.<TypesafeClient>builder()
            .model(
                ModelCandidate.builder("local-decisions")
                    .local(true)
                    .capabilities(Set.of("classification"))
                    .costPerMillionTokens(0.0, 0.0)
                    .timeToFirstTokenMillis(50)
                    .tokensPerSecond(30)
                    .contextWindow(4096)
                    .quality(Map.of("classification", 0.95))
                    .build(),
                failingPrimary)
            .model(
                ModelCandidate.builder("backup-decisions")
                    .local(false)
                    .capabilities(Set.of("classification"))
                    .costPerMillionTokens(0.10, 0.10)
                    .timeToFirstTokenMillis(100)
                    .tokensPerSecond(20)
                    .contextWindow(8192)
                    .quality(Map.of("classification", 0.90))
                    .build(),
                backupClient)
            .build();

    RoutedTypesafeClient routed = new RoutedTypesafeClient(fleet);
    Builder questions = new Builder();
    Builder q = new Builder();
    q.put("type", "noul");
    q.put("instructions", "Is it valid?");
    questions.put("__tool__", q);

    RoutingResult result =
        routed.classify(
            new org.tinystruct.typesafe.client.RoutingRequest("create user", questions, "jev"));

    assertThat(primaryCalls.get()).isEqualTo(1);
    assertThat(fallbackCalls.get()).isEqualTo(1);
    assertThat(result.getChoice("__tool__")).isEqualTo("create-user");
    assertThat(result.getModelVersion()).isEqualTo("backup-jev");
  }

  @Test
  void unwrapsApplicationExceptionWhenAllCandidatesFailWithApplicationException() {
    TypesafeClient failingClient =
        request -> {
          throw new ApplicationException("TypeSafe authentication failed (401)");
        };

    ModelFleet<TypesafeClient> fleet =
        ModelFleet.<TypesafeClient>builder()
            .model(
                ModelCandidate.builder("only-candidate")
                    .local(true)
                    .capabilities(Set.of("classification"))
                    .costPerMillionTokens(0.0, 0.0)
                    .timeToFirstTokenMillis(50)
                    .tokensPerSecond(30)
                    .contextWindow(4096)
                    .quality(Map.of("classification", 0.90))
                    .build(),
                failingClient)
            .build();

    RoutedTypesafeClient routed =
        new RoutedTypesafeClient(
            fleet,
            request ->
                RoutingRequest.builder("test")
                    .taskType("classification")
                    .estimatedTokens(8)
                    .build());

    Builder questions = new Builder();
    questions.put("q", new Builder());

    assertThatThrownBy(
            () ->
                routed.classify(
                    new org.tinystruct.typesafe.client.RoutingRequest("hi", questions, "m")))
        .isInstanceOf(ApplicationException.class)
        .hasMessageContaining("401");
  }

  @Test
  void extractsUsageFromNumericAndStringCounters() {
    assertThat(RoutedTypesafeClient.extractUsage(null)).isNull();
    assertThat(
            RoutedTypesafeClient.extractUsage(new RoutingResult(new Builder(), "m", new Builder())))
        .isNull();

    Builder zeroUsage = new Builder();
    zeroUsage.put("input_tokens", 0);
    zeroUsage.put("output_tokens", 0);
    assertThat(RoutedTypesafeClient.extractUsage(new RoutingResult(new Builder(), "m", zeroUsage)))
        .isNull();

    Builder stringUsage = new Builder();
    stringUsage.put("prompt_tokens", "24");
    stringUsage.put("completion_tokens", "6");
    RoutingUsage extracted =
        RoutedTypesafeClient.extractUsage(new RoutingResult(new Builder(), "m", stringUsage));
    assertThat(extracted).isEqualTo(new RoutingUsage(24, 6));

    Builder invalidUsage = new Builder();
    invalidUsage.put("prompt_tokens", "not-a-number");
    invalidUsage.put("completion_tokens", "4");
    assertThat(
            RoutedTypesafeClient.extractUsage(new RoutingResult(new Builder(), "m", invalidUsage)))
        .isNull();
  }
}
