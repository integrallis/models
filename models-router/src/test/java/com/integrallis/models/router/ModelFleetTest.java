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

import java.util.Map;
import java.util.Optional;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ModelFleetTest {

  @Test
  void bindsLocalAndExternalClientsWithoutAProviderDependency() {
    Client local = prompt -> "local: " + prompt;
    Client hosted = prompt -> "hosted: " + prompt;
    ModelFleet<Client> fleet =
        ModelFleet.<Client>builder()
            .model(candidate("local", true, 0.6, 0), local)
            .model(candidate("openai/gpt", false, 0.95, 20), hosted)
            .policy(RoutingPolicy.BEST_QUALITY)
            .build();

    RoutedModel<Client> selected = fleet.route(RoutingRequest.builder("hard question").build());

    assertThat(selected.candidate().id()).isEqualTo("openai/gpt");
    assertThat(selected.client()).isSameAs(hosted);
  }

  @Test
  void executesOrderedFallbackAndReportsTheFailedAttempt() {
    Client failingLocal =
        prompt -> {
          throw new IllegalStateException("runtime is busy");
        };
    Client hosted = prompt -> "hosted answer";
    ModelFleet<Client> fleet =
        ModelFleet.<Client>builder()
            .model(candidate("local", true, 0.9, 0), failingLocal)
            .model(candidate("hosted", false, 0.8, 1), hosted)
            .policy(RoutingPolicy.CHEAPEST)
            .build();

    RoutedResult<String> result =
        fleet.execute(
            RoutingRequest.builder("hello").sessionId("fallback-session").build(),
            client -> client.call("hello"));

    assertThat(result.value()).isEqualTo("hosted answer");
    assertThat(result.decision().selected().id()).isEqualTo("hosted");
    assertThat(result.attempts())
        .extracting(RoutingAttempt::modelId)
        .containsExactly("local", "hosted");
    assertThat(fleet.router().status("local").orElseThrow().consecutiveFailures()).isOne();
  }

  @Test
  void canBindClientsToAnAlreadyConfiguredRouter() {
    ModelCandidate local = candidate("local", true, 0.6, 0);
    ModelCandidate hosted = candidate("hosted", false, 0.9, 2);
    ModelRouter router =
        ModelRouter.builder()
            .candidates(java.util.List.of(local, hosted))
            .policy(RoutingPolicy.BEST_QUALITY)
            .build();

    ModelFleet<Client> fleet =
        ModelFleet.<Client>bind(router)
            .clients(
                Map.of(
                    "local", (Client) prompt -> "local",
                    "hosted", (Client) prompt -> "hosted"))
            .build();

    assertThat(fleet.route(RoutingRequest.builder("hello").build()).client().call("ignored"))
        .isEqualTo("hosted");
  }

  @Test
  void exposesRouterFiltersAndScorersOnTheFleetBuilder() {
    Client local = prompt -> "local";
    Client hosted = prompt -> "hosted";
    ModelFleet<Client> fleet =
        ModelFleet.<Client>builder()
            .model(candidate("local", true, 0.9, 0), local)
            .model(candidate("hosted", false, 0.5, 10), hosted)
            .filter(
                evaluation ->
                    evaluation.candidate().local()
                        ? Optional.of("tenant requires hosted processing")
                        : Optional.empty())
            .scorer("same-region", 2.0, evaluation -> 1.0)
            .build();

    assertThat(fleet.decide(RoutingRequest.builder("hello").build()).selected().id())
        .isEqualTo("hosted");
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

  @FunctionalInterface
  private interface Client {
    String call(String prompt);
  }
}
