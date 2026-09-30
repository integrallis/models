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

import java.util.List;
import java.util.concurrent.atomic.AtomicInteger;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class RoutingBoundaryTest {
  private static ModelCandidate model(String id) {
    return ModelCandidate.builder(id).local(true).build();
  }

  @Test
  void localBoundaryAppliesBeforeClassification() {
    AtomicInteger calls = new AtomicInteger();
    ModelRouter router =
        ModelRouter.builder()
            .candidates(List.of(model("a")))
            .classifier(
                query -> {
                  calls.incrementAndGet();
                  return "chat";
                })
            .build();
    assertThatThrownBy(
            () ->
                router.route(
                    RoutingRequest.builder("private").build(),
                    RoutingRequirements.builder()
                        .dataBoundary(RoutingDataBoundary.LOCAL_ONLY)
                        .build()))
        .isInstanceOf(NoEligibleModelException.class);
    assertThat(calls).hasValue(0);
    assertThat(
            router
                .route(
                    RoutingRequest.builder("private").taskType("chat").build(),
                    RoutingRequirements.builder()
                        .dataBoundary(RoutingDataBoundary.LOCAL_ONLY)
                        .build())
                .selected()
                .id())
        .isEqualTo("a");
  }

  @Test
  void explicitlyLocalClassifierIsAllowed() {
    ModelRouter router =
        ModelRouter.builder()
            .candidates(List.of(model("a")))
            .classifier(TaskClassifier.local(query -> "chat"))
            .build();
    assertThat(
            router
                .route(
                    RoutingRequest.builder("private").build(),
                    RoutingRequirements.builder()
                        .dataBoundary(RoutingDataBoundary.LOCAL_ONLY)
                        .build())
                .taskType())
        .isEqualTo("chat");
  }

  @Test
  void stateLockRemovesAllOtherFallbacks() {
    ModelRouter router = ModelRouter.builder().candidates(List.of(model("a"), model("b"))).build();
    var decision =
        router.route(
            RoutingRequest.builder("continue").build(),
            RoutingContinuity.builder().activeToolLoop(true).ownerModelId("a").build());
    assertThat(decision.selected().id()).isEqualTo("a");
    assertThat(decision.fallbacks()).isEmpty();
  }

  @Test
  void stateWithoutOwnerFailsClosed() {
    ModelRouter router = ModelRouter.builder().candidates(List.of(model("a"), model("b"))).build();
    assertThatThrownBy(
            () ->
                router.route(
                    RoutingRequest.builder("continue").build(),
                    RoutingContinuity.builder().contextPortable(false).build()))
        .isInstanceOf(NoEligibleModelException.class);
  }

  @Test
  void lockedOwnerCannotEvadeNewRequirements() {
    ModelRouter router =
        ModelRouter.builder()
            .candidates(
                List.of(
                    model("a"),
                    ModelCandidate.builder("b")
                        .local(true)
                        .capabilities(java.util.Set.of("vision"))
                        .build()))
            .build();
    assertThatThrownBy(
            () ->
                router.route(
                    RoutingRequest.builder("continue").build(),
                    RoutingContinuity.builder().contextPortable(false).ownerModelId("a").build(),
                    RoutingRequirements.builder().requireCapability("vision").build()))
        .isInstanceOf(NoEligibleModelException.class);
  }

  @Test
  void unavailableStateOwnerCannotFailOver() {
    ModelRouter router = ModelRouter.builder().candidates(List.of(model("a"), model("b"))).build();
    router.updateRuntimeState("a", ModelRuntimeState.builder().available(false).build());
    assertThatThrownBy(
            () ->
                router.route(
                    RoutingRequest.builder("continue").build(),
                    RoutingContinuity.builder().activeToolLoop(true).ownerModelId("a").build()))
        .isInstanceOf(NoEligibleModelException.class);
  }

  @Test
  void explicitOwnerCannotContradictLiveSession() {
    ModelRouter router = ModelRouter.builder().candidates(List.of(model("a"), model("b"))).build();
    var request = RoutingRequest.builder("continue").sessionId("session").build();
    assertThat(router.route(request).selected().id()).isEqualTo("a");
    assertThatThrownBy(
            () ->
                router.route(
                    request,
                    RoutingContinuity.builder().contextPortable(false).ownerModelId("b").build()))
        .isInstanceOf(NoEligibleModelException.class);
  }
}
