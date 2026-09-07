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

import com.integrallis.models.runtime.chat.ChatMessage;
import com.integrallis.models.runtime.chat.ChatRole;
import com.integrallis.models.runtime.chat.VirtualChatModel;
import java.util.List;
import java.util.Objects;
import java.util.function.Function;

/** Adapts the provider-neutral {@link ModelRouter} to an in-process virtual chat session. */
public final class VirtualChatRouter implements VirtualChatModel.Selector {
  private final ModelRouter router;
  private final Function<VirtualChatModel.Turn, RoutingRequest> requestFactory;
  private final Function<VirtualChatModel.Turn, RoutingContinuity> continuityFactory;

  /** Routes by the turn's explicit task and real per-member prompt-cache evidence. */
  public VirtualChatRouter(ModelRouter router) {
    this(router, VirtualChatRouter::defaultRequest, VirtualChatRouter::defaultContinuity);
  }

  /** Routes with application-supplied request and continuity mappings. */
  public VirtualChatRouter(
      ModelRouter router,
      Function<VirtualChatModel.Turn, RoutingRequest> requestFactory,
      Function<VirtualChatModel.Turn, RoutingContinuity> continuityFactory) {
    this.router = Objects.requireNonNull(router, "router");
    this.requestFactory = Objects.requireNonNull(requestFactory, "requestFactory");
    this.continuityFactory = Objects.requireNonNull(continuityFactory, "continuityFactory");
  }

  @Override
  public VirtualChatModel.Selection select(VirtualChatModel.Turn turn) {
    Objects.requireNonNull(turn, "turn");
    RoutingRequest request = Objects.requireNonNull(requestFactory.apply(turn), "routing request");
    RoutingContinuity continuity =
        Objects.requireNonNull(continuityFactory.apply(turn), "routing continuity");
    RoutingDecision decision = router.route(request, continuity);
    return new VirtualChatModel.Selection(
        decision.modelId(),
        decision.explain(),
        Objects.requireNonNullElse(decision.taskType(), ""));
  }

  @Override
  public void recordSuccess(VirtualChatModel.Turn turn, VirtualChatModel.Response response) {
    RoutingFeedback.Builder feedback =
        RoutingFeedback.success(response.memberId())
            .sessionId(turn.sessionId())
            .tokenUsage(
                response.metrics().usage().promptTokens(),
                response.metrics().promptCache().cacheReadInputTokens());
    if (!response.taskType().isBlank()) {
      feedback.taskType(response.taskType());
    }
    response
        .metrics()
        .timeToFirstToken()
        .ifPresent(duration -> feedback.timeToFirstTokenMillis(duration.toMillis()));
    double throughput = response.metrics().decodeTokensPerSecond();
    if (throughput > 0) {
      feedback.tokensPerSecond(throughput);
    }
    router.record(feedback.build());
  }

  @Override
  public void recordFailure(
      VirtualChatModel.Turn turn, VirtualChatModel.Selection selection, Throwable failure) {
    RoutingFeedback.Builder feedback =
        RoutingFeedback.failure(selection.memberId()).sessionId(turn.sessionId());
    String taskType = selection.taskType().isBlank() ? turn.taskType() : selection.taskType();
    if (!taskType.isBlank()) {
      feedback.taskType(taskType);
    }
    router.record(feedback.build());
  }

  @Override
  public void closeSession(String sessionId) {
    router.closeSession(sessionId);
  }

  private static RoutingRequest defaultRequest(VirtualChatModel.Turn turn) {
    RoutingRequest.Builder request =
        RoutingRequest.builder(latestUserText(turn.history())).sessionId(turn.sessionId());
    if (!turn.taskType().isBlank()) {
      request.taskType(turn.taskType());
    }
    return request.build();
  }

  private static RoutingContinuity defaultContinuity(VirtualChatModel.Turn turn) {
    RoutingContinuity.Builder continuity =
        RoutingContinuity.builder()
            .contextPortable(true)
            .activeToolLoop(turn.input().role() == ChatRole.TOOL);
    turn.members()
        .forEach((id, state) -> continuity.cachedPrefixTokens(id, state.cachedPromptTokens()));
    return continuity.build();
  }

  private static String latestUserText(List<ChatMessage> history) {
    for (int index = history.size() - 1; index >= 0; index--) {
      ChatMessage message = history.get(index);
      if (message.role() == ChatRole.USER) {
        return message.text();
      }
    }
    return history.getLast().text();
  }
}
