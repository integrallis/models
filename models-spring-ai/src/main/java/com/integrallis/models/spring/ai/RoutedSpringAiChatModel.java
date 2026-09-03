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

import com.integrallis.models.router.ModelCandidate;
import com.integrallis.models.router.ModelFleet;
import com.integrallis.models.router.RoutingContinuity;
import com.integrallis.models.router.RoutingDecision;
import com.integrallis.models.router.RoutingFeedback;
import com.integrallis.models.router.RoutingRequest;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.ToolResponseMessage;
import org.springframework.ai.chat.messages.UserMessage;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.model.ChatResponse;
import org.springframework.ai.chat.prompt.Prompt;
import reactor.core.publisher.Flux;

/**
 * A Spring AI {@link ChatModel} that routes unchanged prompts across local and hosted clients.
 *
 * <p>Every bound client is an ordinary Spring AI model, so provider tools, options, response
 * metadata, and local Models behavior stay with the delegate that owns them.
 */
public final class RoutedSpringAiChatModel implements ChatModel {
  private final ModelFleet<ChatModel> fleet;
  private final Function<Prompt, RoutingRequest> requestFactory;
  private final Function<Prompt, RoutingContinuity> continuityFactory;

  /** Routes from the latest user message, without implicit session affinity. */
  public RoutedSpringAiChatModel(ModelFleet<ChatModel> fleet) {
    this(
        fleet, RoutedSpringAiChatModel::defaultRequest, RoutedSpringAiChatModel::defaultContinuity);
  }

  /** Routes with an application-supplied mapping for token estimates, task, and session id. */
  public RoutedSpringAiChatModel(
      ModelFleet<ChatModel> fleet, Function<Prompt, RoutingRequest> requestFactory) {
    this(fleet, requestFactory, RoutedSpringAiChatModel::defaultContinuity);
  }

  /** Routes with application-supplied request and continuity extraction. */
  public RoutedSpringAiChatModel(
      ModelFleet<ChatModel> fleet,
      Function<Prompt, RoutingRequest> requestFactory,
      Function<Prompt, RoutingContinuity> continuityFactory) {
    this.fleet = Objects.requireNonNull(fleet, "fleet");
    this.requestFactory = Objects.requireNonNull(requestFactory, "requestFactory");
    this.continuityFactory = Objects.requireNonNull(continuityFactory, "continuityFactory");
  }

  @Override
  public ChatResponse call(Prompt prompt) {
    Objects.requireNonNull(prompt, "prompt");
    RoutingRequest request = routeRequest(prompt);
    RoutingContinuity continuity = routeContinuity(prompt);
    return fleet.execute(request, continuity, model -> model.call(prompt)).value();
  }

  @Override
  public Flux<ChatResponse> stream(Prompt prompt) {
    Objects.requireNonNull(prompt, "prompt");
    return Flux.defer(
        () -> {
          RoutingRequest request = routeRequest(prompt);
          RoutingDecision decision = fleet.decide(request, routeContinuity(prompt));
          List<ModelCandidate> order = new ArrayList<>();
          order.add(decision.selected());
          order.addAll(decision.fallbacks());
          return streamAttempt(prompt, request, decision.taskType(), order, 0);
        });
  }

  private Flux<ChatResponse> streamAttempt(
      Prompt prompt,
      RoutingRequest request,
      String taskType,
      List<ModelCandidate> order,
      int index) {
    ModelCandidate candidate = order.get(index);
    AtomicBoolean emitted = new AtomicBoolean();
    AtomicLong firstTokenNanos = new AtomicLong(-1);
    long started = System.nanoTime();
    return Flux.defer(() -> fleet.model(candidate.id()).client().stream(prompt))
        .doOnNext(
            ignored -> {
              emitted.set(true);
              firstTokenNanos.compareAndSet(-1, System.nanoTime());
            })
        .doOnComplete(
            () ->
                fleet
                    .router()
                    .record(
                        feedback(
                            request,
                            taskType,
                            candidate.id(),
                            true,
                            firstTokenNanos.get(),
                            started)))
        .onErrorResume(
            failure -> {
              fleet
                  .router()
                  .record(feedback(request, taskType, candidate.id(), false, -1, started));
              if (!emitted.get() && index + 1 < order.size()) {
                return streamAttempt(prompt, request, taskType, order, index + 1);
              }
              return Flux.error(failure);
            });
  }

  private RoutingRequest routeRequest(Prompt prompt) {
    return Objects.requireNonNull(requestFactory.apply(prompt), "routing request");
  }

  private RoutingContinuity routeContinuity(Prompt prompt) {
    return Objects.requireNonNull(continuityFactory.apply(prompt), "routing continuity");
  }

  private static RoutingFeedback feedback(
      RoutingRequest request,
      String taskType,
      String modelId,
      boolean success,
      long firstTokenNanos,
      long started) {
    RoutingFeedback.Builder feedback =
        success ? RoutingFeedback.success(modelId) : RoutingFeedback.failure(modelId);
    request.session().ifPresent(feedback::sessionId);
    if (taskType != null) {
      feedback.taskType(taskType);
    }
    if (success && firstTokenNanos >= started) {
      feedback.timeToFirstTokenMillis((firstTokenNanos - started) / 1_000_000);
    }
    return feedback.build();
  }

  private static RoutingRequest defaultRequest(Prompt prompt) {
    List<Message> messages = prompt.getInstructions();
    for (int index = messages.size() - 1; index >= 0; index--) {
      if (messages.get(index) instanceof UserMessage user) {
        return RoutingRequest.builder(user.getText()).build();
      }
    }
    return RoutingRequest.builder(prompt.getContents()).build();
  }

  private static RoutingContinuity defaultContinuity(Prompt prompt) {
    List<Message> messages = prompt.getInstructions();
    boolean toolLoop = !messages.isEmpty() && messages.getLast() instanceof ToolResponseMessage;
    return RoutingContinuity.builder().activeToolLoop(toolLoop).build();
  }
}
