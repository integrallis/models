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
package com.integrallis.models.langchain4j;

import com.integrallis.models.router.ModelCandidate;
import com.integrallis.models.router.ModelFleet;
import com.integrallis.models.router.RoutingContinuity;
import com.integrallis.models.router.RoutingDecision;
import com.integrallis.models.router.RoutingFeedback;
import com.integrallis.models.router.RoutingRequest;
import dev.langchain4j.data.message.ChatMessage;
import dev.langchain4j.data.message.ToolExecutionResultMessage;
import dev.langchain4j.data.message.UserMessage;
import dev.langchain4j.model.chat.StreamingChatModel;
import dev.langchain4j.model.chat.request.ChatRequest;
import dev.langchain4j.model.chat.response.ChatResponse;
import dev.langchain4j.model.chat.response.StreamingChatResponseHandler;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;

/** LangChain4j streaming model that routes unchanged requests across local and hosted clients. */
public final class RoutedStreamingChatModel implements StreamingChatModel {
  private final ModelFleet<StreamingChatModel> fleet;
  private final Function<ChatRequest, RoutingRequest> requestFactory;
  private final Function<ChatRequest, RoutingContinuity> continuityFactory;

  /** Routes from the latest user message, without implicit session affinity. */
  public RoutedStreamingChatModel(ModelFleet<StreamingChatModel> fleet) {
    this(
        fleet,
        RoutedStreamingChatModel::defaultRequest,
        RoutedStreamingChatModel::defaultContinuity);
  }

  /** Routes with an application-supplied mapping for token estimates, task, and session id. */
  public RoutedStreamingChatModel(
      ModelFleet<StreamingChatModel> fleet, Function<ChatRequest, RoutingRequest> requestFactory) {
    this(fleet, requestFactory, RoutedStreamingChatModel::defaultContinuity);
  }

  /** Routes with application-supplied request and continuity extraction. */
  public RoutedStreamingChatModel(
      ModelFleet<StreamingChatModel> fleet,
      Function<ChatRequest, RoutingRequest> requestFactory,
      Function<ChatRequest, RoutingContinuity> continuityFactory) {
    this.fleet = Objects.requireNonNull(fleet, "fleet");
    this.requestFactory = Objects.requireNonNull(requestFactory, "requestFactory");
    this.continuityFactory = Objects.requireNonNull(continuityFactory, "continuityFactory");
  }

  @Override
  public void doChat(ChatRequest request, StreamingChatResponseHandler handler) {
    Objects.requireNonNull(request, "request");
    Objects.requireNonNull(handler, "handler");
    RoutingRequest routingRequest =
        Objects.requireNonNull(requestFactory.apply(request), "routing request");
    RoutingContinuity continuity =
        Objects.requireNonNull(continuityFactory.apply(request), "routing continuity");
    RoutingDecision decision = fleet.decide(routingRequest, continuity);
    List<ModelCandidate> order = new ArrayList<>();
    order.add(decision.selected());
    order.addAll(decision.fallbacks());
    streamAttempt(
        request, routingRequest, decision.taskType(), order, 0, handler, new AtomicBoolean());
  }

  private void streamAttempt(
      ChatRequest request,
      RoutingRequest routingRequest,
      String taskType,
      List<ModelCandidate> order,
      int index,
      StreamingChatResponseHandler handler,
      AtomicBoolean terminalDelivered) {
    ModelCandidate candidate = order.get(index);
    AtomicBoolean emitted = new AtomicBoolean();
    AtomicBoolean attemptFinished = new AtomicBoolean();
    AtomicLong firstTokenNanos = new AtomicLong(-1);
    long started = System.nanoTime();
    StreamingChatResponseHandler routedHandler =
        new StreamingChatResponseHandler() {
          @Override
          public void onPartialResponse(String partialResponse) {
            if (!attemptFinished.get() && !terminalDelivered.get()) {
              emitted.set(true);
              firstTokenNanos.compareAndSet(-1, System.nanoTime());
              handler.onPartialResponse(partialResponse);
            }
          }

          @Override
          public void onCompleteResponse(ChatResponse response) {
            if (attemptFinished.compareAndSet(false, true)
                && terminalDelivered.compareAndSet(false, true)) {
              record(
                  routingRequest, taskType, candidate.id(), true, firstTokenNanos.get(), started);
              handler.onCompleteResponse(response);
            }
          }

          @Override
          public void onError(Throwable failure) {
            if (attemptFinished.compareAndSet(false, true)) {
              failed(
                  request,
                  routingRequest,
                  taskType,
                  order,
                  index,
                  handler,
                  terminalDelivered,
                  emitted.get(),
                  failure,
                  started);
            }
          }
        };

    try {
      fleet.model(candidate.id()).client().doChat(request, routedHandler);
    } catch (RuntimeException | Error failure) {
      if (attemptFinished.compareAndSet(false, true)) {
        failed(
            request,
            routingRequest,
            taskType,
            order,
            index,
            handler,
            terminalDelivered,
            emitted.get(),
            failure,
            started);
      }
    }
  }

  private void failed(
      ChatRequest request,
      RoutingRequest routingRequest,
      String taskType,
      List<ModelCandidate> order,
      int index,
      StreamingChatResponseHandler handler,
      AtomicBoolean terminalDelivered,
      boolean emitted,
      Throwable failure,
      long started) {
    ModelCandidate candidate = order.get(index);
    record(routingRequest, taskType, candidate.id(), false, -1, started);
    if (!emitted && index + 1 < order.size() && !terminalDelivered.get()) {
      streamAttempt(
          request, routingRequest, taskType, order, index + 1, handler, terminalDelivered);
    } else if (terminalDelivered.compareAndSet(false, true)) {
      handler.onError(failure);
    }
  }

  private void record(
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
    fleet.router().record(feedback.build());
  }

  private static RoutingRequest defaultRequest(ChatRequest request) {
    List<ChatMessage> messages = request.messages();
    for (int index = messages.size() - 1; index >= 0; index--) {
      if (messages.get(index) instanceof UserMessage user) {
        return RoutingRequest.builder(user.singleText()).build();
      }
    }
    return RoutingRequest.builder("").build();
  }

  private static RoutingContinuity defaultContinuity(ChatRequest request) {
    List<ChatMessage> messages = request.messages();
    boolean toolLoop =
        !messages.isEmpty() && messages.getLast() instanceof ToolExecutionResultMessage;
    return RoutingContinuity.builder().activeToolLoop(toolLoop).build();
  }
}
