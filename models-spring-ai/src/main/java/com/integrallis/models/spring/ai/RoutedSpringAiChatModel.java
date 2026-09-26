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
import com.integrallis.models.router.RoutingBudgetExceededException;
import com.integrallis.models.router.RoutingCancellation;
import com.integrallis.models.router.RoutingContinuity;
import com.integrallis.models.router.RoutingDecision;
import com.integrallis.models.router.RoutingExecution;
import com.integrallis.models.router.RoutingExecutionOptions;
import com.integrallis.models.router.RoutingFeedback;
import com.integrallis.models.router.RoutingRequest;
import com.integrallis.models.router.RoutingRequirements;
import com.integrallis.models.router.RoutingUsage;
import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.concurrent.CancellationException;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import java.util.function.Function;
import org.springframework.ai.chat.messages.Message;
import org.springframework.ai.chat.messages.ToolResponseMessage;
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
  private final Function<Prompt, RoutingRequirements> requirementsFactory;
  private final Function<Prompt, RoutingExecutionOptions> executionFactory;

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
    this(fleet, requestFactory, continuityFactory, ignored -> RoutingRequirements.none());
  }

  /**
   * Routes with requirements extracted from the complete original request, including its history.
   *
   * <p>Requirements constrain the selected model and every fallback. The application supplies
   * capability and data-boundary policy; the adapter does not infer sensitive data. A null
   * requirement result fails before any client is invoked.
   */
  public RoutedSpringAiChatModel(
      ModelFleet<ChatModel> fleet,
      Function<Prompt, RoutingRequest> requestFactory,
      Function<Prompt, RoutingContinuity> continuityFactory,
      Function<Prompt, RoutingRequirements> requirementsFactory) {
    this(
        fleet,
        requestFactory,
        continuityFactory,
        requirementsFactory,
        ignored -> RoutingExecutionOptions.unlimited());
  }

  /** Routes with complete token bounds, shared spending controls and an end-to-end deadline. */
  public RoutedSpringAiChatModel(
      ModelFleet<ChatModel> fleet,
      Function<Prompt, RoutingRequest> requestFactory,
      Function<Prompt, RoutingContinuity> continuityFactory,
      Function<Prompt, RoutingRequirements> requirementsFactory,
      Function<Prompt, RoutingExecutionOptions> executionFactory) {
    this.executionFactory = Objects.requireNonNull(executionFactory, "executionFactory");
    this.fleet = Objects.requireNonNull(fleet, "fleet");
    this.requestFactory = Objects.requireNonNull(requestFactory, "requestFactory");
    this.continuityFactory = Objects.requireNonNull(continuityFactory, "continuityFactory");
    this.requirementsFactory = Objects.requireNonNull(requirementsFactory, "requirementsFactory");
  }

  @Override
  public ChatResponse call(Prompt prompt) {
    Objects.requireNonNull(prompt, "prompt");
    RoutingExecutionOptions options = executionOptions(prompt);
    RoutingRequest request = routeRequest(prompt);
    RoutingContinuity continuity = routeContinuity(prompt);
    return fleet
        .execute(
            request,
            routeRequirements(prompt),
            continuity,
            options,
            model -> model.call(prompt),
            RoutedSpringAiChatModel::usage)
        .value();
  }

  @Override
  public Flux<ChatResponse> stream(Prompt prompt) {
    Objects.requireNonNull(prompt, "prompt");
    return Flux.defer(
        () -> {
          if (Thread.currentThread().isInterrupted())
            return Flux.error(new CancellationException("routed stream interrupted"));
          RoutingExecution execution = new RoutingExecution(executionOptions(prompt));
          // Subscribe cancellation before classification or provider setup. The same absolute timer
          // stays active across every chunk and fallback; it is not an idle timeout.
          reactor.core.publisher.Mono<ChatResponse> cancellation =
              reactor.core.publisher.Mono.create(
                  sink -> {
                    AutoCloseable registration =
                        execution.onCancel(
                            () ->
                                sink.error(
                                    new CancellationException(
                                        "routed stream cancelled or deadline exceeded")));
                    sink.onDispose(
                        () -> {
                          try {
                            registration.close();
                          } catch (Exception ignored) {
                          }
                        });
                  });
          return Flux.defer(
                  () -> {
                    execution.checkActive();
                    RoutingRequest request = routeRequest(prompt);
                    RoutingDecision decision =
                        fleet.decide(
                            request, routeRequirements(prompt), routeContinuity(prompt), execution);
                    List<ModelCandidate> order = new ArrayList<>();
                    order.add(decision.selected());
                    order.addAll(decision.fallbacks());
                    return streamAttempt(prompt, request, decision.taskType(), order, 0, execution);
                  })
              .takeUntilOther(cancellation)
              .doFinally(signal -> execution.close());
        });
  }

  private Flux<ChatResponse> streamAttempt(
      Prompt prompt,
      RoutingRequest request,
      String taskType,
      List<ModelCandidate> order,
      int index,
      RoutingExecution execution) {
    return Flux.defer(
        () -> {
          execution.checkActive();
          ModelCandidate candidate = order.get(index);
          RoutingExecution.Attempt reservation;
          try {
            reservation = execution.beginAttempt(candidate);
          } catch (RoutingBudgetExceededException denied) {
            return index + 1 < order.size()
                ? streamAttempt(prompt, request, taskType, order, index + 1, execution)
                : Flux.error(denied);
          }
          AtomicBoolean emitted = new AtomicBoolean();
          AtomicLong firstTokenNanos = new AtomicLong(-1);
          java.util.concurrent.atomic.AtomicReference<RoutingUsage> lastUsage =
              new java.util.concurrent.atomic.AtomicReference<>();
          java.util.concurrent.atomic.AtomicReference<AutoCloseable> cancelRegistration =
              new java.util.concurrent.atomic.AtomicReference<>();
          long started = System.nanoTime();
          return Flux.defer(
                  () -> {
                    execution.checkActive();
                    return fleet.model(candidate.id()).client().stream(prompt);
                  })
              .doOnSubscribe(
                  subscription -> cancelRegistration.set(execution.onCancel(subscription::cancel)))
              .doOnNext(
                  response -> {
                    execution.checkActive();
                    emitted.set(true);
                    firstTokenNanos.compareAndSet(-1, System.nanoTime());
                    // Spring providers expose cumulative usage on the final response; never sum
                    // cumulative chunks.
                    lastUsage.set(usage(response));
                  })
              .doOnComplete(
                  () -> {
                    execution.checkActive();
                    reservation.settle(lastUsage.get());
                    execution.checkActive();
                    fleet
                        .router()
                        .record(
                            feedback(
                                request,
                                taskType,
                                candidate.id(),
                                true,
                                firstTokenNanos.get(),
                                started));
                  })
              .onErrorResume(
                  failure -> {
                    reservation.close();
                    if (failure instanceof RoutingBudgetExceededException)
                      return Flux.error(failure);
                    execution.checkActive();
                    if (RoutingCancellation.isCancellation(failure)) return Flux.error(failure);
                    fleet
                        .router()
                        .record(feedback(request, taskType, candidate.id(), false, -1, started));
                    if (!emitted.get() && index + 1 < order.size())
                      return streamAttempt(prompt, request, taskType, order, index + 1, execution);
                    return Flux.error(failure);
                  })
              .doFinally(
                  signal -> {
                    reservation.close();
                    AutoCloseable registration = cancelRegistration.get();
                    if (registration != null) {
                      try {
                        registration.close();
                      } catch (Exception ignored) {
                      }
                    }
                  });
        });
  }

  private RoutingExecutionOptions executionOptions(Prompt prompt) {
    RoutingExecutionOptions options =
        Objects.requireNonNull(executionFactory.apply(prompt), "routing execution options");
    if (options.budgeted()
        && (prompt.getOptions() == null
            || prompt.getOptions().getMaxTokens() == null
            || prompt.getOptions().getMaxTokens() < 0
            || prompt.getOptions().getMaxTokens() > options.tokenBounds().outputTokens())) {
      throw new IllegalArgumentException(
          "budgeted prompts require an explicit provider maxTokens within the declared output bound");
    }
    if (options.budgeted()
        && prompt.getOptions()
            instanceof org.springframework.ai.model.tool.ToolCallingChatOptions tools
        && internalToolExecutionEnabled(tools)) {
      throw new IllegalArgumentException(
          "budgeted tool calls require internal tool execution disabled; route each tool turn explicitly");
    }
    return options;
  }

  private static boolean internalToolExecutionEnabled(
      org.springframework.ai.model.tool.ToolCallingChatOptions tools) {
    try {
      // Spring AI 1.x can execute extra provider calls internally. 2.0 removed this option.
      var accessor =
          org.springframework.ai.model.tool.ToolCallingChatOptions.class.getMethod(
              "getInternalToolExecutionEnabled");
      return !Boolean.FALSE.equals(accessor.invoke(tools));
    } catch (NoSuchMethodException absentInSpringAi2) {
      return false;
    } catch (ReflectiveOperationException failure) {
      throw new IllegalArgumentException("cannot verify tool execution policy", failure);
    }
  }

  private static RoutingUsage usage(ChatResponse response) {
    if (response == null
        || response.getMetadata() == null
        || response.getMetadata().getUsage() == null) return null;
    var usage = response.getMetadata().getUsage();
    // EmptyUsage reports zeroes even when no provider usage was supplied.
    if (usage.getClass().getSimpleName().equals("EmptyUsage")
        || (Integer.valueOf(0).equals(usage.getPromptTokens())
            && Integer.valueOf(0).equals(usage.getCompletionTokens()))
        || usage.getPromptTokens() == null
        || usage.getCompletionTokens() == null) return null;
    return new RoutingUsage(usage.getPromptTokens(), usage.getCompletionTokens());
  }

  private RoutingRequest routeRequest(Prompt prompt) {
    return Objects.requireNonNull(requestFactory.apply(prompt), "routing request");
  }

  private RoutingRequirements routeRequirements(Prompt prompt) {
    return Objects.requireNonNull(requirementsFactory.apply(prompt), "routing requirements");
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
    return RoutingRequests.estimate(prompt);
  }

  private static RoutingContinuity defaultContinuity(Prompt prompt) {
    List<Message> messages = prompt.getInstructions();
    boolean toolLoop = !messages.isEmpty() && messages.getLast() instanceof ToolResponseMessage;
    return RoutingContinuity.builder().activeToolLoop(toolLoop).build();
  }
}
