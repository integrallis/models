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

import java.time.Duration;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.CancellationException;
import java.util.function.Function;

/**
 * Provider-neutral bindings between routing candidates and callable application clients.
 *
 * <p>The client type is chosen by the application. One fleet can therefore contain an in-process
 * Models adapter, a hosted Spring AI model, or private gateway clients without introducing any of
 * their SDKs into {@code models-router}.
 */
public final class ModelFleet<T> {
  private final ModelRouter router;
  private final Map<String, T> clients;

  private ModelFleet(ModelRouter router, Map<String, T> clients) {
    this.router = router;
    this.clients = Map.copyOf(clients);
    for (ModelCandidate candidate : router.candidates()) {
      if (!clients.containsKey(candidate.id())) {
        throw new IllegalArgumentException("no client is bound for candidate " + candidate.id());
      }
    }
  }

  public static <T> Builder<T> builder() {
    return new Builder<>(null);
  }

  public static <T> Builder<T> bind(ModelRouter router) {
    return new Builder<>(Objects.requireNonNull(router, "router"));
  }

  public ModelRouter router() {
    return router;
  }

  public RoutedModel<T> route(RoutingRequest request) {
    return binding(decide(request).selected());
  }

  /** Routes with per-request capabilities and a data boundary. */
  public RoutedModel<T> route(RoutingRequest request, RoutingRequirements requirements) {
    return binding(decide(request, requirements).selected());
  }

  /** Returns the full explainable decision without invoking a client. */
  public RoutingDecision decide(RoutingRequest request) {
    return router.route(request);
  }

  /** Returns a decision constrained by the request's capabilities and data boundary. */
  public RoutingDecision decide(RoutingRequest request, RoutingRequirements requirements) {
    return router.route(request, requirements);
  }

  /** Returns a full decision that accounts for the current conversation boundary. */
  public RoutingDecision decide(RoutingRequest request, RoutingContinuity continuity) {
    return router.route(request, continuity);
  }

  /** Returns a decision constrained by requirements and conversation continuity together. */
  public RoutingDecision decide(
      RoutingRequest request, RoutingRequirements requirements, RoutingContinuity continuity) {
    return router.route(request, continuity, requirements);
  }

  public RoutedModel<T> route(
      RoutingRequest request, RoutingPolicy policy, RoutingContinuity continuity) {
    RoutingDecision decision = router.route(request, policy, continuity);
    return binding(decision.selected());
  }

  /** Returns the bound client and descriptor for a model id. */
  public RoutedModel<T> model(String modelId) {
    ModelCandidate candidate =
        router.candidates().stream()
            .filter(value -> value.id().equals(modelId))
            .findFirst()
            .orElseThrow(() -> new IllegalArgumentException("unknown candidate id: " + modelId));
    return binding(candidate);
  }

  /** Executes the winner and then its ranked fallbacks, recording each explicit outcome. */
  public <R> RoutedResult<R> execute(
      RoutingRequest request, ModelInvocation<? super T, ? extends R> invocation) {
    return execute(request, RoutingContinuity.none(), invocation);
  }

  /** Executes ordered fallbacks while enforcing per-request routing requirements. */
  public <R> RoutedResult<R> execute(
      RoutingRequest request,
      RoutingRequirements requirements,
      ModelInvocation<? super T, ? extends R> invocation) {
    return execute(request, requirements, RoutingContinuity.none(), invocation);
  }

  /** Executes with tool-loop, provider-state, and model-specific cache evidence. */
  public <R> RoutedResult<R> execute(
      RoutingRequest request,
      RoutingContinuity continuity,
      ModelInvocation<? super T, ? extends R> invocation) {
    return execute(request, RoutingRequirements.none(), continuity, invocation);
  }

  /**
   * Executes with continuity plus explicit per-request capability and data-boundary requirements.
   *
   * <p>Cancellation stops execution without fallback or a model-health penalty. Interruption is
   * propagated as {@link CancellationException}, preserving the thread's interrupt flag.
   */
  public <R> RoutedResult<R> execute(
      RoutingRequest request,
      RoutingRequirements requirements,
      RoutingContinuity continuity,
      ModelInvocation<? super T, ? extends R> invocation) {
    return execute(
        request,
        requirements,
        continuity,
        RoutingExecutionOptions.unlimited(),
        invocation,
        ignored -> null);
  }

  /** Executes with actual spending controls, a completion deadline and provider-reported usage. */
  public <R> RoutedResult<R> execute(
      RoutingRequest request,
      RoutingRequirements requirements,
      RoutingContinuity continuity,
      RoutingExecutionOptions options,
      ModelInvocation<? super T, ? extends R> invocation,
      Function<? super R, RoutingUsage> usage) {
    Objects.requireNonNull(invocation, "invocation");
    Objects.requireNonNull(usage, "usage");
    try (RoutingExecution execution = new RoutingExecution(options)) {
      try {
        return execution.call(
            () ->
                executeControlled(request, requirements, continuity, execution, invocation, usage));
      } catch (RuntimeException failure) {
        throw failure;
      } catch (Exception failure) {
        throw new IllegalStateException("routing execution failed", failure);
      }
    }
  }

  /** Admission shared by blocking and streaming adapters. */
  public RoutingDecision decide(
      RoutingRequest request,
      RoutingRequirements requirements,
      RoutingContinuity continuity,
      RoutingExecution execution) {
    Objects.requireNonNull(request, "request");
    Objects.requireNonNull(requirements, "requirements");
    Objects.requireNonNull(continuity, "continuity");
    execution.checkActive();
    if (execution.options().budgeted()
        && request.taskTypeOverride() == null
        && !router.classifierIsLocal()) {
      throw new IllegalArgumentException(
          "budgeted classification requires a declared local classifier or a task override");
    }
    RoutingDecision decision = router.route(execution.account(request), continuity, requirements);
    execution.checkActive();
    return decision;
  }

  private <R> RoutedResult<R> executeControlled(
      RoutingRequest request,
      RoutingRequirements requirements,
      RoutingContinuity continuity,
      RoutingExecution execution,
      ModelInvocation<? super T, ? extends R> invocation,
      Function<? super R, RoutingUsage> usage) {
    RoutingDecision initial = decide(request, requirements, continuity, execution);
    List<ModelCandidate> order = new ArrayList<>();
    order.add(initial.selected());
    order.addAll(initial.fallbacks());
    List<RoutingAttempt> attempts = new ArrayList<>();
    Throwable lastFailure = null;
    RoutingBudgetExceededException budgetFailure = null;
    for (int index = 0; index < order.size(); index++) {
      execution.checkActive();
      ModelCandidate candidate = order.get(index);
      RoutingExecution.Attempt reservation;
      try {
        reservation = execution.beginAttempt(candidate);
      } catch (RoutingBudgetExceededException denied) {
        budgetFailure = denied;
        continue;
      }
      long started = System.nanoTime();
      R value;
      try (reservation) {
        try {
          execution.checkActive();
          value = invocation.invoke(binding(candidate).client());
        } catch (Exception failure) {
          execution.checkActive();
          if (failure instanceof CancellationException cancellation) throw cancellation;
          if (RoutingCancellation.isInterruption(failure)) {
            Thread.currentThread().interrupt();
          }
          if (RoutingCancellation.isCancellation(failure)) {
            CancellationException cancellation =
                new CancellationException("routed invocation cancelled");
            cancellation.initCause(failure);
            throw cancellation;
          }
          lastFailure = failure;
          attempts.add(
              new RoutingAttempt(
                  candidate.id(),
                  elapsed(started),
                  failure.getClass().getName(),
                  failure.getMessage()));
          router.record(feedback(request, initial.taskType(), candidate.id(), false));
          continue;
        }
        execution.checkActive();
        reservation.settle(usage.apply(value));
        execution.checkActive();
        attempts.add(new RoutingAttempt(candidate.id(), elapsed(started), null, null));
        router.record(feedback(request, initial.taskType(), candidate.id(), true));
        RoutingDecision completed =
            index == 0
                ? initial
                : new RoutingDecision(
                    candidate,
                    order.subList(index + 1, order.size()),
                    initial.taskType(),
                    0.0,
                    Map.of("fallback", 1.0));
        return new RoutedResult<>(value, completed, attempts);
      }
    }
    if (budgetFailure != null) throw budgetFailure;
    throw new RoutingExecutionException(attempts, lastFailure);
  }

  private static void checkInterrupted() {
    if (Thread.currentThread().isInterrupted()) {
      throw new CancellationException("routed invocation interrupted");
    }
  }

  private RoutingFeedback feedback(
      RoutingRequest request, String taskType, String modelId, boolean success) {
    RoutingFeedback.Builder feedback =
        success ? RoutingFeedback.success(modelId) : RoutingFeedback.failure(modelId);
    request.session().ifPresent(feedback::sessionId);
    if (taskType != null) {
      feedback.taskType(taskType);
    }
    return feedback.build();
  }

  private RoutedModel<T> binding(ModelCandidate candidate) {
    T client = clients.get(candidate.id());
    if (client == null) {
      throw new IllegalStateException("no client is bound for candidate " + candidate.id());
    }
    return new RoutedModel<>(candidate, client);
  }

  private static Duration elapsed(long started) {
    return Duration.ofNanos(System.nanoTime() - started);
  }

  /** Fluent fleet builder. */
  public static final class Builder<T> {
    private final ModelRouter configuredRouter;
    private final Map<String, T> clients = new LinkedHashMap<>();
    private final List<ModelCandidate> candidates = new ArrayList<>();
    private TaskClassifier classifier = TaskClassifier.none();
    private RoutingPolicy policy = RoutingPolicy.BALANCED;
    private AdaptiveRoutingOptions adaptiveOptions = AdaptiveRoutingOptions.defaults();
    private final List<CandidateFilter> filters = new ArrayList<>();
    private final List<NamedScorer> scorers = new ArrayList<>();

    private Builder(ModelRouter configuredRouter) {
      this.configuredRouter = configuredRouter;
    }

    public Builder<T> model(ModelCandidate candidate, T client) {
      Objects.requireNonNull(candidate, "candidate");
      if (configuredRouter != null) {
        throw new IllegalStateException("bind clients by id when using an existing router");
      }
      candidates.add(candidate);
      client(candidate.id(), client);
      return this;
    }

    public Builder<T> client(String modelId, T client) {
      Objects.requireNonNull(modelId, "modelId");
      Objects.requireNonNull(client, "client");
      if (clients.putIfAbsent(modelId, client) != null) {
        throw new IllegalArgumentException("duplicate client id: " + modelId);
      }
      return this;
    }

    public Builder<T> clients(Map<String, ? extends T> value) {
      Objects.requireNonNull(value, "clients").forEach(this::client);
      return this;
    }

    public Builder<T> classifier(TaskClassifier value) {
      requireNewRouter("configure the classifier");
      this.classifier = Objects.requireNonNull(value, "classifier");
      return this;
    }

    public Builder<T> policy(RoutingPolicy value) {
      requireNewRouter("configure the policy");
      this.policy = Objects.requireNonNull(value, "policy");
      return this;
    }

    public Builder<T> adaptiveOptions(AdaptiveRoutingOptions value) {
      requireNewRouter("configure adaptive options");
      this.adaptiveOptions = Objects.requireNonNull(value, "adaptiveOptions");
      return this;
    }

    public Builder<T> filter(CandidateFilter value) {
      requireNewRouter("configure filters");
      filters.add(Objects.requireNonNull(value, "filter"));
      return this;
    }

    public Builder<T> scorer(String name, double weight, CandidateScorer scorer) {
      requireNewRouter("configure scorers");
      scorers.add(new NamedScorer(name, weight, scorer));
      return this;
    }

    public ModelFleet<T> build() {
      ModelRouter router = configuredRouter;
      if (router == null) {
        ModelRouter.Builder routerBuilder =
            ModelRouter.builder()
                .candidates(candidates)
                .classifier(classifier)
                .policy(policy)
                .adaptiveOptions(adaptiveOptions);
        filters.forEach(routerBuilder::filter);
        scorers.forEach(
            scorer -> routerBuilder.scorer(scorer.name(), scorer.weight(), scorer.scorer()));
        router = routerBuilder.build();
      }
      return new ModelFleet<>(router, clients);
    }

    private void requireNewRouter(String operation) {
      if (configuredRouter != null) {
        throw new IllegalStateException(operation + " on the ModelRouter before binding clients");
      }
    }

    private record NamedScorer(String name, double weight, CandidateScorer scorer) {}
  }
}
