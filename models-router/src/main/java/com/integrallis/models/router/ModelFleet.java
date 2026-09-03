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

  /** Returns the full explainable decision without invoking a client. */
  public RoutingDecision decide(RoutingRequest request) {
    return router.route(request);
  }

  /** Returns a full decision that accounts for the current conversation boundary. */
  public RoutingDecision decide(RoutingRequest request, RoutingContinuity continuity) {
    return router.route(request, continuity);
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

  /** Executes with tool-loop, provider-state, and model-specific cache evidence. */
  public <R> RoutedResult<R> execute(
      RoutingRequest request,
      RoutingContinuity continuity,
      ModelInvocation<? super T, ? extends R> invocation) {
    Objects.requireNonNull(request, "request");
    Objects.requireNonNull(continuity, "continuity");
    Objects.requireNonNull(invocation, "invocation");
    RoutingDecision initial = router.route(request, continuity);
    List<ModelCandidate> order = new ArrayList<>();
    order.add(initial.selected());
    order.addAll(initial.fallbacks());
    List<RoutingAttempt> attempts = new ArrayList<>();
    Throwable lastFailure = null;
    for (int index = 0; index < order.size(); index++) {
      ModelCandidate candidate = order.get(index);
      long started = System.nanoTime();
      try {
        R value = invocation.invoke(binding(candidate).client());
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
      } catch (Exception failure) {
        lastFailure = failure;
        attempts.add(
            new RoutingAttempt(
                candidate.id(),
                elapsed(started),
                failure.getClass().getName(),
                failure.getMessage()));
        router.record(feedback(request, initial.taskType(), candidate.id(), false));
      }
    }
    throw new RoutingExecutionException(attempts, lastFailure);
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
