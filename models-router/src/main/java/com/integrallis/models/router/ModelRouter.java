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

import java.time.Clock;
import java.time.Instant;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Chooses which model should answer a request, across in-process and hosted models.
 *
 * <p>Hard constraints filter candidates before independently named scorers rank them. Immutable
 * qualification evidence provides the cold-start prior; live availability, load, residency, and
 * explicit call feedback refine it at runtime.
 *
 * <p>Conversation continuity is bounded and evidence-based. The router can retain a warm model or
 * hard-lock an active tool/provider-state turn, but never treats KV tensors as portable between
 * different models.
 *
 * <p>The router selects; it never loads a provider SDK. Use {@link ModelFleet} to associate each
 * candidate with any application-supplied client and execute ordered fallback.
 */
public final class ModelRouter {

  private final List<ModelCandidate> candidates;
  private final Map<String, ModelCandidate> candidatesById;
  private final TaskClassifier classifier;
  private final RoutingPolicy policy;
  private final AdaptiveRoutingOptions adaptiveOptions;
  private final Clock clock;
  private final List<CandidateFilter> filters;
  private final List<NamedScorer> scorers;
  private final Map<String, MutableModelStatistics> statistics = new ConcurrentHashMap<>();
  private final Map<String, ModelRuntimeState> runtimeStates = new ConcurrentHashMap<>();
  private final Object sessionLock = new Object();
  private final LinkedHashMap<String, SessionState> sessions = new LinkedHashMap<>(16, 0.75f, true);

  private ModelRouter(Builder builder) {
    this.candidates = List.copyOf(builder.candidates);
    this.candidatesById = new LinkedHashMap<>();
    for (ModelCandidate candidate : candidates) {
      candidatesById.put(candidate.id(), candidate);
      statistics.put(candidate.id(), new MutableModelStatistics(candidate));
    }
    this.classifier = builder.classifier;
    this.policy = builder.policy;
    this.adaptiveOptions = builder.adaptiveOptions;
    this.clock = builder.clock;
    this.filters = List.copyOf(builder.filters);
    this.scorers = List.copyOf(builder.scorers);
  }

  /** Routes a plain query under the configured policy. */
  public RoutingDecision route(String query) {
    return route(RoutingRequest.builder(query).build(), policy, RoutingContinuity.none());
  }

  /** Routes a request under the configured policy. */
  public RoutingDecision route(RoutingRequest request) {
    return route(request, policy, RoutingContinuity.none());
  }

  /** Routes a request under a one-off policy. */
  public RoutingDecision route(RoutingRequest request, RoutingPolicy override) {
    return route(request, override, RoutingContinuity.none());
  }

  /** Routes a request under the configured policy with explicit continuity evidence. */
  public RoutingDecision route(RoutingRequest request, RoutingContinuity continuity) {
    return route(request, policy, continuity);
  }

  /**
   * Routes with explicit conversation-continuity evidence.
   *
   * <p>An active tool loop or non-portable provider state retains the current model while it is
   * healthy and eligible. Cached-prefix token counts are model-specific affinity evidence only;
   * they are never transferred to another model.
   */
  public RoutingDecision route(
      RoutingRequest request, RoutingPolicy override, RoutingContinuity continuity) {
    Objects.requireNonNull(request, "request");
    Objects.requireNonNull(override, "policy");
    Objects.requireNonNull(continuity, "continuity");

    String taskType = request.taskType().orElseGet(() -> classifier.classify(request.query()));
    SessionState session = activeSession(request);
    String previousModel = session == null ? null : session.modelId;
    RoutingContinuity effectiveContinuity = mergeContinuity(continuity, session);
    List<ModelCandidate> eligible =
        eligible(request, override, taskType, previousModel, effectiveContinuity);
    Scoring scoring =
        new Scoring(request, override, taskType, eligible, previousModel, effectiveContinuity);
    List<ScoredCandidate> ranked = new ArrayList<>(eligible.size());
    for (ModelCandidate candidate : eligible) {
      ranked.add(scoring.score(candidate));
    }
    // List.sort is stable, so an exact tie preserves the application's declared fleet order.
    ranked.sort(Comparator.comparingDouble(ScoredCandidate::score).reversed());

    ScoredCandidate selected = ranked.getFirst();
    boolean hardLocked = false;
    boolean retainedBySwitchMargin = false;
    ScoredCandidate current = find(ranked, previousModel);
    if (current != null && mustPreserveContinuity(effectiveContinuity, session)) {
      selected = current;
      hardLocked = true;
    } else if (current != null
        && selected != current
        && !switchAdvantageIsMaterial(selected, current, session)) {
      selected = current;
      retainedBySwitchMargin = true;
    }

    rememberSelection(request, selected.candidate());
    List<ModelCandidate> fallbacks = new ArrayList<>();
    for (ScoredCandidate scored : ranked) {
      if (scored != selected) {
        fallbacks.add(scored.candidate());
      }
    }
    Map<String, Double> breakdown = new LinkedHashMap<>(selected.breakdown());
    if (hardLocked) {
      breakdown.put("continuity-lock", 1.0);
    } else if (retainedBySwitchMargin) {
      breakdown.put("switch-margin", adaptiveOptions.switchMargin());
    }
    return new RoutingDecision(
        selected.candidate(), fallbacks, taskType, selected.score(), breakdown);
  }

  /** Returns the immutable candidate descriptors configured on this router. */
  public List<ModelCandidate> candidates() {
    return candidates;
  }

  /** Replaces the latest operational state for one candidate. */
  public void updateRuntimeState(String modelId, ModelRuntimeState state) {
    requireCandidate(modelId);
    runtimeStates.put(modelId, Objects.requireNonNull(state, "state"));
  }

  /** Removes operational state so the candidate uses qualification priors again. */
  public void clearRuntimeState(String modelId) {
    requireCandidate(modelId);
    runtimeStates.remove(modelId);
  }

  /** Records an explicit call outcome and any genuine TTFT/throughput/cache measurements. */
  public void record(RoutingFeedback feedback) {
    Objects.requireNonNull(feedback, "feedback");
    MutableModelStatistics model = statistics.get(feedback.modelId());
    if (model == null) {
      throw new IllegalArgumentException("unknown candidate id: " + feedback.modelId());
    }
    model.record(feedback, adaptiveOptions, clock.instant());
    feedback.sessionId().ifPresent(sessionId -> recordSessionFeedback(sessionId, feedback));
  }

  /** Returns the effective model status used for routing. */
  public Optional<RoutingModelStatus> status(String modelId) {
    MutableModelStatistics model = statistics.get(modelId);
    if (model == null) {
      return Optional.empty();
    }
    return Optional.of(model.snapshot(modelId, runtimeState(modelId), clock.instant()));
  }

  /** Explicitly ends a conversation and releases its affinity/cache accounting state. */
  public boolean closeSession(String sessionId) {
    Objects.requireNonNull(sessionId, "sessionId");
    synchronized (sessionLock) {
      return sessions.remove(sessionId) != null;
    }
  }

  /** Clears all conversation routing state. */
  public void clearSessions() {
    synchronized (sessionLock) {
      sessions.clear();
    }
  }

  /** Returns the number of live bounded session entries. */
  public int activeSessionCount() {
    synchronized (sessionLock) {
      evictExpiredSessions(clock.instant());
      return sessions.size();
    }
  }

  private List<ModelCandidate> eligible(
      RoutingRequest request,
      RoutingPolicy override,
      String taskType,
      String previousModel,
      RoutingContinuity continuity) {
    List<ModelCandidate> eligible = new ArrayList<>();
    LinkedHashSet<String> rejections = new LinkedHashSet<>();
    Instant now = clock.instant();
    for (ModelCandidate candidate : candidates) {
      ModelRuntimeState runtime = runtimeState(candidate.id());
      RoutingModelStatus status =
          statistics.get(candidate.id()).snapshot(candidate.id(), runtime, now);
      if (!runtime.available()) {
        rejections.add(candidate.id() + " is unavailable");
        continue;
      }
      if (status.coolingDown()) {
        rejections.add(candidate.id() + " is cooling down after repeated failures");
        continue;
      }
      if (override.isLocalOnly() && !candidate.local()) {
        rejections.add("policy is local-only");
        continue;
      }
      if (candidate.contextWindow() < request.estimatedTokens()) {
        rejections.add("no model has a context window of " + request.estimatedTokens() + " tokens");
        continue;
      }
      if (taskType != null && !candidate.tags().isEmpty() && !candidate.tags().contains(taskType)) {
        rejections.add("no model declares the " + taskType + " tag");
        continue;
      }
      if (override.minimumQuality().isPresent()
          && candidate.qualityFor(taskType) < override.minimumQuality().getAsDouble()) {
        rejections.add(
            "no model reaches the quality floor of " + override.minimumQuality().getAsDouble());
        continue;
      }
      if (override.maximumCostPerMillionTokens().isPresent()
          && candidate.blendedCostPerMillionTokens()
              > override.maximumCostPerMillionTokens().getAsDouble()) {
        rejections.add(
            "no model costs at most " + override.maximumCostPerMillionTokens().getAsDouble());
        continue;
      }
      if (override.maximumTimeToFirstTokenMillis().isPresent()
          && status.timeToFirstTokenMillis()
              > override.maximumTimeToFirstTokenMillis().getAsDouble()) {
        rejections.add(
            "no model answers within "
                + override.maximumTimeToFirstTokenMillis().getAsDouble()
                + "ms");
        continue;
      }
      RoutingEvaluation evaluation =
          new RoutingEvaluation(
              request,
              override,
              taskType,
              candidate,
              candidates,
              status,
              previousModel,
              continuity);
      Optional<String> customRejection = customRejection(evaluation);
      if (customRejection.isPresent()) {
        rejections.add(customRejection.orElseThrow());
        continue;
      }
      eligible.add(candidate);
    }
    if (eligible.isEmpty()) {
      throw new NoEligibleModelException(
          rejections.isEmpty() ? "no candidates were registered" : String.join("; ", rejections));
    }
    return eligible;
  }

  private Optional<String> customRejection(RoutingEvaluation evaluation) {
    for (CandidateFilter filter : filters) {
      Optional<String> rejection =
          Objects.requireNonNull(filter.rejection(evaluation), "filter result");
      if (rejection.isPresent()) {
        return rejection;
      }
    }
    return Optional.empty();
  }

  private boolean mustPreserveContinuity(RoutingContinuity continuity, SessionState session) {
    if (session == null) {
      return false;
    }
    if (continuity.activeToolLoop() && adaptiveOptions.toolLoopHardLock()) {
      return true;
    }
    if (!continuity.contextPortable() && adaptiveOptions.nonPortableContextHardLock()) {
      return true;
    }
    return session.turns < adaptiveOptions.minimumTurnsBeforeSwitch();
  }

  private boolean switchAdvantageIsMaterial(
      ScoredCandidate challenger, ScoredCandidate current, SessionState session) {
    if (session == null) {
      return true;
    }
    return challenger.score() - current.score() > adaptiveOptions.switchMargin();
  }

  private SessionState activeSession(RoutingRequest request) {
    if (request.session().isEmpty()) {
      return null;
    }
    String sessionId = request.session().orElseThrow();
    Instant now = clock.instant();
    synchronized (sessionLock) {
      evictExpiredSessions(now);
      SessionState state = sessions.get(sessionId);
      return state == null ? null : state.snapshot();
    }
  }

  private void rememberSelection(RoutingRequest request, ModelCandidate selected) {
    request
        .session()
        .ifPresent(
            sessionId -> {
              Instant now = clock.instant();
              synchronized (sessionLock) {
                SessionState state = sessions.get(sessionId);
                if (state == null) {
                  state = new SessionState(selected.id(), now);
                  sessions.put(sessionId, state);
                } else {
                  if (!state.modelId.equals(selected.id())) {
                    state.modelId = selected.id();
                    state.cachedInputTokens = 0;
                  }
                  state.lastSeen = now;
                }
                state.turns++;
                enforceSessionBound();
              }
            });
  }

  private void recordSessionFeedback(String sessionId, RoutingFeedback feedback) {
    Instant now = clock.instant();
    synchronized (sessionLock) {
      evictExpiredSessions(now);
      SessionState state = sessions.get(sessionId);
      if (state == null) {
        state = new SessionState(feedback.modelId(), now);
        sessions.put(sessionId, state);
      }
      if (feedback.successful() && !state.modelId.equals(feedback.modelId())) {
        state.modelId = feedback.modelId();
      }
      if (feedback.promptTokens().isPresent()) {
        state.cachedInputTokens = feedback.cachedInputTokens().orElse(0);
      }
      state.lastSeen = now;
      enforceSessionBound();
    }
  }

  private RoutingContinuity mergeContinuity(RoutingContinuity supplied, SessionState session) {
    if (session == null || session.cachedInputTokens <= 0) {
      return supplied;
    }
    Map<String, Integer> cache = new LinkedHashMap<>(supplied.cachedPrefixTokens());
    cache.putIfAbsent(session.modelId, session.cachedInputTokens);
    return new RoutingContinuity(supplied.activeToolLoop(), supplied.contextPortable(), cache);
  }

  private void evictExpiredSessions(Instant now) {
    if (adaptiveOptions.sessionIdleTimeout().isZero()) {
      sessions.clear();
      return;
    }
    sessions
        .entrySet()
        .removeIf(
            entry ->
                !entry.getValue().lastSeen.plus(adaptiveOptions.sessionIdleTimeout()).isAfter(now));
  }

  private void enforceSessionBound() {
    while (sessions.size() > adaptiveOptions.maximumSessions()) {
      String eldest = sessions.keySet().iterator().next();
      sessions.remove(eldest);
    }
  }

  private ModelRuntimeState runtimeState(String modelId) {
    return runtimeStates.getOrDefault(modelId, ModelRuntimeState.unknown());
  }

  private void requireCandidate(String modelId) {
    Objects.requireNonNull(modelId, "modelId");
    if (!candidatesById.containsKey(modelId)) {
      throw new IllegalArgumentException("unknown candidate id: " + modelId);
    }
  }

  private static ScoredCandidate find(List<ScoredCandidate> ranked, String modelId) {
    if (modelId == null) {
      return null;
    }
    for (ScoredCandidate candidate : ranked) {
      if (candidate.candidate().id().equals(modelId)) {
        return candidate;
      }
    }
    return null;
  }

  /** Starts a router over every model the installed catalogs report. */
  public static Builder discoverLocal() {
    return discoverLocal(false);
  }

  /** Starts a router over installed catalogs, optionally including on-device intelligence. */
  public static Builder discoverLocal(boolean includeOnDeviceIntelligence) {
    return builder().candidates(CatalogDiscovery.discover(includeOnDeviceIntelligence));
  }

  /** Starts building a router with no candidates registered. */
  public static Builder builder() {
    return new Builder();
  }

  private final class Scoring {
    private final RoutingRequest request;
    private final RoutingPolicy policy;
    private final String taskType;
    private final List<ModelCandidate> eligible;
    private final String previousModel;
    private final RoutingContinuity continuity;
    private final double minCost;
    private final double maxCost;
    private final long minTtft;
    private final long maxTtft;
    private final int minQueue;
    private final int maxQueue;
    private final boolean hasQueueEvidence;
    private final boolean hasResidencyEvidence;
    private final boolean hasCacheEvidence;

    private Scoring(
        RoutingRequest request,
        RoutingPolicy policy,
        String taskType,
        List<ModelCandidate> eligible,
        String previousModel,
        RoutingContinuity continuity) {
      this.request = request;
      this.policy = policy;
      this.taskType = taskType;
      this.eligible = eligible;
      this.previousModel = previousModel;
      this.continuity = continuity;
      double lowCost = Double.MAX_VALUE;
      double highCost = 0;
      long lowTtft = Long.MAX_VALUE;
      long highTtft = 0;
      int lowQueue = Integer.MAX_VALUE;
      int highQueue = 0;
      boolean queueEvidence = false;
      boolean residencyEvidence = false;
      for (ModelCandidate candidate : eligible) {
        RoutingModelStatus status = status(candidate.id()).orElseThrow();
        lowCost = Math.min(lowCost, candidate.blendedCostPerMillionTokens());
        highCost = Math.max(highCost, candidate.blendedCostPerMillionTokens());
        lowTtft = Math.min(lowTtft, status.timeToFirstTokenMillis());
        highTtft = Math.max(highTtft, status.timeToFirstTokenMillis());
        ModelRuntimeState runtime = status.runtimeState();
        residencyEvidence |= runtime.resident().isPresent();
        if (runtime.queueDepth().isPresent()) {
          queueEvidence = true;
          lowQueue = Math.min(lowQueue, runtime.queueDepth().getAsInt());
          highQueue = Math.max(highQueue, runtime.queueDepth().getAsInt());
        }
      }
      this.minCost = lowCost;
      this.maxCost = highCost;
      this.minTtft = lowTtft;
      this.maxTtft = highTtft;
      this.minQueue = queueEvidence ? lowQueue : 0;
      this.maxQueue = queueEvidence ? highQueue : 0;
      this.hasQueueEvidence = queueEvidence;
      this.hasResidencyEvidence = residencyEvidence;
      this.hasCacheEvidence =
          continuity.cachedPrefixTokens().values().stream().anyMatch(value -> value > 0);
    }

    private ScoredCandidate score(ModelCandidate candidate) {
      RoutingModelStatus status = status(candidate.id()).orElseThrow();
      double totalWeight =
          policy.costWeight()
              + policy.qualityWeight()
              + policy.latencyWeight()
              + policy.localityWeight();
      if (hasCacheEvidence) {
        totalWeight += adaptiveOptions.cacheAffinityWeight();
      }
      if (hasResidencyEvidence) {
        totalWeight += adaptiveOptions.residencyWeight();
      }
      if (hasQueueEvidence) {
        totalWeight += adaptiveOptions.loadWeight();
      }
      for (NamedScorer scorer : scorers) {
        totalWeight += scorer.weight();
      }
      if (totalWeight <= 0) {
        totalWeight = 1.0;
      }

      Map<String, Double> parts = new LinkedHashMap<>();
      parts.put("cost", policy.costWeight() / totalWeight * cheapness(candidate));
      parts.put("quality", policy.qualityWeight() / totalWeight * candidate.qualityFor(taskType));
      parts.put("latency", policy.latencyWeight() / totalWeight * quickness(status));
      parts.put(
          "locality", policy.localityWeight() / totalWeight * (candidate.local() ? 1.0 : 0.0));
      if (hasCacheEvidence) {
        parts.put(
            "cache-affinity",
            adaptiveOptions.cacheAffinityWeight() / totalWeight * cacheAffinity(candidate));
      }
      if (hasResidencyEvidence) {
        double resident =
            status.runtimeState().resident().map(value -> value ? 1.0 : 0.0).orElse(0.5);
        parts.put("residency", adaptiveOptions.residencyWeight() / totalWeight * resident);
      }
      if (hasQueueEvidence) {
        parts.put("load", adaptiveOptions.loadWeight() / totalWeight * queueAvailability(status));
      }
      RoutingEvaluation evaluation =
          new RoutingEvaluation(
              request, policy, taskType, candidate, eligible, status, previousModel, continuity);
      for (NamedScorer scorer : scorers) {
        double value = scorer.scorer().score(evaluation);
        if (!Double.isFinite(value) || value < 0 || value > 1) {
          throw new IllegalArgumentException(
              "scorer " + scorer.name() + " returned " + value + "; expected [0, 1]");
        }
        parts.put(scorer.name(), scorer.weight() / totalWeight * value);
      }
      double reliability = reliability(status);
      parts.put("availability", reliability);
      double merit =
          parts.entrySet().stream()
              .filter(entry -> !entry.getKey().equals("availability"))
              .mapToDouble(Map.Entry::getValue)
              .sum();
      return new ScoredCandidate(candidate, clamp(merit * reliability), parts);
    }

    private double reliability(RoutingModelStatus status) {
      double sharpness = policy.availabilityWeight() * 10.0;
      return sharpness <= 0 ? 1.0 : Math.pow(status.successRate(), sharpness);
    }

    private double cheapness(ModelCandidate candidate) {
      if (maxCost - minCost < 1.0e-9) {
        return 1.0;
      }
      return 1.0 - (candidate.blendedCostPerMillionTokens() - minCost) / (maxCost - minCost);
    }

    private double quickness(RoutingModelStatus status) {
      if (maxTtft == minTtft) {
        return 1.0;
      }
      return 1.0
          - (double) (status.timeToFirstTokenMillis() - minTtft) / (double) (maxTtft - minTtft);
    }

    private double queueAvailability(RoutingModelStatus status) {
      if (status.runtimeState().queueDepth().isEmpty()) {
        return 0.5;
      }
      if (maxQueue == minQueue) {
        return 1.0;
      }
      return 1.0
          - (double) (status.runtimeState().queueDepth().getAsInt() - minQueue)
              / (double) (maxQueue - minQueue);
    }

    private double cacheAffinity(ModelCandidate candidate) {
      int cached = continuity.cachedPrefixTokens().getOrDefault(candidate.id(), 0);
      if (cached == 0) {
        return 0;
      }
      int promptTokens = request.estimatedTokens();
      if (promptTokens > 0) {
        return clamp((double) cached / promptTokens);
      }
      int maximum =
          continuity.cachedPrefixTokens().values().stream()
              .mapToInt(Integer::intValue)
              .max()
              .orElse(1);
      return (double) cached / maximum;
    }
  }

  private static double clamp(double value) {
    return Math.max(0.0, Math.min(1.0, value));
  }

  private record NamedScorer(String name, double weight, CandidateScorer scorer) {}

  private record ScoredCandidate(
      ModelCandidate candidate, double score, Map<String, Double> breakdown) {}

  private static final class SessionState {
    private String modelId;
    private Instant lastSeen;
    private int turns;
    private int cachedInputTokens;

    private SessionState(String modelId, Instant lastSeen) {
      this.modelId = modelId;
      this.lastSeen = lastSeen;
    }

    private SessionState snapshot() {
      SessionState copy = new SessionState(modelId, lastSeen);
      copy.turns = turns;
      copy.cachedInputTokens = cachedInputTokens;
      return copy;
    }
  }

  private static final class MutableModelStatistics {
    private double successRate;
    private double timeToFirstTokenMillis;
    private double tokensPerSecond;
    private int consecutiveFailures;
    private Instant cooldownUntil;

    private MutableModelStatistics(ModelCandidate candidate) {
      this.successRate = candidate.successRate();
      this.timeToFirstTokenMillis = candidate.timeToFirstTokenMillis();
      this.tokensPerSecond = candidate.tokensPerSecond();
    }

    private synchronized void record(
        RoutingFeedback feedback, AdaptiveRoutingOptions options, Instant now) {
      double weight = options.feedbackWeight();
      successRate = (1.0 - weight) * successRate + weight * (feedback.successful() ? 1.0 : 0.0);
      if (feedback.successful()) {
        consecutiveFailures = 0;
        cooldownUntil = null;
      } else {
        consecutiveFailures++;
        if (consecutiveFailures >= options.consecutiveFailuresBeforeCooldown()) {
          cooldownUntil = now.plus(options.failureCooldown());
        }
      }
      if (feedback.timeToFirstTokenMillis().isPresent()) {
        timeToFirstTokenMillis =
            (1.0 - weight) * timeToFirstTokenMillis
                + weight * feedback.timeToFirstTokenMillis().getAsLong();
      }
      if (feedback.tokensPerSecond().isPresent()) {
        tokensPerSecond =
            (1.0 - weight) * tokensPerSecond + weight * feedback.tokensPerSecond().getAsDouble();
      }
    }

    private synchronized RoutingModelStatus snapshot(
        String modelId, ModelRuntimeState runtimeState, Instant now) {
      return new RoutingModelStatus(
          modelId,
          clamp(successRate),
          Math.max(0, Math.round(timeToFirstTokenMillis)),
          Math.max(Double.MIN_NORMAL, tokensPerSecond),
          consecutiveFailures,
          cooldownUntil != null && cooldownUntil.isAfter(now),
          runtimeState);
    }
  }

  /** Fluent builder. */
  public static final class Builder {
    private List<ModelCandidate> candidates = List.of();
    private TaskClassifier classifier = TaskClassifier.none();
    private RoutingPolicy policy = RoutingPolicy.BALANCED;
    private AdaptiveRoutingOptions adaptiveOptions = AdaptiveRoutingOptions.defaults();
    private Clock clock = Clock.systemUTC();
    private final List<CandidateFilter> filters = new ArrayList<>();
    private final List<NamedScorer> scorers = new ArrayList<>();

    private Builder() {}

    public Builder candidates(List<ModelCandidate> value) {
      this.candidates = List.copyOf(Objects.requireNonNull(value, "candidates"));
      return this;
    }

    public Builder classifier(TaskClassifier value) {
      this.classifier = Objects.requireNonNull(value, "classifier");
      return this;
    }

    public Builder policy(RoutingPolicy value) {
      this.policy = Objects.requireNonNull(value, "policy");
      return this;
    }

    public Builder adaptiveOptions(AdaptiveRoutingOptions value) {
      this.adaptiveOptions = Objects.requireNonNull(value, "adaptiveOptions");
      return this;
    }

    public Builder filter(CandidateFilter value) {
      filters.add(Objects.requireNonNull(value, "filter"));
      return this;
    }

    public Builder scorer(String name, double weight, CandidateScorer scorer) {
      if (name == null || name.isBlank()) {
        throw new IllegalArgumentException("scorer name must not be blank");
      }
      if (!Double.isFinite(weight) || weight <= 0) {
        throw new IllegalArgumentException("scorer weight must be finite and positive");
      }
      scorers.add(new NamedScorer(name, weight, Objects.requireNonNull(scorer, "scorer")));
      return this;
    }

    /** Supplies a clock for deterministic simulations and session-lifecycle tests. */
    public Builder clock(Clock value) {
      this.clock = Objects.requireNonNull(value, "clock");
      return this;
    }

    public ModelRouter build() {
      if (candidates.isEmpty()) {
        throw new IllegalArgumentException("at least one candidate model is required");
      }
      LinkedHashSet<String> ids = new LinkedHashSet<>();
      for (ModelCandidate candidate : candidates) {
        if (!ids.add(candidate.id())) {
          throw new IllegalArgumentException("duplicate candidate id: " + candidate.id());
        }
      }
      LinkedHashSet<String> scorerNames = new LinkedHashSet<>();
      for (NamedScorer scorer : scorers) {
        if (!scorerNames.add(scorer.name())) {
          throw new IllegalArgumentException("duplicate scorer name: " + scorer.name());
        }
        if (ReservedNames.ALL.contains(scorer.name())) {
          throw new IllegalArgumentException("reserved scorer name: " + scorer.name());
        }
      }
      return new ModelRouter(this);
    }
  }

  private static final class ReservedNames {
    private static final Set<String> ALL =
        Set.of(
            "cost",
            "quality",
            "latency",
            "locality",
            "availability",
            "cache-affinity",
            "residency",
            "load",
            "continuity-lock",
            "switch-margin",
            "fallback");

    private ReservedNames() {}
  }
}
