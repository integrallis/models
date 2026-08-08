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

import java.util.ArrayList;
import java.util.Comparator;
import java.util.HashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.concurrent.ConcurrentHashMap;

/**
 * Chooses which model should answer a request, across in-process and hosted models.
 *
 * <p>Scoring is a weighted sum over normalized dimensions rather than a difficulty classifier that
 * picks a tier. Difficulty alone cannot express "this is easy but the cheap model is currently
 * failing" or "this is hard but the prompt does not fit the frontier model's cheaper sibling", and
 * routing has to balance cost, latency, specialization and reliability at once.
 *
 * <p>Hard constraints filter before scoring. A context window too small for the prompt is not a
 * preference that a large enough quality advantage should overcome.
 *
 * <p>The router selects; it never calls. Applications map {@link RoutingDecision#modelId()} back to
 * their own client, which is why no provider SDK appears here.
 *
 * <pre>{@code
 * ModelRouter router = ModelRouter.builder()
 *     .candidates(List.of(localSmall, hostedBudget, frontier))
 *     .policy(RoutingPolicy.BALANCED)
 *     .build();
 *
 * RoutingDecision decision = router.route("refactor this function");
 * ChatModel model = myClients.get(decision.modelId());
 * }</pre>
 */
public final class ModelRouter {

  private final List<ModelCandidate> candidates;
  private final TaskClassifier classifier;
  private final RoutingPolicy policy;
  private final Map<String, String> sessionPins = new ConcurrentHashMap<>();

  private ModelRouter(Builder builder) {
    this.candidates = List.copyOf(builder.candidates);
    this.classifier = builder.classifier;
    this.policy = builder.policy;
  }

  /**
   * Routes a plain query under the configured policy.
   *
   * @param query the user's request
   * @return the chosen model and why
   */
  public RoutingDecision route(String query) {
    return route(RoutingRequest.builder(query).build(), policy);
  }

  /**
   * Routes a request under the configured policy.
   *
   * @param request the request
   * @return the chosen model and why
   */
  public RoutingDecision route(RoutingRequest request) {
    return route(request, policy);
  }

  /**
   * Routes a request under a one-off policy.
   *
   * @param request the request
   * @param override policy to apply instead of the configured one
   * @return the chosen model and why
   */
  public RoutingDecision route(RoutingRequest request, RoutingPolicy override) {
    Objects.requireNonNull(request, "request");
    Objects.requireNonNull(override, "policy");

    String taskType = request.taskType().orElseGet(() -> classifier.classify(request.query()));
    List<ModelCandidate> eligible = eligible(request, override, taskType);

    // Keep a conversation on its first choice while that choice stays eligible: switching models
    // mid-conversation discards the provider's prompt cache, which is often the larger saving.
    String pinned = request.session().map(sessionPins::get).orElse(null);
    if (pinned != null) {
      for (ModelCandidate candidate : eligible) {
        if (candidate.id().equals(pinned)) {
          return decide(candidate, eligible, taskType, override);
        }
      }
    }

    ModelCandidate best = null;
    double bestScore = -1;
    Map<ModelCandidate, Double> scores = new HashMap<>();
    Normalizer normalizer = new Normalizer(eligible, taskType);
    for (ModelCandidate candidate : eligible) {
      double score = normalizer.score(candidate, override);
      scores.put(candidate, score);
      if (score > bestScore) {
        bestScore = score;
        best = candidate;
      }
    }

    List<ModelCandidate> ranked = new ArrayList<>(eligible);
    ranked.sort(Comparator.comparingDouble((ModelCandidate c) -> scores.get(c)).reversed());
    request.session().ifPresent(id -> sessionPins.put(id, ranked.get(0).id()));
    return decide(best, ranked, taskType, override);
  }

  private RoutingDecision decide(
      ModelCandidate winner, List<ModelCandidate> ranked, String taskType, RoutingPolicy override) {
    Normalizer normalizer = new Normalizer(ranked, taskType);
    List<ModelCandidate> fallbacks = new ArrayList<>(ranked);
    fallbacks.remove(winner);
    return new RoutingDecision(
        winner,
        fallbacks,
        taskType,
        normalizer.score(winner, override),
        normalizer.breakdown(winner, override));
  }

  private List<ModelCandidate> eligible(
      RoutingRequest request, RoutingPolicy override, String taskType) {
    List<ModelCandidate> eligible = new ArrayList<>();
    // Report every reason, not just the last one: with a mixed fleet the informative rejection is
    // rarely the one that happens to be evaluated last.
    LinkedHashSet<String> rejections = new LinkedHashSet<>();
    for (ModelCandidate candidate : candidates) {
      if (override.isLocalOnly() && !candidate.local()) {
        rejections.add("policy is local-only");
        continue;
      }
      if (candidate.contextWindow() < request.estimatedTokens()) {
        rejections.add("no model has a context window of " + request.estimatedTokens() + " tokens");
        continue;
      }
      // Tags declare what a model is for. An untagged model is treated as general purpose.
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
          && candidate.timeToFirstTokenMillis()
              > override.maximumTimeToFirstTokenMillis().getAsDouble()) {
        rejections.add(
            "no model answers within "
                + override.maximumTimeToFirstTokenMillis().getAsDouble()
                + "ms");
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

  /**
   * Starts a router over every model the installed catalogs report.
   *
   * <p>Equivalent to {@code builder().candidates(CatalogDiscovery.discover())}, which is what most
   * callers want: describing a fleet by hand means inventing latency and quality figures nobody
   * has. Add hosted models to the discovered ones with {@link Builder#candidates(List)}.
   *
   * <p>Logs and returns an empty fleet when no catalog is installed rather than failing — routing
   * between hosted models only is a legitimate use.
   *
   * @return a builder preloaded with the discovered models
   */
  public static Builder discoverLocal() {
    return discoverLocal(false);
  }

  /**
   * Starts a router over the installed catalogs, optionally including on-device intelligence.
   *
   * <p>Apple Foundation Models is present because of the hardware, not because anyone installed it,
   * so it stays out of the fleet unless asked for. Defaulting the other way would make the same
   * code route differently on a developer's Mac than in production, and that difference would
   * surface as unexplained behaviour rather than as a decision.
   *
   * @param includeOnDeviceIntelligence whether to include platform-provided models
   * @return a builder preloaded with the discovered models
   */
  public static Builder discoverLocal(boolean includeOnDeviceIntelligence) {
    return builder().candidates(CatalogDiscovery.discover(includeOnDeviceIntelligence));
  }

  /**
   * Starts building a router with no candidates registered.
   *
   * @return a new builder
   */
  public static Builder builder() {
    return new Builder();
  }

  /** Normalizes each dimension across the eligible set so weights compare like with like. */
  private static final class Normalizer {
    private final double minCost;
    private final double maxCost;
    private final double minTtft;
    private final double maxTtft;
    private final String taskType;

    Normalizer(List<ModelCandidate> eligible, String taskType) {
      this.taskType = taskType;
      double lowCost = Double.MAX_VALUE;
      double highCost = 0;
      double lowTtft = Double.MAX_VALUE;
      double highTtft = 0;
      for (ModelCandidate candidate : eligible) {
        lowCost = Math.min(lowCost, candidate.blendedCostPerMillionTokens());
        highCost = Math.max(highCost, candidate.blendedCostPerMillionTokens());
        lowTtft = Math.min(lowTtft, candidate.timeToFirstTokenMillis());
        highTtft = Math.max(highTtft, candidate.timeToFirstTokenMillis());
      }
      this.minCost = lowCost;
      this.maxCost = highCost;
      this.minTtft = lowTtft;
      this.maxTtft = highTtft;
    }

    Map<String, Double> breakdown(ModelCandidate candidate, RoutingPolicy policy) {
      double total = meritWeight(policy);
      Map<String, Double> parts = new HashMap<>();
      parts.put("cost", policy.costWeight() / total * cheapness(candidate));
      parts.put("quality", policy.qualityWeight() / total * candidate.qualityFor(taskType));
      parts.put("latency", policy.latencyWeight() / total * quickness(candidate));
      parts.put("locality", policy.localityWeight() / total * (candidate.local() ? 1.0 : 0.0));
      parts.put("availability", reliability(candidate, policy));
      return parts;
    }

    /**
     * Merit scaled by the chance the call actually succeeds.
     *
     * <p>Reliability multiplies rather than adding. A model that fails four calls in five is not
     * "slightly worse" than a healthy one, and any additive weight small enough to be reasonable
     * still lets a cheap, fast, broken model win on the other dimensions.
     */
    double score(ModelCandidate candidate, RoutingPolicy policy) {
      Map<String, Double> parts = breakdown(candidate, policy);
      double merit =
          parts.get("cost") + parts.get("quality") + parts.get("latency") + parts.get("locality");
      return merit * parts.get("availability");
    }

    /** Availability weight sets how sharply failures are punished; zero ignores them. */
    private double reliability(ModelCandidate candidate, RoutingPolicy policy) {
      double sharpness = policy.availabilityWeight() * 10.0;
      if (sharpness <= 0) {
        return 1.0;
      }
      return Math.pow(candidate.successRate(), sharpness);
    }

    /** Cheapest scores 1, dearest scores 0; a single price point scores 1. */
    private double cheapness(ModelCandidate candidate) {
      if (maxCost - minCost < 1.0e-9) {
        return 1.0;
      }
      return 1.0 - (candidate.blendedCostPerMillionTokens() - minCost) / (maxCost - minCost);
    }

    /** Fastest scores 1, slowest scores 0; a single latency scores 1. */
    private double quickness(ModelCandidate candidate) {
      if (maxTtft - minTtft < 1.0e-9) {
        return 1.0;
      }
      return 1.0 - (candidate.timeToFirstTokenMillis() - minTtft) / (maxTtft - minTtft);
    }

    /** Availability is excluded: it multiplies the others rather than competing with them. */
    private static double meritWeight(RoutingPolicy policy) {
      double total =
          policy.costWeight()
              + policy.qualityWeight()
              + policy.latencyWeight()
              + policy.localityWeight();
      return total > 0 ? total : 1.0;
    }
  }

  /** Fluent builder. */
  public static final class Builder {
    private List<ModelCandidate> candidates = List.of();
    private TaskClassifier classifier = TaskClassifier.none();
    private RoutingPolicy policy = RoutingPolicy.BALANCED;

    private Builder() {}

    /**
     * Registers the models this application can actually reach.
     *
     * @param value candidate models
     * @return this builder
     */
    public Builder candidates(List<ModelCandidate> value) {
      this.candidates = Objects.requireNonNull(value, "candidates");
      return this;
    }

    /**
     * Sets the task classifier.
     *
     * @param value classifier
     * @return this builder
     */
    public Builder classifier(TaskClassifier value) {
      this.classifier = Objects.requireNonNull(value, "classifier");
      return this;
    }

    /**
     * Sets the default policy.
     *
     * @param value policy
     * @return this builder
     */
    public Builder policy(RoutingPolicy value) {
      this.policy = Objects.requireNonNull(value, "policy");
      return this;
    }

    /**
     * Builds the router.
     *
     * @return an immutable router
     */
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
      return new ModelRouter(this);
    }
  }
}
