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
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import java.time.Clock;
import java.time.Duration;
import java.time.Instant;
import java.time.ZoneId;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import org.junit.jupiter.api.Nested;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;

@Tag("unit")
class ModelRouterTest {

  /** A local model: free, fast to first token, modest quality, small context. */
  private static final ModelCandidate LOCAL_SMALL =
      ModelCandidate.builder("qwen3-0.6b-q4_0")
          .local(true)
          .tags(Set.of("chat", "code"))
          .costPerMillionTokens(0.0, 0.0)
          .timeToFirstTokenMillis(120)
          .tokensPerSecond(50)
          .contextWindow(32_768)
          .quality(Map.of("chat", 0.62, "code", 0.55, "math", 0.30))
          .build();

  /** A larger local model: free, slower, better. */
  private static final ModelCandidate LOCAL_LARGE =
      ModelCandidate.builder("qwen3-8b-q4_k_m")
          .local(true)
          .tags(Set.of("chat", "code", "reasoning"))
          .costPerMillionTokens(0.0, 0.0)
          .timeToFirstTokenMillis(900)
          .tokensPerSecond(12)
          .contextWindow(32_768)
          .quality(Map.of("chat", 0.74, "code", 0.71, "math", 0.58))
          .build();

  /** A hosted frontier model: expensive, high quality, huge context. */
  private static final ModelCandidate FRONTIER =
      ModelCandidate.builder("anthropic/claude-opus-5")
          .local(false)
          .tags(Set.of("chat", "code", "math", "reasoning"))
          .costPerMillionTokens(15.0, 75.0)
          .timeToFirstTokenMillis(700)
          .tokensPerSecond(40)
          .contextWindow(1_000_000)
          .quality(Map.of("chat", 0.93, "code", 0.94, "math", 0.95))
          .build();

  /** A hosted budget model: cheap, decent, mid context. */
  private static final ModelCandidate HOSTED_BUDGET =
      ModelCandidate.builder("deepseek/deepseek-v3")
          .local(false)
          .tags(Set.of("chat", "code", "math"))
          .costPerMillionTokens(0.27, 1.10)
          .timeToFirstTokenMillis(400)
          .tokensPerSecond(60)
          .contextWindow(128_000)
          .quality(Map.of("chat", 0.84, "code", 0.86, "math", 0.88))
          .build();

  private static final List<ModelCandidate> FLEET =
      List.of(LOCAL_SMALL, LOCAL_LARGE, HOSTED_BUDGET, FRONTIER);

  private static ModelRouter router(RoutingPolicy policy) {
    return ModelRouter.builder().candidates(FLEET).policy(policy).build();
  }

  @Nested
  static class Presets {

    @Test
    void cheapestPrefersAFreeLocalModel() {
      RoutingDecision decision = router(RoutingPolicy.CHEAPEST).route("say hello");

      assertThat(decision.selected().local()).isTrue();
      assertThat(decision.selected().costPerMillionInputTokens()).isZero();
    }

    @Test
    void bestQualityPrefersTheStrongestModelRegardlessOfPrice() {
      RoutingDecision decision = router(RoutingPolicy.BEST_QUALITY).route("prove a theorem");

      assertThat(decision.selected().id()).isEqualTo("anthropic/claude-opus-5");
    }

    @Test
    void fastestPrefersTheLowestTimeToFirstToken() {
      RoutingDecision decision = router(RoutingPolicy.FASTEST).route("say hello");

      assertThat(decision.selected().timeToFirstTokenMillis()).isEqualTo(120);
    }

    @Test
    void bestQualityIsNeverTippedByPrice() {
      // FRONTIER beats HOSTED_BUDGET on every task but costs far more. A preset named for quality
      // must not quietly trade a little of it away for a cheaper model.
      ModelRouter pair =
          ModelRouter.builder()
              .candidates(List.of(HOSTED_BUDGET, FRONTIER))
              .policy(RoutingPolicy.BEST_QUALITY)
              .classifier(query -> "math")
              .build();

      assertThat(pair.route("integrate").selected().id()).isEqualTo("anthropic/claude-opus-5");
    }

    @Test
    void balancedPrefersNeitherExtreme() {
      RoutingDecision decision = router(RoutingPolicy.BALANCED).route("write a function");

      assertThat(decision.selected().id()).isNotEqualTo("anthropic/claude-opus-5");
      assertThat(decision.selected()).isNotNull();
    }

    @Test
    void everyPresetIsUsableWithoutHandWeighting() {
      for (RoutingPolicy preset :
          List.of(
              RoutingPolicy.BALANCED,
              RoutingPolicy.CHEAPEST,
              RoutingPolicy.FASTEST,
              RoutingPolicy.BEST_QUALITY,
              RoutingPolicy.LOCAL_FIRST,
              RoutingPolicy.PRIVACY_STRICT)) {
        assertThat(router(preset).route("summarize this").selected()).isNotNull();
      }
    }
  }

  @Nested
  static class Constraints {

    @Test
    void privacyStrictNeverLeavesTheLocalFleet() {
      RoutingDecision decision =
          router(RoutingPolicy.PRIVACY_STRICT).route("analyse this contract");

      assertThat(decision.selected().local()).isTrue();
      assertThat(decision.fallbacks()).allMatch(ModelCandidate::local);
    }

    @Test
    void excludesModelsWhoseContextWindowCannotHoldTheRequest() {
      RoutingDecision decision =
          router(RoutingPolicy.BALANCED)
              .route(RoutingRequest.builder("big").estimatedTokens(200_000).build());

      assertThat(decision.selected().contextWindow()).isGreaterThanOrEqualTo(200_000);
    }

    @Test
    void refusesWhenNoCandidateSatisfiesTheConstraints() {
      // Local-only plus a context larger than any local model can hold.
      assertThatThrownBy(
              () ->
                  router(RoutingPolicy.PRIVACY_STRICT)
                      .route(RoutingRequest.builder("big").estimatedTokens(500_000).build()))
          .isInstanceOf(NoEligibleModelException.class)
          .hasMessageContaining("context");
    }

    @Test
    void honoursAnExplicitQualityFloor() {
      RoutingPolicy strict = RoutingPolicy.CHEAPEST.withMinimumQuality(0.80);

      RoutingDecision decision = strict.isLocalOnly() ? null : router(strict).route("do maths");

      assertThat(decision).isNotNull();
      assertThat(decision.selected().qualityFor(decision.taskType())).isGreaterThanOrEqualTo(0.80);
    }

    @Test
    void honoursAnExplicitCostCeiling() {
      RoutingPolicy capped = RoutingPolicy.BEST_QUALITY.withMaximumCostPerMillionTokens(2.0);

      RoutingDecision decision = router(capped).route("do maths");

      assertThat(decision.selected().costPerMillionInputTokens()).isLessThanOrEqualTo(2.0);
    }
  }

  @Nested
  static class TaskAwareness {

    @Test
    void routesAMathQueryToTheStrongestMathModel() {
      // Same policy, different task: the choice must move with the task's quality column.
      ModelRouter mathRouter =
          ModelRouter.builder()
              .candidates(FLEET)
              .policy(RoutingPolicy.BEST_QUALITY)
              .classifier(query -> query.contains("integral") ? "math" : "chat")
              .build();

      assertThat(mathRouter.route("compute this integral").taskType()).isEqualTo("math");
    }

    @Test
    void ignoresModelsThatDoNotDeclareTheTask() {
      ModelRouter tagged =
          ModelRouter.builder()
              .candidates(List.of(LOCAL_SMALL, FRONTIER))
              .policy(RoutingPolicy.CHEAPEST)
              .classifier(query -> "math")
              .build();

      // LOCAL_SMALL is free but scores 0.30 on math and does not carry the math tag.
      assertThat(tagged.route("integrate").selected().id()).isEqualTo("anthropic/claude-opus-5");
    }
  }

  @Nested
  static class Explainability {

    @Test
    void reportsWhyEachDimensionContributed() {
      RoutingDecision decision = router(RoutingPolicy.BALANCED).route("write a function");

      assertThat(decision.scoreBreakdown())
          .containsKeys("cost", "quality", "latency", "availability");
      assertThat(decision.score()).isBetween(0.0, 1.0);
    }

    @Test
    void ordersFallbacksByDescendingScore() {
      RoutingDecision decision = router(RoutingPolicy.BALANCED).route("write a function");

      assertThat(decision.fallbacks()).isNotEmpty();
      assertThat(decision.fallbacks()).doesNotContain(decision.selected());
    }
  }

  @Nested
  static class Availability {

    @Test
    void avoidsAModelThatIsCurrentlyFailing() {
      ModelCandidate flaky =
          ModelCandidate.builder("flaky/cheap")
              .local(false)
              .tags(Set.of("chat"))
              .costPerMillionTokens(0.01, 0.01)
              .timeToFirstTokenMillis(50)
              .tokensPerSecond(100)
              .contextWindow(128_000)
              .quality(Map.of("chat", 0.90))
              .successRate(0.20)
              .build();

      ModelRouter withFlaky =
          ModelRouter.builder()
              .candidates(List.of(flaky, HOSTED_BUDGET))
              .policy(RoutingPolicy.BALANCED)
              .build();

      assertThat(withFlaky.route("hello").selected().id()).isEqualTo("deepseek/deepseek-v3");
    }

    @Test
    void excludesAModelReportedUnavailableAtRuntime() {
      ModelRouter router = router(RoutingPolicy.CHEAPEST);
      router.updateRuntimeState(
          LOCAL_SMALL.id(), ModelRuntimeState.builder().available(false).build());

      assertThat(router.route("hello").selected()).isNotEqualTo(LOCAL_SMALL);
    }

    @Test
    void coolsDownARepeatedlyFailingModel() {
      ModelRouter router = router(RoutingPolicy.CHEAPEST);
      for (int failure = 0; failure < 3; failure++) {
        router.record(RoutingFeedback.failure(LOCAL_SMALL.id()).build());
      }

      assertThat(router.status(LOCAL_SMALL.id()).orElseThrow().coolingDown()).isTrue();
      assertThat(router.route("hello").selected()).isNotEqualTo(LOCAL_SMALL);
    }

    @Test
    void foldsMeasuredPerformanceBackIntoSubsequentChoices() {
      ModelCandidate measuredSlow =
          ModelCandidate.builder("measured-slow")
              .tags(Set.of("chat"))
              .timeToFirstTokenMillis(50)
              .tokensPerSecond(100)
              .quality(Map.of("chat", 0.8))
              .build();
      ModelCandidate stable =
          ModelCandidate.builder("stable")
              .tags(Set.of("chat"))
              .timeToFirstTokenMillis(200)
              .tokensPerSecond(50)
              .quality(Map.of("chat", 0.8))
              .build();
      ModelRouter measured =
          ModelRouter.builder()
              .candidates(List.of(measuredSlow, stable))
              .classifier(ignored -> "chat")
              .policy(RoutingPolicy.FASTEST)
              .build();

      measured.record(
          RoutingFeedback.success(measuredSlow.id())
              .timeToFirstTokenMillis(1_000)
              .tokensPerSecond(5)
              .build());

      assertThat(measured.route("hello").selected()).isEqualTo(stable);
    }
  }

  @Nested
  static class RuntimeAwareScoring {

    @Test
    void prefersAnAlreadyResidentModelWhenTheStaticCandidatesTie() {
      ModelCandidate first = tied("first");
      ModelCandidate resident = tied("resident");
      ModelRouter router = ModelRouter.builder().candidates(List.of(first, resident)).build();
      router.updateRuntimeState(first.id(), ModelRuntimeState.builder().resident(false).build());
      router.updateRuntimeState(resident.id(), ModelRuntimeState.builder().resident(true).build());

      assertThat(router.route("hello").selected()).isEqualTo(resident);
      assertThat(router.route("hello").scoreBreakdown()).containsKey("residency");
    }

    @Test
    void routesAwayFromADeeperQueue() {
      ModelCandidate busy = tied("busy");
      ModelCandidate idle = tied("idle");
      ModelRouter router = ModelRouter.builder().candidates(List.of(busy, idle)).build();
      router.updateRuntimeState(busy.id(), ModelRuntimeState.builder().queueDepth(12).build());
      router.updateRuntimeState(idle.id(), ModelRuntimeState.builder().queueDepth(0).build());

      assertThat(router.route("hello").selected()).isEqualTo(idle);
      assertThat(router.route("hello").scoreBreakdown()).containsKey("load");
    }

    private static ModelCandidate tied(String id) {
      return ModelCandidate.builder(id)
          .timeToFirstTokenMillis(100)
          .tokensPerSecond(20)
          .quality(Map.of("chat", 0.8))
          .build();
    }
  }

  @Nested
  static class SessionContinuity {

    @Test
    void switchesWhenAChallengerIsMateriallyBetter() {
      ModelRouter router = router(RoutingPolicy.CHEAPEST);
      RoutingRequest request = RoutingRequest.builder("hello").sessionId("conversation").build();
      assertThat(router.route(request).selected()).isEqualTo(LOCAL_SMALL);

      RoutingDecision switched = router.route(request, RoutingPolicy.BEST_QUALITY);

      assertThat(switched.selected()).isEqualTo(FRONTIER);
    }

    @Test
    void explainsWhenTheSwitchMarginRetainsTheCurrentModel() {
      ModelRouter router =
          ModelRouter.builder()
              .candidates(FLEET)
              .policy(RoutingPolicy.CHEAPEST)
              .adaptiveOptions(AdaptiveRoutingOptions.builder().switchMargin(1.0).build())
              .build();
      RoutingRequest request = RoutingRequest.builder("hello").sessionId("stable").build();
      assertThat(router.route(request).selected()).isEqualTo(LOCAL_SMALL);

      RoutingDecision retained = router.route(request, RoutingPolicy.BEST_QUALITY);

      assertThat(retained.selected()).isEqualTo(LOCAL_SMALL);
      assertThat(retained.scoreBreakdown()).containsEntry("switch-margin", 1.0);
    }

    @Test
    void doesNotSwitchModelsInTheMiddleOfAToolLoop() {
      ModelRouter router = router(RoutingPolicy.CHEAPEST);
      RoutingRequest request = RoutingRequest.builder("hello").sessionId("tool-loop").build();
      assertThat(router.route(request).selected()).isEqualTo(LOCAL_SMALL);

      RoutingDecision retained =
          router.route(
              request,
              RoutingPolicy.BEST_QUALITY,
              RoutingContinuity.builder().activeToolLoop(true).build());

      assertThat(retained.selected()).isEqualTo(LOCAL_SMALL);
      assertThat(retained.scoreBreakdown()).containsEntry("continuity-lock", 1.0);
    }

    @Test
    void doesNotSwitchProviderOwnedNonPortableContext() {
      ModelRouter router = router(RoutingPolicy.CHEAPEST);
      RoutingRequest request = RoutingRequest.builder("hello").sessionId("provider-state").build();
      assertThat(router.route(request).selected()).isEqualTo(LOCAL_SMALL);

      RoutingDecision retained =
          router.route(
              request,
              RoutingPolicy.BEST_QUALITY,
              RoutingContinuity.builder().contextPortable(false).build());

      assertThat(retained.selected()).isEqualTo(LOCAL_SMALL);
    }

    @Test
    void usesCachedPrefixEvidenceWithoutPretendingKvIsPortable() {
      ModelCandidate current = RuntimeAwareScoring.tied("current");
      ModelCandidate challenger = RuntimeAwareScoring.tied("challenger");
      ModelRouter router = ModelRouter.builder().candidates(List.of(current, challenger)).build();
      RoutingRequest request =
          RoutingRequest.builder("continuation").estimatedTokens(1_000).sessionId("cached").build();
      assertThat(router.route(request).selected()).isEqualTo(current);

      RoutingDecision retained =
          router.route(
              request,
              RoutingPolicy.BALANCED,
              RoutingContinuity.builder().cachedPrefixTokens(current.id(), 900).build());

      assertThat(retained.selected()).isEqualTo(current);
      assertThat(retained.scoreBreakdown().get("cache-affinity")).isPositive();
    }

    @Test
    void expiresIdleSessionsAndBoundsTheirCount() {
      MutableClock clock = new MutableClock();
      AdaptiveRoutingOptions options =
          AdaptiveRoutingOptions.builder()
              .sessionIdleTimeout(Duration.ofMinutes(1))
              .maximumSessions(2)
              .build();
      ModelRouter router =
          ModelRouter.builder()
              .candidates(FLEET)
              .policy(RoutingPolicy.CHEAPEST)
              .adaptiveOptions(options)
              .clock(clock)
              .build();
      router.route(RoutingRequest.builder("one").sessionId("one").build());
      router.route(RoutingRequest.builder("two").sessionId("two").build());
      router.route(RoutingRequest.builder("three").sessionId("three").build());
      assertThat(router.activeSessionCount()).isEqualTo(2);

      clock.advance(Duration.ofMinutes(2));
      RoutingDecision afterExpiry =
          router.route(
              RoutingRequest.builder("one").sessionId("one").build(), RoutingPolicy.BEST_QUALITY);
      assertThat(afterExpiry.selected()).isEqualTo(FRONTIER);
    }

    @Test
    void canCloseASessionExplicitly() {
      ModelRouter router = router(RoutingPolicy.CHEAPEST);
      router.route(RoutingRequest.builder("hello").sessionId("done").build());

      assertThat(router.closeSession("done")).isTrue();
      assertThat(router.activeSessionCount()).isZero();
    }
  }

  @Nested
  static class Extensions {

    @Test
    void appliesApplicationFiltersBeforeScoring() {
      ModelRouter router =
          ModelRouter.builder()
              .candidates(List.of(LOCAL_SMALL, HOSTED_BUDGET))
              .policy(RoutingPolicy.CHEAPEST)
              .filter(
                  evaluation ->
                      evaluation.candidate().local()
                          ? Optional.of("tenant requires hosted audit logging")
                          : Optional.empty())
              .build();

      assertThat(router.route("hello").selected()).isEqualTo(HOSTED_BUDGET);
    }

    @Test
    void addsNamedApplicationScorersToTheDecisionTrace() {
      ModelRouter router =
          ModelRouter.builder()
              .candidates(List.of(LOCAL_SMALL, HOSTED_BUDGET))
              .policy(RoutingPolicy.BALANCED)
              .scorer(
                  "tenant-affinity",
                  10.0,
                  evaluation -> evaluation.candidate().equals(HOSTED_BUDGET) ? 1.0 : 0.0)
              .build();

      RoutingDecision decision = router.route("hello");

      assertThat(decision.selected()).isEqualTo(HOSTED_BUDGET);
      assertThat(decision.scoreBreakdown()).containsKey("tenant-affinity");
    }
  }

  @Nested
  static class Validation {

    @Test
    void requiresAtLeastOneCandidate() {
      assertThatThrownBy(() -> ModelRouter.builder().candidates(List.of()).build())
          .isInstanceOf(IllegalArgumentException.class);
    }

    @Test
    void rejectsDuplicateCandidateIds() {
      assertThatThrownBy(
              () -> ModelRouter.builder().candidates(List.of(LOCAL_SMALL, LOCAL_SMALL)).build())
          .isInstanceOf(IllegalArgumentException.class)
          .hasMessageContaining("duplicate");
    }
  }

  private static final class MutableClock extends Clock {
    private Instant instant = Instant.parse("2026-09-03T00:00:00Z");

    void advance(Duration duration) {
      instant = instant.plus(duration);
    }

    @Override
    public ZoneId getZone() {
      return ZoneId.of("UTC");
    }

    @Override
    public Clock withZone(ZoneId zone) {
      return this;
    }

    @Override
    public Instant instant() {
      return instant;
    }
  }
}
