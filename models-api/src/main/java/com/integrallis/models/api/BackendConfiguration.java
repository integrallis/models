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
package com.integrallis.models.api;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * Registry-neutral recommendations and evidence attached to one exact backend load.
 *
 * <p>Catalogs such as ModelJars can validate artifact and runtime identity, then pass the resulting
 * configuration to a Models backend without making the backend depend on that catalog.
 *
 * <p>{@code recommendations} and {@code overrides} are two different claims and resolve at
 * different strengths. A recommendation is what the ModelJar says about itself and loses to a
 * deployment-wide system property. An override is what the caller asked for on this one load, and
 * beats the property: an application that sizes one runtime for its own sequences cannot say so
 * through a JVM-global property without changing every other runtime in the process, which is the
 * complaint #357 was filed about. {@code environment} is neither -- it is diagnostics, merged into
 * {@link BackendDiagnostics} by {@link #enrich}, and it reaches no planning decision.
 */
public record BackendConfiguration(
    Map<String, String> environment,
    Map<String, String> recommendations,
    List<OptimizationDecision> optimizations,
    Map<String, String> overrides) {

  public BackendConfiguration {
    environment = Map.copyOf(Objects.requireNonNull(environment, "environment"));
    recommendations = Map.copyOf(Objects.requireNonNull(recommendations, "recommendations"));
    optimizations = List.copyOf(Objects.requireNonNull(optimizations, "optimizations"));
    optimizations.forEach(value -> Objects.requireNonNull(value, "optimization"));
    overrides = Map.copyOf(Objects.requireNonNull(overrides, "overrides"));
  }

  /**
   * Creates a configuration that supplies no caller overrides.
   *
   * @param environment provenance and diagnostics for this load
   * @param recommendations the catalog's settings for this model, weakest of the three tiers
   * @param optimizations decisions the catalog made about this load
   * @deprecated use the canonical constructor; this exists so adding {@code overrides} does not
   *     break callers compiled against the three-component record, and it will go in the next
   *     major.
   */
  @Deprecated(since = "0.5.15", forRemoval = true)
  public BackendConfiguration(
      Map<String, String> environment,
      Map<String, String> recommendations,
      List<OptimizationDecision> optimizations) {
    this(environment, recommendations, optimizations, Map.of());
  }

  /** Returns a configuration with no external recommendations or diagnostics. */
  public static BackendConfiguration empty() {
    return new BackendConfiguration(Map.of(), Map.of(), List.of(), Map.of());
  }

  /** Adds this configuration's provenance and decisions to backend-native diagnostics. */
  public BackendDiagnostics enrich(BackendDiagnostics diagnostics) {
    Objects.requireNonNull(diagnostics, "diagnostics");
    Map<String, String> combinedEnvironment = new LinkedHashMap<>(diagnostics.environment());
    combinedEnvironment.putAll(environment);
    List<OptimizationDecision> combinedOptimizations = new ArrayList<>(diagnostics.optimizations());
    combinedOptimizations.addAll(optimizations);
    return new BackendDiagnostics(
        diagnostics.backend(),
        diagnostics.planVersion(),
        combinedEnvironment,
        combinedOptimizations);
  }
}
