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
 */
public record BackendConfiguration(
    Map<String, String> environment,
    Map<String, String> recommendations,
    List<OptimizationDecision> optimizations) {

  public BackendConfiguration {
    environment = Map.copyOf(Objects.requireNonNull(environment, "environment"));
    recommendations = Map.copyOf(Objects.requireNonNull(recommendations, "recommendations"));
    optimizations = List.copyOf(Objects.requireNonNull(optimizations, "optimizations"));
    optimizations.forEach(value -> Objects.requireNonNull(value, "optimization"));
  }

  /** Returns a configuration with no external recommendations or diagnostics. */
  public static BackendConfiguration empty() {
    return new BackendConfiguration(Map.of(), Map.of(), List.of());
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
