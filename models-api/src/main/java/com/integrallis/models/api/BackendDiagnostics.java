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

import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Objects;
import java.util.Optional;
import java.util.Set;

/** Immutable description of the runtime plan selected by an inference backend. */
public record BackendDiagnostics(
    String backend,
    String planVersion,
    Map<String, String> environment,
    List<OptimizationDecision> optimizations) {

  public BackendDiagnostics {
    backend = requireText(backend, "backend");
    planVersion = requireText(planVersion, "planVersion");
    environment = Map.copyOf(Objects.requireNonNull(environment, "environment"));
    optimizations = List.copyOf(Objects.requireNonNull(optimizations, "optimizations"));
    Set<String> ids = new HashSet<>();
    for (OptimizationDecision optimization : optimizations) {
      Objects.requireNonNull(optimization, "optimization");
      if (!ids.add(optimization.id())) {
        throw new IllegalArgumentException("duplicate optimization id: " + optimization.id());
      }
    }
  }

  /** Finds the decision for one stable optimization identifier. */
  public Optional<OptimizationDecision> optimization(String id) {
    Objects.requireNonNull(id, "id");
    return optimizations.stream().filter(value -> value.id().equals(id)).findFirst();
  }

  /** Returns diagnostics for a backend that does not expose an execution plan. */
  public static BackendDiagnostics unavailable(String backend) {
    return new BackendDiagnostics(backend, "unavailable", Map.of(), List.of());
  }

  private static String requireText(String value, String name) {
    if (value == null || value.isBlank()) {
      throw new IllegalArgumentException(name + " must not be blank");
    }
    return value.trim();
  }
}
