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

import java.util.List;
import java.util.Map;
import java.util.Objects;

/**
 * The chosen model, the ranked alternatives, and why.
 *
 * <p>The breakdown is part of the contract rather than a debugging aid. A router that cannot say
 * why it escalated to a frontier model is one nobody can tune, and tuning is the whole point of
 * exposing weights.
 *
 * @param selected the model to call
 * @param fallbacks remaining eligible models, best first
 * @param taskType how the query was classified, null when unknown
 * @param score the winner's overall score in [0, 1]
 * @param scoreBreakdown weighted contribution of each dimension to the winner's score
 */
public record RoutingDecision(
    ModelCandidate selected,
    List<ModelCandidate> fallbacks,
    String taskType,
    double score,
    Map<String, Double> scoreBreakdown) {

  /** Validates and defensively copies. */
  public RoutingDecision {
    Objects.requireNonNull(selected, "selected");
    fallbacks = List.copyOf(Objects.requireNonNull(fallbacks, "fallbacks"));
    scoreBreakdown = Map.copyOf(Objects.requireNonNull(scoreBreakdown, "scoreBreakdown"));
  }

  /**
   * Returns the chosen model's identifier.
   *
   * @return the id an application maps back to its client
   */
  public String modelId() {
    return selected.id();
  }

  /**
   * Renders the decision for logs.
   *
   * @return a one-line explanation
   */
  public String explain() {
    StringBuilder text = new StringBuilder();
    text.append(selected.id());
    if (taskType != null) {
      text.append(" for ").append(taskType);
    }
    text.append(String.format(" (score %.3f", score));
    scoreBreakdown.entrySet().stream()
        .sorted(Map.Entry.comparingByKey())
        .forEach(
            entry -> text.append(String.format(", %s %.3f", entry.getKey(), entry.getValue())));
    return text.append(')').toString();
  }
}
