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
package com.integrallis.models.api.catalog;

import java.util.Map;
import java.util.Objects;
import java.util.Set;

/**
 * A model a catalog knows about, described in measurements rather than opinions.
 *
 * <p>Deliberately not a routing type. A catalog reports what a model <em>is</em> — how fast it ran
 * here, what it scored, what it costs — and something downstream decides what that is worth.
 * Keeping the two apart is what lets a catalog be written without knowing how anyone routes.
 *
 * @param id stable identifier, unique within the providing catalog
 * @param local whether the weights run on this machine
 * @param tags task labels this model is suitable for, e.g. {@code code}, {@code sql}
 * @param contextWindow maximum tokens the model accepts
 * @param costPerMillionInputTokens currency-neutral input price; zero for local models
 * @param costPerMillionOutputTokens currency-neutral output price; zero for local models
 * @param performance measured throughput on this machine, or null when nothing was measured for
 *     this hardware
 * @param quality per-task scores in [0, 1], keyed by the same labels as {@code tags}
 * @param successRate share of requests that completed, in [0, 1]
 */
public record DiscoveredModel(
    String id,
    boolean local,
    Set<String> tags,
    int contextWindow,
    double costPerMillionInputTokens,
    double costPerMillionOutputTokens,
    Performance performance,
    Map<String, Double> quality,
    double successRate) {

  /** Validates and defensively copies. */
  public DiscoveredModel {
    Objects.requireNonNull(id, "id");
    if (id.isBlank()) {
      throw new IllegalArgumentException("id must not be blank");
    }
    tags = Set.copyOf(Objects.requireNonNull(tags, "tags"));
    quality = Map.copyOf(Objects.requireNonNull(quality, "quality"));
    if (contextWindow < 1) {
      throw new IllegalArgumentException("contextWindow must be positive: " + contextWindow);
    }
    if (costPerMillionInputTokens < 0 || costPerMillionOutputTokens < 0) {
      throw new IllegalArgumentException("costs must not be negative");
    }
    if (successRate < 0.0 || successRate > 1.0) {
      throw new IllegalArgumentException("successRate must be within [0, 1]: " + successRate);
    }
  }

  /**
   * What a model actually did on this machine.
   *
   * <p>Absent rather than estimated when no profile matches the current hardware. These numbers
   * vary by CPU, core count and JVM, so a figure carried over from another machine is not a
   * measurement — and a router scoring on it would prefer the wrong model with complete confidence.
   *
   * @param timeToFirstTokenMillis latency before the first token
   * @param tokensPerSecond sustained generation rate
   */
  public record Performance(long timeToFirstTokenMillis, double tokensPerSecond) {

    /** Validates. */
    public Performance {
      if (timeToFirstTokenMillis < 0) {
        throw new IllegalArgumentException("timeToFirstTokenMillis must not be negative");
      }
      if (!(tokensPerSecond > 0)) {
        throw new IllegalArgumentException("tokensPerSecond must be positive: " + tokensPerSecond);
      }
    }
  }
}
